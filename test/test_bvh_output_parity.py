import math
import os
import random
import tempfile
from typing import Tuple

import numpy as np
import torch

import cubvh
import cubvh_origin


def _seed_everything(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_noisy_stack(num_cubes: int = 3, noise: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Generate multiple cubes stacked along z with random jitter."""
    base_cube = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.int32)

    vertices = []
    triangles = []
    for i in range(num_cubes):
        shift = np.array([
            (i % 2) * 0.5,
            ((i + 1) % 2) * 0.5,
            i * 0.9,
        ], dtype=np.float32)

        jitter = np.random.normal(scale=noise, size=base_cube.shape).astype(np.float32)
        verts = base_cube + shift + jitter
        base_idx = len(vertices)
        vertices.extend(verts)
        triangles.extend((faces + base_idx).tolist())

    return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def _sample_points(n: int = 8192, scale: float = 3.0) -> torch.Tensor:
    pts = torch.empty(n, 3).uniform_(-scale, scale)
    return pts


def _run_bvh(module, vertices, triangles, positions, ray_origins, ray_dirs, device):
    if module is cubvh_origin:
        vertices_np = vertices.detach().cpu().numpy() if torch.is_tensor(vertices) else vertices
        triangles_np = triangles.detach().cpu().numpy() if torch.is_tensor(triangles) else triangles
        bvh = module.cuBVH(vertices=vertices_np, triangles=triangles_np)
    else:
        bvh = module.cuBVH(vertices=vertices, triangles=triangles, device=device)

    unsigned = bvh.unsigned_distance(positions, return_uvw=True)
    signed = bvh.signed_distance(positions, return_uvw=True, mode='watertight')
    ray = bvh.ray_trace(ray_origins, ray_dirs)

    return {
        "unsigned": unsigned,
        "signed": signed,
        "ray": ray,
    }


def _compare_tuples(lhs, rhs, name: str, eps: float = 1e-4):
    for idx, (l, r) in enumerate(zip(lhs, rhs)):
        if l is None and r is None:
            continue
        if l is None or r is None:
            raise AssertionError(f"Mismatch in {name}[{idx}]: one result is None")
        if l.dtype.is_floating_point:
            if l.device != r.device:
                r = r.to(l.device)
            if not torch.allclose(l, r, atol=eps, rtol=eps):
                max_diff = (l - r).abs().max().item()
                raise AssertionError(f"{name}[{idx}] mismatch: max diff {max_diff}")
        else:
            if l.device != r.device:
                r = r.to(l.device)
            if not torch.equal(l, r):
                raise AssertionError(f"{name}[{idx}] mismatch: tensors differ")


def main():
    _seed_everything()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to compare BVH outputs.")

    device = torch.device("cuda")

    vertices, triangles = _make_noisy_stack(num_cubes=4, noise=0.08)
    vertices_t = torch.from_numpy(vertices)
    triangles_t = torch.from_numpy(triangles)

    sample_positions = _sample_points().to(device=device, dtype=torch.float32)
    ray_origins = torch.randn_like(sample_positions)
    ray_dirs = torch.randn_like(sample_positions)
    ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    current_res = _run_bvh(cubvh, vertices_t, triangles_t, sample_positions, ray_origins, ray_dirs, device)
    origin_res = _run_bvh(cubvh_origin, vertices_t, triangles_t, sample_positions, ray_origins, ray_dirs, device)

    _compare_tuples(current_res["unsigned"], origin_res["unsigned"], "unsigned_distance")
    _compare_tuples(current_res["signed"], origin_res["signed"], "signed_distance")
    _compare_tuples(current_res["ray"], origin_res["ray"], "ray_trace")

    print("cuBVH parity test passed: current version matches cubvh_origin outputs.")


if __name__ == "__main__":
    main()
