import argparse
import gc
import math
import os
import tempfile
import time

import numpy as np
import torch
import trimesh

import cubvh


def _load_mesh(path, normalize):
    mesh = trimesh.load(path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.faces, dtype=np.int32)

    if normalize:
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        center = 0.5 * (mins + maxs)
        extent = (maxs - mins).max()
        scale = max(extent, 1e-6)
        vertices = (vertices - center) / (0.5 * scale)

    return (
        torch.from_numpy(vertices.copy()),
        torch.from_numpy(triangles.copy()),
    )


def _synchronize(device):
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.synchronize(index)


def _maybe_empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _time_many(fn, *, device, repeats, warmup):
    durations = []

    for _ in range(warmup):
        result = fn()
        del result
        _synchronize(device)
        gc.collect()
        _maybe_empty_cache(device)

    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter()
        result = fn()
        _synchronize(device)
        end = time.perf_counter()
        durations.append(end - start)
        del result
        gc.collect()
        _maybe_empty_cache(device)

    return durations


def _summarize(label, durations):
    mean = sum(durations) / len(durations)
    if len(durations) > 1:
        variance = sum((d - mean) ** 2 for d in durations) / (len(durations) - 1)
        std = math.sqrt(max(variance, 0.0))
    else:
        std = 0.0
    print(f"{label:<32} mean={mean:.4f}s  std={std:.4f}s  min={min(durations):.4f}s  max={max(durations):.4f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark cuBVH build vs restoration")
    parser.add_argument("--mesh", default=os.path.join("test", "girl.obj"), help="Path to OBJ mesh")
    parser.add_argument("--device", default="cuda", help="Torch device to target (default: cuda)")
    parser.add_argument("--repeats", type=int, default=10, help="Number of timed runs per scenario")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs per scenario")
    parser.add_argument("--normalize", action="store_true", help="Normalize mesh into unit box before building")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available but was requested")

    vertices, triangles = _load_mesh(args.mesh, args.normalize)

    # duplicate the mesh 10 times with random translations
    copies = 100
    orig_v = vertices
    orig_t = triangles
    V = orig_v.shape[0]

    # compute a scale from the mesh bbox so copies are spaced out
    mins = orig_v.min(dim=0).values
    maxs = orig_v.max(dim=0).values
    extent = (maxs - mins).max().item()
    if extent == 0.0:
        extent = 1.0

    # reproducible random translations
    rng = torch.Generator()
    rng.manual_seed(42)
    translations = torch.randn((copies, 3), dtype=orig_v.dtype, generator=rng) * (extent * 3.0)

    v_list = []
    t_list = []
    for i in range(copies):
        tvec = translations[i]
        v_list.append(orig_v + tvec)
        t_list.append(orig_t + i * V)

    vertices = torch.cat(v_list, dim=0)
    triangles = torch.cat(t_list, dim=0)
    print(f"Loaded mesh: {args.mesh}")
    print(f"Vertices: {vertices.shape[0]}  Triangles: {triangles.shape[0]}")

    baseline = cubvh.cuBVH(vertices=vertices, triangles=triangles, device=device)
    state = baseline.export_state()
    tmpdir = tempfile.TemporaryDirectory()
    ckpt_path = os.path.join(tmpdir.name, "bvh.pt")
    torch.save(baseline, ckpt_path)
    del baseline
    _maybe_empty_cache(device)

    def build_from_mesh():
        return cubvh.cuBVH(vertices=vertices, triangles=triangles, device=device)

    def restore_from_state():
        return cubvh.cuBVH(state=state, device=device)

    def restore_from_torch_checkpoint():
        restored = torch.load(ckpt_path)
        return restored.to(device)

    build_times = _time_many(build_from_mesh, device=device, repeats=args.repeats, warmup=args.warmup)
    state_times = _time_many(restore_from_state, device=device, repeats=args.repeats, warmup=args.warmup)
    ckpt_times = _time_many(restore_from_torch_checkpoint, device=device, repeats=args.repeats, warmup=args.warmup)

    print("\nResults (over", args.repeats, "runs):")
    _summarize("Build from mesh", build_times)
    _summarize("Restore from export_state", state_times)
    _summarize("torch.load + to(device)", ckpt_times)

    tmpdir.cleanup()


if __name__ == "__main__":
    main()
