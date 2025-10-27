"""Utility script to inspect cuBVH GPU memory footprint across devices."""

from __future__ import annotations

import argparse
import statistics
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

import cubvh


def _bytes_to_mib(num_bytes: int) -> float:
    """Convert a byte count into mebibytes."""

    return num_bytes / (1024**2)


def _cube_mesh() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a simple cube mesh as (vertices, triangles)."""

    vertices = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    triangles = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=torch.int32,
    )
    return vertices, triangles


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-bvhs", type=int, default=100, help="How many CPU BVHs to build before deploying.")
    parser.add_argument(
        "--device-ids",
        type=int,
        nargs="*",
        default=None,
        help="CUDA device ids to target; defaults to all visible devices.",
    )
    parser.add_argument(
        "--uniform-tolerance",
        type=float,
        default=5.0,
        help="Tolerance in MiB for the GPU usage delta when deciding uniformity.",
    )
    parser.add_argument(
        "--scale-step",
        type=float,
        default=0.05,
        help="Scale increment applied to each successive cube before building the BVH.",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="Optional random jitter amplitude applied to vertices before building the BVH.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for vertex jitter.")
    parser.add_argument("--verbose", action="store_true", help="Print per-transfer measurements.")
    return parser.parse_args(argv)


def _validate_devices(device_ids: Iterable[int]) -> List[int]:
    available = list(range(torch.cuda.device_count()))
    devices = list(device_ids)
    missing = sorted(set(devices) - set(available))
    if missing:
        raise SystemExit(f"Requested CUDA devices not available: {missing}; visible devices: {available}")
    return devices


def _measure_usage(device: int) -> int:
    """Return current used memory on the given device in bytes."""

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return total_bytes - free_bytes


def _build_cpu_bvhs(num_bvhs: int, scale_step: float, jitter: float, seed: int) -> List[cubvh.cuBVH]:
    # 1
    base_vertices_, base_triangles_ = _cube_mesh()
    vert_num = base_vertices_.shape[0]
    base_vertices, base_triangles = torch.empty((0, 3)), torch.empty((0, 3), dtype=torch.int32)
    for i in range(1000):
        translation = torch.empty(1, 3).uniform_(-5.0, 5.0)
        base_vertices = torch.cat([base_vertices, base_vertices_ + translation], dim=0)
        base_triangles = torch.cat(
            [base_triangles, base_triangles_ + vert_num * i], dim=0
        )
    # 2
    base_vertices_, base_triangles_ = base_vertices, base_triangles
    vert_num = base_vertices_.shape[0]
    base_vertices, base_triangles = torch.empty((0, 3)), torch.empty((0, 3), dtype=torch.int32)
    for i in range(100):
        translation = torch.empty(1, 3).uniform_(-50.0, 50.0)
        base_vertices = torch.cat([base_vertices, base_vertices_ + translation], dim=0)
        base_triangles = torch.cat(
            [base_triangles, base_triangles_ + vert_num * i], dim=0
        )
    print(f"Built base mesh with {base_vertices.shape[0]} vertices and {base_triangles.shape[0]} triangles.")
    torch.manual_seed(seed)

    cpu_bvhs: List[cubvh.cuBVH] = []
    for bvh_idx in range(num_bvhs):
        scale = 1.0 + bvh_idx * scale_step
        vertices = base_vertices * scale
        if jitter:
            vertices = vertices + torch.empty_like(vertices).uniform_(-jitter, jitter)
        cpu_bvhs.append(cubvh.cuBVH(vertices=vertices.contiguous(), triangles=base_triangles, device="cpu"))
    return cpu_bvhs


def _distribute_bvhs(
    cpu_bvhs: Sequence[cubvh.cuBVH],
    devices: Sequence[int],
    baseline_usage: Dict[int, int],
    verbose: bool,
) -> Dict[int, int]:
    per_device_usage = {device: baseline_usage[device] for device in devices}
    bvh_handles = {device: [] for device in devices}  # keep GPU handles alive for accurate tracking

    for bvh_idx, bvh_cpu in enumerate(cpu_bvhs):
        device = devices[bvh_idx % len(devices)]
        torch.cuda.synchronize(device)
        bvh_gpu = bvh_cpu.to(f"cuda:{device}")
        bvh_handles[device].append(bvh_gpu)
        torch.cuda.synchronize(device)
        current_usage = _measure_usage(device)
        per_device_usage[device] = current_usage
        if verbose:
            delta_mib = _bytes_to_mib(current_usage - baseline_usage[device])
            print(f"[GPU {device}] Loaded BVH {bvh_idx:02d}: +{delta_mib:.2f} MiB over baseline")

    return {device: usage - baseline_usage[device] for device, usage in per_device_usage.items()}


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to evaluate GPU memory usage.")

    if args.device_ids is None:
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = _validate_devices(args.device_ids)

    if not devices:
        raise SystemExit("No CUDA devices available for measurement.")

    cpu_bvhs = _build_cpu_bvhs(args.num_bvhs, args.scale_step, args.jitter, args.seed)

    baseline_usage = {device: _measure_usage(device) for device in devices}
    for device in devices:
        usage_mib = _bytes_to_mib(baseline_usage[device])
        print(f"[GPU {device}] Baseline usage: {usage_mib:.2f} MiB")

    deltas = _distribute_bvhs(cpu_bvhs, devices, baseline_usage, args.verbose)
    import time
    time.sleep(5)

    summary: Dict[int, float] = {}
    for device, delta_bytes in deltas.items():
        delta_mib = _bytes_to_mib(delta_bytes)
        summary[device] = delta_mib
        print(f"[GPU {device}] Total BVH delta: {delta_mib:.2f} MiB (device: {torch.cuda.get_device_name(device)})")

    time.sleep(1)

    if summary:
        values = list(summary.values())
        max_delta = max(values)
        min_delta = min(values)
        std_dev = statistics.pstdev(values) if len(values) > 1 else 0.0
        spread = max_delta - min_delta
        uniform = spread <= args.uniform_tolerance
        print(
            "Uniform distribution: {result} (spread={spread:.2f} MiB, std={std_dev:.2f} MiB, tolerance={tol:.2f} MiB)".format(
                result="yes" if uniform else "no",
                spread=spread,
                std_dev=std_dev,
                tol=args.uniform_tolerance,
            )
        )


if __name__ == "__main__":
    main()
