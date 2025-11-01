"""Stress-test cubvh memory usage with AniGenSparseFeat2Skeleton.

This test spins up one Python thread per CUDA device. Each thread builds its own
`DataLoader` (with multiple workers) for the AniGen sparse-feat-to-skeleton
dataset, instantiates cuBVH instances from the mesh data, and queries unsigned
distances on random point clouds. After a short run we compare GPU memory
profiles to make sure that (1) memory usage is balanced across devices and
(2) GPU 0 does not accumulate memory across iterations.

To keep the runtime reasonable the number of batches and points per batch are
kept modest. Set the environment variable `ANIGEN_DATASET_ROOTS` (comma-separated
list) to point at the dataset root(s) that contain `metadata.csv`, or leave it
unset to fall back to `/mnt/pfs/data/huangyihua/AniGen`.
"""

import argparse
import os
import sys
import threading
import unittest
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch.multiprocessing as mp

try:  # pragma: no cover - pytest might not be present in lint context
    import pytest  # type: ignore
except ImportError:  # pragma: no cover - fallback for tooling without pytest
    pytest = None  # type: ignore

import torch
from torch.utils.data import DataLoader

from cubvh import cuBVH
from dataset import AniGenSparseFeat2Skeleton

try:  # pragma: no cover - optional dependency for NVML snapshots
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover
    pynvml = None  # type: ignore


_DEFAULT_ROOT = os.environ.get(
	"ANIGEN_DATASET_ROOTS",
	os.environ.get("ANIGEN_DATA_ROOT", "/mnt/pfs/data/huangyihua/AniGen"),
)


@dataclass
class _MemoryTestConfig:

	roots: str
	dataset_kwargs: Dict[str, object]
	batch_size: int = 1
	loader_workers: int = 2
	steps_per_thread: int = 3
	points_per_sample: int = 131_072
	imbalance_tolerance: float = 0.2  # 20% relative difference allowed
	memory_growth_tolerance_bytes: int = 64 * 1024 * 1024  # 64 MiB
	max_threads: int | None = None


def _skip(reason: str) -> None:
	if pytest is not None:
		pytest.skip(reason)
	else:
		raise unittest.SkipTest(reason)


def _fail(message: str) -> None:
	if pytest is not None:
		pytest.fail(message)
	else:
		raise AssertionError(message)


def _bytes_to_megabytes(num_bytes: int) -> float:
	return num_bytes / (1024 ** 2)


_NVML_INIT_LOCK = threading.Lock()
_NVML_HANDLES: Dict[int, object] = {}


def _ensure_nvml_initialized() -> bool:
    if pynvml is None:
        return False
    if _NVML_HANDLES:
        return True
    with _NVML_INIT_LOCK:
        if _NVML_HANDLES:
            return True
        try:
            pynvml.nvmlInit()
        except Exception:
            return False
        for device_index in range(torch.cuda.device_count()):
            try:
                _NVML_HANDLES[device_index] = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception:
                _NVML_HANDLES.clear()
                return False
    return True


def _nvml_memory_snapshot(device_index: int) -> Optional[Tuple[int, int]]:
    if not _ensure_nvml_initialized():
        return None
    handle = _NVML_HANDLES.get(device_index)
    if handle is None:
        return None
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    except Exception:
        return None
    return info.used, info.total


def _global_memory_snapshot(device: torch.device) -> Optional[Tuple[int, int]]:
    """Return (used_bytes, total_bytes) for the whole device via CUDA driver."""
    if device.type != "cuda":
        return None
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    nvml_snapshot = _nvml_memory_snapshot(device_index)
    if nvml_snapshot is not None:
        return nvml_snapshot
    mem_get_info = getattr(torch.cuda, "mem_get_info", None)
    if mem_get_info is None:
        return None
    free_bytes, total_bytes = mem_get_info(device_index)
    return total_bytes - free_bytes, total_bytes


def _check_environment(roots: str) -> Tuple[bool, str]:
	if not torch.cuda.is_available():
		return False, "CUDA is required for the cuBVH memory probe."

	first_root = roots.split(",")[0]
	if not os.path.isdir(first_root) or not os.path.isfile(os.path.join(first_root, "metadata.csv")):
		return False, "AniGen dataset metadata not found. Set ANIGEN_DATASET_ROOTS to a valid dataset root."

	return True, ""


def _process_batch(batch: Dict[str, object], device: torch.device, points_per_sample: int) -> None:
    bvhs = batch["cubvh"] if 'cubvh' in batch else []

    for bvh in bvhs:
        bvh = bvh.to(device)
        points: torch.Tensor | None = None
        distances: torch.Tensor | None = None
        try:
            points = torch.randn(points_per_sample, 3, device=device, dtype=torch.float32)
            distances, _, _ = bvh.unsigned_distance(points, return_uvw=True)
            # Force materialization so CUDA work finishes before the object is freed.
            distances.mean().item()
        finally:
            bvh.to("cpu")
            if distances is not None:
                del distances
            if points is not None:
                del points
            del bvh

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def _memory_probe_worker(
    device_index: int,
    config: _MemoryTestConfig,
    results: Dict[int, Dict[str, object]],
    lock: threading.Lock,
) -> None:
    os.environ['LOCAL_RANK'] = str(device_index)

    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)

    if hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats(device)

    dataset = AniGenSparseFeat2Skeleton(config.roots, **config.dataset_kwargs, local_rank=device_index)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.loader_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
    )

    iterator = iter(dataloader)
    trace: List[Dict[str, Optional[int]]] = []
    processed_batches = 0

    try:
        while processed_batches < config.steps_per_thread:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            _process_batch(batch, device, config.points_per_sample)

            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            snapshot = _global_memory_snapshot(device)
            global_used: Optional[int]
            global_total: Optional[int]
            if snapshot is None:
                global_used = None
                global_total = None
            else:
                global_used, global_total = snapshot
            trace.append(
                {
                    "step": processed_batches,
                    "allocated": allocated,
                    "reserved": reserved,
                    "global_used": global_used,
                    "global_total": global_total,
                }
            )
            processed_batches += 1

    finally:
        if hasattr(dataloader, "_shutdown_workers"):
            dataloader._shutdown_workers()  # type: ignore[attr-defined]

    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    global_used_values = [entry["global_used"] for entry in trace if entry.get("global_used") is not None]
    global_peak_used = max(global_used_values) if global_used_values else None
    final_snapshot = _global_memory_snapshot(device)
    if final_snapshot is not None:
        final_used, final_total = final_snapshot
        global_peak_used = max(global_peak_used or 0, final_used) if global_peak_used is not None else final_used
    else:
        final_used = None
        final_total = None
    global_last_used = final_used if final_used is not None else (
        trace[-1]["global_used"] if trace and trace[-1].get("global_used") is not None else None
    )
    global_total = final_total if final_total is not None else (
        trace[-1]["global_total"] if trace and trace[-1].get("global_total") is not None else None
    )

    with lock:
        results[device_index] = {
            "trace": trace,
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_reserved,
            "batches": processed_batches,
            "global_peak_used": global_peak_used,
            "global_last_used": global_last_used,
            "global_total": global_total,
        }


def _launch_threads(config: _MemoryTestConfig) -> Dict[int, Dict[str, object]]:
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("CUDA devices are required for the cuBVH memory probe.")

    num_threads = num_devices if config.max_threads is None else min(num_devices, config.max_threads)
    results: Dict[int, Dict[str, object]] = {}
    lock = threading.Lock()
    threads: List[threading.Thread] = []

    for device_index in range(num_threads):
        thread = threading.Thread(
            target=_memory_probe_worker,
            args=(device_index, config, results, lock),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return results


def _assert_balanced_memory(results: Dict[int, Dict[str, object]], config: _MemoryTestConfig) -> None:
    active_results = {idx: info for idx, info in results.items() if info.get("batches", 0) > 0}
    if not active_results:
        _fail("No GPU batches were processed during the cuBVH memory probe.")

    peaks = [info["peak_reserved"] for info in active_results.values()]
    if len(peaks) > 1:
        max_peak = max(peaks)
        min_peak = min(peaks)
        if max_peak > 0:
            imbalance = (max_peak - min_peak) / max_peak
            assert (
                imbalance <= config.imbalance_tolerance
            ), (
                "GPU memory usage is imbalanced: max reserved "
                f"{_bytes_to_megabytes(max_peak):.1f} MiB vs min reserved "
                f"{_bytes_to_megabytes(min_peak):.1f} MiB (Δ={imbalance * 100:.1f}%)."
            )

    gpu0_info = results.get(0)
    if gpu0_info:
        trace = gpu0_info.get("trace", [])
        if len(trace) > 1:
            first_alloc = trace[0]["allocated"] if trace[0]["allocated"] is not None else 0
            last_alloc = trace[-1]["allocated"] if trace[-1]["allocated"] is not None else 0
            growth = last_alloc - first_alloc
            assert (
                growth <= config.memory_growth_tolerance_bytes
            ), (
                "Detected GPU0 memory growth across batches: first "
                f"{_bytes_to_megabytes(first_alloc):.1f} MiB -> last "
                f"{_bytes_to_megabytes(last_alloc):.1f} MiB (Δ={_bytes_to_megabytes(growth):.1f} MiB)."
            )
        global_values = [entry["global_used"] for entry in trace if entry.get("global_used") is not None]
        if len(global_values) >= 2:
            first_global = global_values[0]
            last_global = global_values[-1]
            growth_global = last_global - first_global
            if growth_global > config.memory_growth_tolerance_bytes:
                print(
                    "[WARN] GPU0 global memory usage grew by "
                    f"{_bytes_to_megabytes(growth_global):.1f} MiB across trace entries."
                )


def _print_cli_summary(results: Dict[int, Dict[str, object]]) -> None:
    if not results:
        print("[WARN] No GPU results captured.")
        return

    print("cuBVH memory probe summary:")
    for idx in sorted(results):
        info = results[idx]
        batches = info.get("batches", 0)
        peak_alloc = _bytes_to_megabytes(info.get("peak_allocated", 0))
        peak_reserved = _bytes_to_megabytes(info.get("peak_reserved", 0))
        trace = info.get("trace", [])
        if trace:
            final_alloc_bytes = trace[-1]["allocated"] if trace[-1]["allocated"] is not None else 0
            final_reserved_bytes = trace[-1]["reserved"] if trace[-1]["reserved"] is not None else 0
            final_global_used = trace[-1].get("global_used")
        else:
            final_alloc_bytes = 0
            final_reserved_bytes = 0
            final_global_used = None
        final_alloc = _bytes_to_megabytes(final_alloc_bytes)
        final_reserved = _bytes_to_megabytes(final_reserved_bytes)
        peak_global_used = info.get("global_peak_used")
        global_total = info.get("global_total")
        message = (
            f"  [GPU {idx}] batches={batches} final_alloc={final_alloc:.1f} MiB "
            f"final_reserved={final_reserved:.1f} MiB peak_alloc={peak_alloc:.1f} MiB "
            f"peak_reserved={peak_reserved:.1f} MiB"
        )
        if final_global_used is not None:
            final_global_mb = _bytes_to_megabytes(final_global_used)
            message += f" global_used={final_global_mb:.1f} MiB"
            if peak_global_used is not None:
                message += f" (peak {_bytes_to_megabytes(peak_global_used):.1f} MiB)"
            if global_total is not None:
                message += f" of {_bytes_to_megabytes(global_total):.1f} MiB"
        print(message)


def main(argv: List[str] | None = None) -> int:
    # This is needed to make sure CUDA context is created in each thread.
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="cuBVH GPU memory balance probe")
    parser.add_argument("--roots", default=_DEFAULT_ROOT, help="Comma-separated AniGen dataset roots")
    parser.add_argument("--image-size", type=int, default=256, dest="image_size", help="Image resolution for dataset loading")
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size", help="DataLoader batch size per thread")
    parser.add_argument("--loader-workers", type=int, default=2, dest="loader_workers", help="Number of DataLoader workers per thread")
    parser.add_argument("--steps", type=int, default=100, help="Batches processed per thread")
    parser.add_argument("--points-per-sample", type=int, default=200_000, dest="points_per_sample", help="Random query points per mesh")
    parser.add_argument("--imbalance-tolerance", type=float, default=0.2, dest="imbalance_tolerance", help="Relative GPU memory imbalance tolerance")
    parser.add_argument("--growth-tolerance-mb", type=float, default=64.0, dest="growth_tolerance_mb", help="Allowed GPU0 memory growth across batches (MiB)")
    parser.add_argument("--max-threads", type=int, default=None, dest="max_threads", help="Limit number of GPU threads to launch")
    parser.add_argument("--test-mode", action="store_true", help="Use dataset test split instead of train/val")
    parser.add_argument("--filter-bad-skin", action="store_true", help="Filter out instances flagged with bad skin")

    args = parser.parse_args(argv)

    roots = args.roots
    ok, reason = _check_environment(roots)
    if not ok:
        print(f"[SKIP] {reason}")
        return 0

    config = _MemoryTestConfig(
        roots=roots,
        dataset_kwargs={
            "image_size": args.image_size,
            "test_mode": args.test_mode,
            "filter_bad_skin": args.filter_bad_skin,
            "load_cubvh": True,
        },
        batch_size=args.batch_size,
        loader_workers=args.loader_workers,
        steps_per_thread=args.steps,
        points_per_sample=args.points_per_sample,
        imbalance_tolerance=args.imbalance_tolerance,
        memory_growth_tolerance_bytes=int(args.growth_tolerance_mb * 1024 * 1024),
        max_threads=args.max_threads,
    )

    try:
        results = _launch_threads(config)
        _assert_balanced_memory(results, config)
    except AssertionError as err:
        print(f"[FAIL] {err}", file=sys.stderr)
        return 1
    except Exception as err:  # pragma: no cover - unexpected runtime failure
        print(f"[ERROR] {err}", file=sys.stderr)
        return 1

    _print_cli_summary(results)
    print("[OK] GPU memory usage is balanced within specified tolerances.")
    return 0


def test_cubvh_memory_balance() -> None:
    roots = _DEFAULT_ROOT
    ok, reason = _check_environment(roots)
    if not ok:
        _skip(reason)

    config = _MemoryTestConfig(
        roots=roots,
        dataset_kwargs={
            "image_size": 256,
            "test_mode": False,
            "filter_bad_skin": False,
        },
        steps_per_thread=3,
        loader_workers=2,
        points_per_sample=200_000,
    )

    results = _launch_threads(config)
    assert results, "No results collected from cuBVH memory probe threads."
    _assert_balanced_memory(results, config)


if __name__ == "__main__":
    sys.exit(main())

