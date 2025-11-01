import itertools
import logging
import os
import threading
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import cubvh
import trimesh


LAMBDA_SKIN_FEATS_L2 = 0.1
LAMBDA_SKIN_KL = 0.05
LAMBDA_SKIN_VAR = 0.02


def _l2_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)


def _mesh_instances(root: str) -> Iterable[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Mesh root does not exist: {root}")
    for name in sorted(os.listdir(root)):
        npz_path = os.path.join(root, name, "skeleton_voxelized.npz")
        if os.path.isfile(npz_path):
            yield name


def _load_mesh_from_npz(root: str, instance: str) -> Tuple[torch.Tensor, torch.Tensor]:
    npz_path = os.path.join(root, instance, "skeleton_voxelized.npz")
    data = np.load(npz_path)
    vertices = data["vertices"].astype(np.float32)
    triangles = data["faces"].astype(np.int32)

    return (
        torch.from_numpy(vertices.copy()),
        torch.from_numpy(triangles.copy()),
    )


class _MeshShardDataset(Dataset):
    def __init__(self, mesh_root: str, instances: List[str]) -> None:
        self._mesh_root = mesh_root
        self._instances = instances

    def __len__(self) -> int:
        return len(self._instances)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        instance = self._instances[index]
        vertices_cpu, triangles_cpu = _load_mesh_from_npz(self._mesh_root, instance)
        return instance, vertices_cpu, triangles_cpu

def _single_item_collate(batch: List[Tuple[str, torch.Tensor, torch.Tensor]]) -> Tuple[str, torch.Tensor, torch.Tensor]:
    if len(batch) != 1:
        raise RuntimeError(f"Expected single item batch, got {len(batch)} elements")
    return batch[0]


def _worker(
    device_index: int,
    mesh_root: str,
    shard: List[str],
    results: Dict[int, Dict[str, Any]],
) -> None:
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)

    # Track peak usage per device so we can flag imbalances.
    if hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats(device)
    else:
        torch.cuda.empty_cache()

    if not shard:
        results[device_index] = {
            "status": "ok",
            "message": "no instances assigned",
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
        return

    dataloader_workers = 8
    dataset = _MeshShardDataset(mesh_root, shard)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=dataloader_workers,
        collate_fn=_single_item_collate,
        pin_memory=False,
        persistent_workers=False,
    )

    processed_instances = 0
    last_message = ""

    for _, (instance, vertices_cpu, triangles_cpu) in enumerate(dataloader):
        bvh = cubvh.cuBVH(vertices=vertices_cpu, triangles=triangles_cpu, device=device)
        mesh_verts = torch.randn(2000_000, 3, device=device)
        distances, face_id_raw, uvw = bvh.unsigned_distance(mesh_verts, return_uvw=True)
        distances.max().item()
        last_message = instance
        processed_instances += 1

        # Ensure temporary tensors are released promptly to keep usage stable.
        bvh.to("cpu")
        del distances, face_id_raw, uvw, mesh_verts, bvh

    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    if processed_instances:
        message = f"processed {processed_instances} instances; last {last_message}"
    else:
        message = "no instances processed"

    results[device_index] = {
        "status": "ok",
        "message": message,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }


def main(mesh_root: str = "/mnt/pfs/data/huangyihua/AniGen_test/skeleton") -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this multithreaded device test.")

    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise SystemExit("No CUDA devices detected.")

    num_threads = min(device_count, 8)

    instances = list(_mesh_instances(mesh_root))
    if not instances:
        raise SystemExit(f"No skeleton instances found under {mesh_root}")

    shard_size = (len(instances) + num_threads - 1) // num_threads
    shards: List[List[str]] = [
        [
            instances[j]
            for j in range(i * shard_size, min((i + 1) * shard_size, len(instances)))
        ]
        for i in range(num_threads)
    ]

    threads = []
    results: Dict[int, Dict[str, Any]] = {}

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for idx in range(num_threads):
        shard = shards[idx] if idx < len(shards) else []
        thread = threading.Thread(target=_worker, args=(idx, mesh_root, shard, results), daemon=True)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    missing = [idx for idx in range(num_threads) if idx not in results]
    if missing:
        raise SystemExit(f"cuBVH thread/device test did not record results for devices: {missing}")

    failures = {idx: info for idx, info in results.items() if info["status"] != "ok"}
    if failures:
        for idx, info in failures.items():
            print(f"[Device {idx}] FAILED: {info['message']}")
        raise SystemExit(f"cuBVH thread/device test failed on {len(failures)} device(s).")

    for idx, info in sorted(results.items()):
        peak_alloc_mb = info["peak_allocated_bytes"] / (1024 ** 2)
        peak_reserved_mb = info["peak_reserved_bytes"] / (1024 ** 2)
        status = info["status"].upper()
        print(
            f"[Device {idx}] {status}: {info['message']} | "
            f"peak_alloc={peak_alloc_mb:.1f} MB, peak_reserved={peak_reserved_mb:.1f} MB"
        )

    peak_reserved_values = [info["peak_reserved_bytes"] for info in results.values() if info["status"] == "ok"]
    if peak_reserved_values:
        max_reserved = max(peak_reserved_values)
        min_reserved = min(peak_reserved_values)
        max_reserved_mb = max_reserved / (1024 ** 2)
        min_reserved_mb = min_reserved / (1024 ** 2)
        diff_mb = max_reserved_mb - min_reserved_mb
        print(
            f"Peak reserved memory range: {min_reserved_mb:.1f} MB - {max_reserved_mb:.1f} MB"
            f" (Δ={diff_mb:.1f} MB)"
        )
        if len(peak_reserved_values) > 1 and max_reserved_mb > 0:
            imbalance_ratio = diff_mb / max_reserved_mb
            if imbalance_ratio > 0.15:
                print("WARNING: GPU memory usage appears imbalanced across devices.")
            else:
                print("INFO: GPU memory usage appears balanced across devices.")

    print(f"cuBVH multi-threaded multi-device test passed with {len(instances)} instances on {num_threads} devices.")


if __name__ == "__main__":
    main()
