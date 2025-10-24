import itertools
import logging
import os
import threading
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import torch

import cubvh
import trimesh


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


class _ThreadState:
    def __init__(self, instances: List[str]):
        self.instances = instances
        self.error: Optional[BaseException] = None


def _worker(device_index: int, instances: List[str], root: str, results: Dict[int, Tuple[str, str]]) -> None:
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)

    for instance in instances:
        try:
            vertices, triangles = _load_mesh_from_npz(root, instance)
            vertices = vertices.to(device)
            triangles = triangles.to(device)

            bvh = cubvh.cuBVH(vertices=vertices, triangles=triangles, device="cpu")
            bvh = bvh.to(device)

            skin_gt = torch.randn_like(vertices, device=device)
            sample_points = torch.randn(200_000, 3, device=device, dtype=torch.float32, requires_grad=True)
            sample_points = torch.cat([sample_points, vertices], dim=0)
            skin_pred = torch.nn.Parameter(torch.randn_like(sample_points, device=device))
            distances, face_id, uvw = bvh.unsigned_distance(sample_points, return_uvw=True)

            assert face_id.shape[0] == uvw.shape[0], f'The output shapes are not matched! with face_id: {face_id.shape} and uvw: {uvw.shape}'
            
            if face_id.shape[0] == uvw.shape[0]:
                face_id = triangles[face_id]
                skin_nn_gt = (skin_gt[face_id] * uvw[..., None]).sum(1)
                skin_kl_loss = torch.nn.functional.kl_div(skin_pred, skin_nn_gt, reduction='batchmean')
                skin_kl_loss.backward()

                if not distances.is_cuda or distances.device != device:
                    raise RuntimeError("Distance tensor not on expected device")

                torch.cuda.synchronize(device)

                bvh_cpu = bvh.to("cpu")
                if bvh_cpu.device.type != "cpu":
                    raise RuntimeError("cuBVH did not move back to CPU")

                results[device_index] = ("ok", f"instance={instance} mean_distance={float(distances.mean())}")

        except Exception as exc:  # noqa: BLE001
            logging.exception("[Device %s] Failure on instance %s", device_index, instance)
            results[device_index] = ("error", instance)
            raise


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
    shards = [instances[i * shard_size:(i + 1) * shard_size] for i in range(num_threads)]

    threads = []
    results: Dict[int, Tuple[str, str]] = {}

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for idx in range(num_threads):
        shard = shards[idx] if idx < len(shards) else []
        thread = threading.Thread(target=_worker, args=(idx, shard, mesh_root, results), daemon=True)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    missing = [idx for idx in range(num_threads) if idx not in results]
    if missing:
        raise SystemExit(f"cuBVH thread/device test did not record results for devices: {missing}")

    failures = {idx: info for idx, info in results.items() if info[0] != "ok"}
    if failures:
        for idx, info in failures.items():
            print(f"[Device {idx}] FAILED: {info[1]}")
        raise SystemExit(f"cuBVH thread/device test failed on {len(failures)} device(s).")

    for idx, info in results.items():
        print(f"[Device {idx}] PASS: {info[1]}")

    print(f"cuBVH multi-threaded multi-device test passed with {len(instances)} instances on {num_threads} devices.")


if __name__ == "__main__":
    main()
