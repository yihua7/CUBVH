import itertools
import logging
import os
import threading
from typing import Dict, Iterable, List, Tuple

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
        return instance, vertices_cpu.contiguous(), triangles_cpu.contiguous()


def _single_item_collate(batch: List[Tuple[str, torch.Tensor, torch.Tensor]]) -> Tuple[str, torch.Tensor, torch.Tensor]:
    if len(batch) != 1:
        raise RuntimeError(f"Expected single item batch, got {len(batch)} elements")
    return batch[0]


def _worker(
    device_index: int,
    mesh_root: str,
    shard: List[str],
    results: Dict[int, Tuple[str, str]],
) -> None:
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)

    if not shard:
        results[device_index] = ("ok", "no instances assigned")
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

    for local_idx, (instance, vertices_cpu, triangles_cpu) in enumerate(dataloader):
        try:
            vertices = vertices_cpu.to(device)
            triangles = triangles_cpu.to(device)
            triangle_index = triangles.to(device=device, dtype=torch.long)

            bvh_path = os.path.join(mesh_root, instance, "cubvh.pth")
            if os.path.exists(bvh_path):
                bvh_cpu = torch.load(bvh_path, map_location="cpu")
            else:
                bvh_cpu = cubvh.cuBVH(vertices=vertices_cpu, triangles=triangles_cpu, device="cpu")
            bvh = bvh_cpu.to(device)

            num_vertices = vertices.shape[0]
            num_joints = min(64, max(4, num_vertices // 4))
            vert_embed_dim = 32
            joint_embed_dim = 48

            joint_skin_embeds_gt_list = [torch.randn(num_joints, joint_embed_dim, device=device)]
            vert_skin_embeds_gt_list = [torch.randn(num_vertices, vert_embed_dim, device=device)]
            joint_skin_embeds_gt, vert_skin_embeds_gt = (
                joint_skin_embeds_gt_list[0].detach(),
                vert_skin_embeds_gt_list[0].detach(),
            )

            skin_gt = torch.softmax(
                torch.randn(num_vertices, num_joints, device=device, dtype=torch.float32),
                dim=-1,
            )

            mesh_verts = (vertices + 0.01 * torch.randn_like(vertices)).detach()

            distances, face_id_raw, uvw = bvh.unsigned_distance(mesh_verts, return_uvw=True)

            if face_id_raw.shape[0] != uvw.shape[0]:
                raise RuntimeError(
                    f"unsigned_distance returned mismatched outputs: face_id={face_id_raw.shape}, uvw={uvw.shape}"
                )

            face_id_raw = face_id_raw.long()
            if torch.any(face_id_raw < 0) or torch.any(face_id_raw >= triangle_index.shape[0]):
                raise RuntimeError("BVH returned invalid face indices")

            face_vertices = triangle_index[face_id_raw]

            uvw = uvw.clamp(min=0.0)
            uvw_sum = uvw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            barycentric = uvw / uvw_sum

            skin_nn_gt = (skin_gt[face_vertices] * barycentric.unsqueeze(-1)).sum(dim=1)
            vert_skin_embeds_gt_nn = (vert_skin_embeds_gt[face_vertices] * barycentric.unsqueeze(-1)).sum(dim=1)

            joints_nn_idx = torch.randint(0, num_joints, (num_joints,), device=device)
            joint_skin_embeds_gt_nn = joint_skin_embeds_gt[joints_nn_idx]

            vert_skin_embeds_pred = torch.randn(num_vertices, vert_embed_dim, device=device, requires_grad=True)
            joint_skin_embeds_pred = torch.randn(num_joints, joint_embed_dim, device=device, requires_grad=True)
            skin_pred_logits = torch.randn(num_vertices, num_joints, device=device, requires_grad=True)
            skin_pred = torch.softmax(skin_pred_logits, dim=-1)
            skin_feats_joints_var = torch.randn((), device=device, requires_grad=True)

            is_bad_skin = False

            if is_bad_skin:
                skin_feats_l2_loss = _l2_loss(vert_skin_embeds_pred, vert_skin_embeds_pred.detach())
                skin_feats_l2_loss = skin_feats_l2_loss + _l2_loss(
                    joint_skin_embeds_pred, joint_skin_embeds_pred.detach()
                )
                total_loss = LAMBDA_SKIN_FEATS_L2 * skin_feats_l2_loss
            else:
                skin_feats_l2_loss = _l2_loss(vert_skin_embeds_pred, vert_skin_embeds_gt_nn) + _l2_loss(
                    joint_skin_embeds_pred, joint_skin_embeds_gt_nn
                )
                skin_kl_loss = F.kl_div(
                    torch.log(skin_pred + 1e-10),
                    skin_nn_gt,
                    reduction="batchmean",
                )
                joint_var_loss = skin_feats_joints_var.abs()
                total_loss = (
                    LAMBDA_SKIN_FEATS_L2 * skin_feats_l2_loss
                    + LAMBDA_SKIN_KL * skin_kl_loss
                    + LAMBDA_SKIN_VAR * joint_var_loss
                )

            distances_mean = float(distances.mean())
            total_loss.backward()

            if not distances.is_cuda or distances.device != device:
                raise RuntimeError("Distance tensor not on expected device")

            torch.cuda.synchronize(device)

            bvh_cpu = bvh.to("cpu")
            if bvh_cpu.device.type != "cpu":
                raise RuntimeError("cuBVH did not move back to CPU")

            processed_instances += 1
            last_message = f"instance={instance} mean_distance={distances_mean}"

        except Exception as exc:  # noqa: BLE001
            logging.exception("[Device %s] Failure on instance %s", device_index, instance)
            results[device_index] = ("error", instance)
            raise

    if processed_instances:
        results[device_index] = ("ok", f"processed {processed_instances} instances; last {last_message}")
    else:
        results[device_index] = ("ok", "no instances processed")


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
    results: Dict[int, Tuple[str, str]] = {}

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
