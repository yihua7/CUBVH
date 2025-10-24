import itertools
import logging
import os
import threading
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

import torch
import torch.nn.functional as F

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


def _worker(
    device_index: int,
    shard: List[Tuple[str, Dict[str, Any]]],
    results: Dict[int, Tuple[str, str]],
) -> None:
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)

    for local_idx, (instance, payload) in enumerate(shard):
        try:
            vertices = payload["vertices"].to(device)
            triangles = payload["triangles"].to(device)
            triangle_index = triangles.to(device=device, dtype=torch.long)

            bvh = payload["bvh"].to(device)

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

            results[device_index] = ("ok", f"instance={instance} mean_distance={distances_mean}")

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

    instance_payloads: Dict[str, Dict[str, Any]] = {}
    for name in instances:
        vertices_cpu, triangles_cpu = _load_mesh_from_npz(mesh_root, name)
        bvh_cpu = cubvh.cuBVH(vertices=vertices_cpu, triangles=triangles_cpu, device="cpu")
        instance_payloads[name] = {
            "vertices": vertices_cpu.contiguous(),
            "triangles": triangles_cpu.contiguous(),
            "bvh": bvh_cpu,
        }

    shard_size = (len(instances) + num_threads - 1) // num_threads
    shards: List[List[Tuple[str, Dict[str, Any]]]] = [
        [
            (instances[j], instance_payloads[instances[j]])
            for j in range(i * shard_size, min((i + 1) * shard_size, len(instances)))
        ]
        for i in range(num_threads)
    ]

    threads = []
    results: Dict[int, Tuple[str, str]] = {}

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for idx in range(num_threads):
        shard = shards[idx] if idx < len(shards) else []
        thread = threading.Thread(target=_worker, args=(idx, shard, results), daemon=True)
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
