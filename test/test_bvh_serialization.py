import os
import tempfile

import torch
import numpy as np

import cubvh


def _make_cube_mesh(scale=1.0):
    vertices = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=torch.float32) * float(scale)

    triangles = torch.tensor([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=torch.int32)

    return vertices, triangles


def _assert_close(a, b, tol=1e-4):
    if torch.is_tensor(a) and torch.is_tensor(b):
        assert torch.allclose(a, b, atol=tol, rtol=tol), f"Tensor mismatch: max diff {torch.max(torch.abs(a - b))}"
    else:
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=tol, rtol=tol)


def run_roundtrip(device):
    vertices, triangles = _make_cube_mesh()
    bvh = cubvh.cuBVH(vertices=vertices, triangles=triangles, device=device)

    sample_points = torch.randn(128, 3, device=device, dtype=torch.float32) * 0.5
    distances, face_id, uvw = bvh.unsigned_distance(sample_points, return_uvw=True)
    assert distances.shape == (128,), "Unexpected distance shape"
    assert face_id.shape == (128,)
    assert uvw.shape == (128, 3)

    tmpdir = tempfile.mkdtemp(prefix="cubvh_test_")
    ckpt_path = os.path.join(tmpdir, "bvh.pt")
    torch.save(bvh, ckpt_path)

    reloaded = torch.load(ckpt_path)
    reloaded.to(device)

    distances_rt, face_id_rt, uvw_rt = reloaded.unsigned_distance(sample_points, return_uvw=True)
    _assert_close(distances, distances_rt)
    _assert_close(face_id, face_id_rt)
    _assert_close(uvw, uvw_rt)

    # Exercise device transfers.
    reloaded_cpu = reloaded.to("cpu")
    assert reloaded_cpu.device.type == "cpu"
    reloaded_back = reloaded_cpu.to(device)
    assert reloaded_back.device.type == device.type

    # Export state and ensure keys exist.
    state = reloaded_back.export_state()
    expected_keys = {"triangles", "triangle_ids", "node_mins", "node_maxs", "node_children"}
    assert expected_keys.issubset(state.keys())

    # Trigger numpy exports to verify CPU copies stay valid.
    tris_np = reloaded_back.triangles_cpu
    nodes_np = reloaded_back.bvh_nodes_cpu
    assert tris_np.shape[0] == 12
    assert set(nodes_np.keys()) == {"mins", "maxs", "children"}

    os.remove(ckpt_path)

    # Explicitly delete to help catch double-free issues.
    del reloaded_back
    del reloaded_cpu
    del reloaded
    del bvh


def run_cpu_only_build():
    vertices, triangles = _make_cube_mesh()
    bvh_cpu = cubvh.cuBVH(vertices=vertices, triangles=triangles, device="cpu")
    assert bvh_cpu.device.type == "cpu"

    state = bvh_cpu.export_state()
    expected_keys = {"triangles", "triangle_ids", "node_mins", "node_maxs", "node_children"}
    assert expected_keys.issubset(state.keys())

    tris_cpu = bvh_cpu.triangles_cpu
    nodes_cpu = bvh_cpu.bvh_nodes_cpu
    assert tris_cpu.shape[0] == 12
    assert set(nodes_cpu.keys()) == {"mins", "maxs", "children"}

    del bvh_cpu


def main():
    run_cpu_only_build()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run cuBVH tests.")

    device = torch.device("cuda")

    run_roundtrip(device)

    # Ensure CPU residency works even when starting from CUDA build.
    vertices, triangles = _make_cube_mesh()
    bvh = cubvh.cuBVH(vertices=vertices, triangles=triangles, device=device)
    bvh_cpu = bvh.to("cpu")
    assert bvh_cpu.device.type == "cpu"
    del bvh_cpu
    del bvh

    print("cuBVH serialization and device transfer test passed.")


if __name__ == "__main__":
    main()
