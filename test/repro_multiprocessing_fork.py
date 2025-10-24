"""Minimal reproduction of the CUDA fork reinitialization error with cuBVH.

This demonstrates that constructing a BVH on a forked subprocess while the parent
has already initialized CUDA used to raise:

    RuntimeError: Cannot re-initialize CUDA in forked subprocess.

After the host-side state builder change, the same workflow succeeds because no
CUDA context is touched when targeting the CPU.
"""

import multiprocessing as mp

import torch

import cubvh


def _make_mesh():
    # Simple cube mesh; identical to the serialization test helper.
    vertices = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=torch.float32)

    triangles = torch.tensor([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=torch.int32)

    return vertices, triangles


def _worker():
    vertices, triangles = _make_mesh()
    cubvh.cuBVH(vertices=vertices, triangles=triangles, device="cpu")
    # Succeed without touching CUDA when targeting CPU only.


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required to reproduce fork initialisation behaviour")

    # Ensure the parent initialises CUDA before forking. This is what triggers the
    # infamous reinitialisation error when children try to touch CUDA using the
    # default "fork" start method.
    torch.randn(1, device="cuda")

    ctx = mp.get_context("fork")
    proc = ctx.Process(target=_worker)
    proc.start()
    proc.join()

    if proc.exitcode != 0:
        raise SystemExit(f"Subprocess exited with code {proc.exitcode}")

    print("Forked subprocess constructed cuBVH on CPU without touching CUDA.")


if __name__ == "__main__":
    main()
