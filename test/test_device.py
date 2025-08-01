#!/usr/bin/env python3

import numpy as np
import torch
import sys
import os

# Import the cuBVH class
from cubvh.api import cuBVH

def test_device_support():
    """Test device specification and .to(device) functionality."""
    
    # Check if multiple GPUs are available
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    if num_gpus < 1:
        print("No CUDA devices available. Skipping device tests.")
        return
    
    # Create a simple cube mesh for testing
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ], dtype=np.float32)
    
    triangles = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom face
        [4, 7, 6], [4, 6, 5],  # top face
        [0, 4, 5], [0, 5, 1],  # front face
        [2, 6, 7], [2, 7, 3],  # back face
        [0, 3, 7], [0, 7, 4],  # left face
        [1, 5, 6], [1, 6, 2]   # right face
    ], dtype=np.uint32)
    
    # Test 1: Create BVH without specifying device (should use current device)
    print("\\n1. Testing default device creation...")
    bvh_default = cuBVH(vertices, triangles)
    print(f"   Default BVH device: {bvh_default.device}")
    
    # Test 2: Create BVH on specific device
    print("\\n2. Testing device specification...")
    device_0 = torch.device('cuda:0')
    bvh_cuda0 = cuBVH(vertices, triangles, device=device_0)
    print(f"   BVH on cuda:0 device: {bvh_cuda0.device}")
    
    # Test 3: Create BVH with string device specification
    print("\\n3. Testing string device specification...")
    bvh_cuda0_str = cuBVH(vertices, triangles, device='cuda:0')
    print(f"   BVH with 'cuda:0' string device: {bvh_cuda0_str.device}")
    
    # Test 4: Test ray tracing with device handling
    print("\\n4. Testing ray tracing with automatic device handling...")
    # Create rays on a different device if possible
    if num_gpus > 1:
        rays_o = torch.tensor([[0.5, 0.5, -1.0]], dtype=torch.float32, device='cuda:1')
        rays_d = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda:1')
        print(f"   Rays created on: cuda:1")
    else:
        rays_o = torch.tensor([[0.5, 0.5, -1.0]], dtype=torch.float32, device='cuda:0')
        rays_d = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda:0')
        print(f"   Rays created on: cuda:0")
    
    positions, face_id, depth = bvh_cuda0.ray_trace(rays_o, rays_d)
    print(f"   Ray trace result device: {positions.device}")
    print(f"   Result - depth: {depth}, face_id: {face_id}")
    
    # Test 5: Test .to(device) method if multiple GPUs available
    if num_gpus > 1:
        print("\\n5. Testing .to(device) method...")
        print(f"   Original BVH device: {bvh_cuda0.device}")
        
        # Move to cuda:1
        bvh_cuda1 = bvh_cuda0.to('cuda:1')
        print(f"   After .to('cuda:1'): {bvh_cuda1.device}")
        
        # Test that the moved BVH works
        rays_o_1 = torch.tensor([[0.5, 0.5, -1.0]], dtype=torch.float32, device='cuda:1')
        rays_d_1 = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda:1')
        
        positions_1, face_id_1, depth_1 = bvh_cuda1.ray_trace(rays_o_1, rays_d_1)
        print(f"   Ray trace on cuda:1 - depth: {depth_1}, face_id: {face_id_1}")
        print(f"   Results device: {positions_1.device}")
        
        # Verify results are the same
        assert torch.allclose(depth, depth_1.to(depth.device), atol=1e-6), "Results should be the same!"
        print("   ✅ Results match between devices!")
        
        # Test moving back to original device
        bvh_back = bvh_cuda1.to('cuda:0')
        print(f"   After moving back to cuda:0: {bvh_back.device}")
    else:
        print("\\n5. Skipping .to(device) test (only one GPU available)")
    
    # Test 6: Test that device is preserved through pickle/unpickle
    print("\\n6. Testing device preservation through pickle...")
    import tempfile
    # with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    #     temp_path = f.name
    temp_path = './bvh.pt'
    
    try:
        # Save BVH
        torch.save(bvh_cuda0, temp_path)
        
        # Load BVH
        bvh_loaded = torch.load(temp_path, weights_only=False)
        print(f"   Original device: {bvh_cuda0.device}")
        print(f"   Loaded device: {bvh_loaded.device}")
        
        # Test that loaded BVH works
        positions_loaded, face_id_loaded, depth_loaded = bvh_loaded.ray_trace(
            torch.tensor([[0.5, 0.5, -1.0]], dtype=torch.float32, device=bvh_loaded.device),
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=bvh_loaded.device)
        )
        print(f"   Loaded BVH ray trace - depth: {depth_loaded}, face_id: {face_id_loaded}")
        print("   ✅ Device preserved through pickle!")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print("\\n🎉 All device tests passed!")

if __name__ == "__main__":
    test_device_support()
