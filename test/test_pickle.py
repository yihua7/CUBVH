#!/usr/bin/env python3

import numpy as np
import torch
import tempfile
import os
import trimesh
import time

# Import the cuBVH class
import sys
from cubvh.api import cuBVH

def test_bvh_pickle():
    """Test that cuBVH can be pickled and unpickled correctly."""

    # Load mesh.ply from the test directory
    mesh_path = './mesh.ply'
    if not os.path.exists(mesh_path):
        print(f"Error: {mesh_path} not found!")
        return
    
    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path, process=False)
    vertices = mesh.vertices.astype(np.float32)
    triangles = mesh.faces.astype(np.uint32)
    
    print(f"Mesh loaded: {len(vertices)} vertices, {len(triangles)} triangles")
    
    print("Creating cuBVH...")
    bvh = cuBVH(vertices, triangles)
    
    # Generate more random rays for testing
    print("Testing ray trace with random rays before save...")
    n_rays = 1000
    torch.manual_seed(42)  # For reproducible results
    rays_o = torch.rand(n_rays, 3, device='cuda') * 2 - 1  # Random origins in [-1, 1]
    rays_d = torch.rand(n_rays, 3, device='cuda') * 2 - 1  # Random directions
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)  # Normalize directions
    
    positions_orig, face_id_orig, depth_orig = bvh.ray_trace(rays_o, rays_d)
    print(f"Original results - mean depth: {depth_orig.mean():.4f}, "
          f"hit rate: {(depth_orig < 10.0).float().mean():.2%}")
    
    # Also test with a few specific rays for detailed comparison
    print("Testing with specific test rays...")
    test_rays_o = torch.tensor([
        [0.0, 0.0, -2.0],
        [0.5, 0.5, -2.0], 
        [-0.5, -0.5, -2.0]
    ], dtype=torch.float32, device='cuda')
    test_rays_d = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device='cuda')
    
    test_positions_orig, test_face_id_orig, test_depth_orig = bvh.ray_trace(test_rays_o, test_rays_d)
    print(f"Test ray results - depths: {test_depth_orig}, face_ids: {test_face_id_orig}")
    
    # Test saving and loading
    # with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    #     temp_path = f.name
    temp_path = './bvh.pt'
    
    try:
        print("Saving BVH with torch.save...")
        torch.save(bvh, temp_path)
        print("Save successful!")
        
        print("Loading BVH with torch.load...")
        bvh_loaded = torch.load(temp_path, weights_only=False)
        print("Load successful!")
        
        print("Testing ray trace after load...")
        positions_loaded, face_id_loaded, depth_loaded = bvh_loaded.ray_trace(rays_o, rays_d)
        print(f"Loaded results - mean depth: {depth_loaded.mean():.4f}, "
              f"hit rate: {(depth_loaded < 10.0).float().mean():.2%}")
        
        # Test the specific rays too
        test_positions_loaded, test_face_id_loaded, test_depth_loaded = bvh_loaded.ray_trace(test_rays_o, test_rays_d)
        print(f"Test ray results after load - depths: {test_depth_loaded}, face_ids: {test_face_id_loaded}")
        
        # Verify results are the same
        print("Verifying results match...")
        assert torch.allclose(depth_orig, depth_loaded, atol=1e-6), "Random ray depth results don't match!"
        assert torch.equal(face_id_orig, face_id_loaded), "Random ray face ID results don't match!"
        assert torch.allclose(test_depth_orig, test_depth_loaded, atol=1e-6), "Test ray depth results don't match!"
        assert torch.equal(test_face_id_orig, test_face_id_loaded), "Test ray face ID results don't match!"
        
        print("✅ All tests passed! BVH pickle/unpickle works correctly.")
        
        # Performance comparison: Creating BVH vs Loading BVH
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        
        print("Testing BVH creation speed (100 iterations)...")
        creation_times = []
        for i in range(100):
            start_time = time.time()
            temp_bvh = cuBVH(vertices, triangles)
            creation_times.append(time.time() - start_time)
        
        avg_creation_time = np.mean(creation_times)
        std_creation_time = np.std(creation_times)
        print(f"BVH Creation - Average: {avg_creation_time:.4f}s, Std: {std_creation_time:.4f}s")
        
        print("\nTesting BVH loading speed (100 iterations)...")
        loading_times = []
        for i in range(100):
            start_time = time.time()
            temp_bvh_loaded = torch.load(temp_path, weights_only=False)
            loading_times.append(time.time() - start_time)
        
        avg_loading_time = np.mean(loading_times)
        std_loading_time = np.std(loading_times)
        print(f"BVH Loading - Average: {avg_loading_time:.4f}s, Std: {std_loading_time:.4f}s")
        
        # Calculate speedup
        speedup = avg_creation_time / avg_loading_time
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"   Creation time: {avg_creation_time:.4f} ± {std_creation_time:.4f}s")
        print(f"   Loading time:  {avg_loading_time:.4f} ± {std_loading_time:.4f}s")
        print(f"   Speedup:       {speedup:.2f}x faster to load vs create")
        
        if speedup > 1:
            print(f"   💡 Loading is {speedup:.2f}x faster than creation - BVH caching is beneficial!")
        else:
            print(f"   ⚠️  Creation is faster than loading - consider optimizing serialization")
        
        # Test that loaded BVH still works correctly
        print(f"\n🔍 Verifying loaded BVH functionality...")
        quick_test_positions, quick_test_face_id, quick_test_depth = temp_bvh_loaded.ray_trace(test_rays_o[:1], test_rays_d[:1])
        assert torch.allclose(test_depth_orig[:1], quick_test_depth, atol=1e-6), "Loaded BVH doesn't work correctly!"
        print("   ✅ Loaded BVH works correctly!")
        
        # Test device functionality
        print(f"\n🖥️  DEVICE FUNCTIONALITY TEST:")
        print(f"   Original BVH device: {bvh.device}")
        print(f"   Loaded BVH device: {temp_bvh_loaded.device}")
        
        # Test moving to different device if multiple GPUs available
        if torch.cuda.device_count() > 1:
            print(f"   Testing device transfer to cuda:1...")
            bvh_on_cuda1 = temp_bvh_loaded.to('cuda:1')
            print(f"   BVH moved to: {bvh_on_cuda1.device}")
            
            # Test that it works on the new device
            test_rays_cuda1 = test_rays_o[:1].to('cuda:1')
            test_dirs_cuda1 = test_rays_d[:1].to('cuda:1')
            pos_cuda1, fid_cuda1, depth_cuda1 = bvh_on_cuda1.ray_trace(test_rays_cuda1, test_dirs_cuda1)
            
            print(f"   Ray trace on cuda:1 - depth: {depth_cuda1}")
            assert torch.allclose(test_depth_orig[:1].to('cuda:1'), depth_cuda1, atol=1e-6), "Device transfer failed!"
            print("   ✅ Device transfer works correctly!")
        else:
            print(f"   Only one GPU available, skipping device transfer test")
        
    finally:
        # Count the temp file size
        if os.path.exists(temp_path):
            temp_size = os.path.getsize(temp_path) / (1024 * 1024)
            print(f"\nTemporary file size: {temp_size:.2f} MB")
            # Compare with mesh file size
            mesh_size = os.path.getsize(mesh_path) / (1024 * 1024)
            print(f"Mesh file size: {mesh_size:.2f} MB")
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    test_bvh_pickle()
