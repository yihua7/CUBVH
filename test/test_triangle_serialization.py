#!/usr/bin/env python3
"""
Test script to demonstrate the new triangle serialization functionality.
This script shows how to:
1. Create a BVH 
2. Get the sorted triangles and BVH nodes
3. Recreate the BVH from the saved data
4. Verify that the BVH works correctly
"""

import numpy as np
import torch
import cubvh

def create_test_mesh():
    """Create a simple test mesh (a cube)."""
    # Cube vertices
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom face
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],  # top face
    ], dtype=np.float32)
    
    # Cube triangles (12 triangles, 2 per face)
    triangles = np.array([
        # bottom face
        [0, 1, 2], [0, 2, 3],
        # top face  
        [4, 6, 5], [4, 7, 6],
        # front face
        [0, 4, 5], [0, 5, 1],
        # back face
        [2, 6, 7], [2, 7, 3],
        # left face
        [0, 3, 7], [0, 7, 4],
        # right face
        [1, 5, 6], [1, 6, 2],
    ], dtype=np.uint32)
    
    return vertices, triangles

def test_triangle_serialization():
    """Test the triangle serialization functionality."""
    print("Creating test mesh...")
    vertices, triangles = create_test_mesh()
    
    # Create original BVH
    print("Creating original BVH...")
    bvh_original = cubvh.cuBVH(vertices, triangles)
    
    # Get the BVH nodes and sorted triangles
    print("Getting BVH nodes and sorted triangles...")
    bvh_nodes = bvh_original.get_bvh_nodes()
    triangles_data = bvh_original.get_triangles_data()
    
    print(f"BVH nodes shape: {bvh_nodes.shape}")
    print(f"Triangles data shape: {triangles_data.shape}")
    
    # Create new BVH from the saved data
    print("Creating BVH from saved data...")
    bvh_restored = cubvh.cuBVH(vertices, triangles, bvh_nodes=bvh_nodes, triangles_data=triangles_data)
    
    # Test that both BVHs give the same results
    print("Testing ray tracing...")
    test_rays_o = torch.tensor([[0, 0, -5], [2, 0, 0], [0, 2, 0]], dtype=torch.float32, device='cuda')
    test_rays_d = torch.tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=torch.float32, device='cuda')
    
    # Ray trace with original BVH
    pos1, face_id1, depth1 = bvh_original.ray_trace(test_rays_o, test_rays_d)
    
    # Ray trace with restored BVH  
    pos2, face_id2, depth2 = bvh_restored.ray_trace(test_rays_o, test_rays_d)
    
    # Check if results are the same
    pos_diff = torch.abs(pos1 - pos2).max().item()
    face_diff = torch.abs(face_id1 - face_id2).max().item()
    depth_diff = torch.abs(depth1 - depth2).max().item()
    
    print(f"Position difference: {pos_diff}")
    print(f"Face ID difference: {face_diff}")
    print(f"Depth difference: {depth_diff}")
    
    if pos_diff < 1e-6 and face_diff == 0 and depth_diff < 1e-6:
        print("✓ Ray tracing results match!")
    else:
        print("✗ Ray tracing results don't match!")
        return False
        
    # Test signed distance
    print("Testing signed distance...")
    test_points = torch.tensor([[0, 0, 0], [0, 0, 2], [0, 0, -2]], dtype=torch.float32, device='cuda')
    
    dist1, fid1, _ = bvh_original.signed_distance(test_points)
    dist2, fid2, _ = bvh_restored.signed_distance(test_points)
    
    dist_diff = torch.abs(dist1 - dist2).max().item()
    fid_diff = torch.abs(fid1 - fid2).max().item()
    
    print(f"Distance difference: {dist_diff}")
    print(f"Face ID difference: {fid_diff}")
    
    if dist_diff < 1e-6 and fid_diff == 0:
        print("✓ Signed distance results match!")
    else:
        print("✗ Signed distance results don't match!")
        return False
    
    # Test the get_sorted_triangles method
    print("Testing get_sorted_triangles...")
    sorted_vertices, sorted_triangles, triangle_ids = bvh_restored.get_sorted_triangles()
    print(f"Sorted vertices shape: {sorted_vertices.shape}")
    print(f"Sorted triangles shape: {sorted_triangles.shape}")
    print(f"Triangle IDs shape: {triangle_ids.shape}")
    print(f"Triangle IDs: {triangle_ids}")
    
    return True

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available. Running test...")
        success = test_triangle_serialization()
        if success:
            print("\n✓ All tests passed! Triangle serialization works correctly.")
        else:
            print("\n✗ Some tests failed!")
    else:
        print("CUDA is not available. Cannot run test.")
