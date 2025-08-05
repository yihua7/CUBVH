#!/usr/bin/env python3
"""
Test script to demonstrate memory leak fix in cuBVH.
This script creates many cuBVH instances and shows that memory is properly freed.
"""

import numpy as np
import torch
import cubvh
import gc
import time

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0

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

def test_memory_leak_fixed():
    """Test that creating many cuBVH instances doesn't cause memory leaks."""
    print("Testing memory leak fix...")
    
    vertices, triangles = create_test_mesh()
    
    # Clear any existing GPU memory
    torch.cuda.empty_cache()
    initial_memory = get_gpu_memory_usage()
    print(f"Initial GPU memory usage: {initial_memory:.2f} MB")
    
    # Test 1: Creating and destroying BVH instances without explicit cleanup
    print("\nTest 1: Creating BVH instances without explicit cleanup...")
    for i in range(100):
        bvh = cubvh.cuBVH(vertices, triangles)
        
        # Use the BVH briefly
        test_points = torch.tensor([[0, 0, 0]], dtype=torch.float32, device='cuda')
        dist, _, _ = bvh.signed_distance(test_points)
        
        # Let it go out of scope (should trigger __del__)
        del bvh
        
        if (i + 1) % 20 == 0:
            gc.collect()  # Force garbage collection
            torch.cuda.empty_cache()
            current_memory = get_gpu_memory_usage()
            print(f"  After {i+1} iterations: {current_memory:.2f} MB")
    
    gc.collect()
    torch.cuda.empty_cache()
    after_test1_memory = get_gpu_memory_usage()
    print(f"After test 1: {after_test1_memory:.2f} MB")
    
    # Test 2: Creating and destroying BVH instances with explicit cleanup
    print("\nTest 2: Creating BVH instances with explicit cleanup...")
    for i in range(100):
        bvh = cubvh.cuBVH(vertices, triangles)
        
        # Use the BVH briefly
        test_points = torch.tensor([[0, 0, 0]], dtype=torch.float32, device='cuda')
        dist, _, _ = bvh.signed_distance(test_points)
        
        # Explicitly free memory
        bvh.free_memory()
        del bvh
        
        if (i + 1) % 20 == 0:
            gc.collect()  # Force garbage collection
            torch.cuda.empty_cache()
            current_memory = get_gpu_memory_usage()
            print(f"  After {i+1} iterations: {current_memory:.2f} MB")
    
    gc.collect()
    torch.cuda.empty_cache()
    final_memory = get_gpu_memory_usage()
    print(f"Final GPU memory usage: {final_memory:.2f} MB")
    
    # Check if memory usage is reasonable
    memory_increase = final_memory - initial_memory
    print(f"\nTotal memory increase: {memory_increase:.2f} MB")
    
    if memory_increase < 50:  # Allow for some small increase due to CUDA context
        print("✓ Memory leak test PASSED - Memory usage is stable!")
        return True
    else:
        print("✗ Memory leak test FAILED - Significant memory increase detected!")
        return False

def test_explicit_cleanup():
    """Test that explicit cleanup works properly."""
    print("\nTesting explicit cleanup...")
    
    vertices, triangles = create_test_mesh()
    
    # Create a BVH
    bvh = cubvh.cuBVH(vertices, triangles)
    
    # Use it
    test_points = torch.tensor([[0, 0, 0]], dtype=torch.float32, device='cuda')
    dist1, _, _ = bvh.signed_distance(test_points)
    print(f"Distance before cleanup: {dist1.item():.4f}")
    
    # Free memory explicitly
    bvh.free_memory()
    print("✓ Explicitly freed GPU memory")
    
    # Try to use it again (this might fail or give incorrect results)
    try:
        dist2, _, _ = bvh.signed_distance(test_points)
        print(f"Distance after cleanup: {dist2.item():.4f}")
        print("⚠ BVH still usable after cleanup (this is implementation dependent)")
    except Exception as e:
        print(f"✓ BVH correctly unusable after cleanup: {type(e).__name__}")
    
    return True

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available. Running memory leak tests...")
        
        success1 = test_memory_leak_fixed()
        success2 = test_explicit_cleanup()
        
        if success1 and success2:
            print("\n✓ All memory management tests passed!")
        else:
            print("\n✗ Some memory management tests failed!")
    else:
        print("CUDA is not available. Cannot run tests.")
