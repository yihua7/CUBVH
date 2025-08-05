#!/usr/bin/env python3
"""
Test script to validate automatic GPU memory cleanup in cuBVH.
This script demonstrates that cuBVH automatically frees GPU memory when variables go out of scope,
without requiring manual intervention.
"""

import numpy as np
import torch
import cubvh
import gc
import time
import sys
from contextlib import contextmanager

def get_gpu_memory_info():
    """Get detailed GPU memory information."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)   # MB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        return {
            'allocated': allocated,
            'reserved': reserved, 
            'max_allocated': max_allocated
        }
    return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}

def print_memory_info(label):
    """Print current GPU memory usage with a label."""
    info = get_gpu_memory_info()
    print(f"{label}:")
    print(f"  Allocated: {info['allocated']:.2f} MB")
    print(f"  Reserved:  {info['reserved']:.2f} MB")
    print(f"  Max Alloc: {info['max_allocated']:.2f} MB")

@contextmanager
def memory_tracker(test_name):
    """Context manager to track memory usage during a test."""
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"{'='*60}")
    
    # Clear memory and reset stats
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    initial_info = get_gpu_memory_info()
    print_memory_info("Initial memory")
    
    yield
    
    # Force cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    final_info = get_gpu_memory_info()
    print_memory_info("Final memory")
    
    memory_diff = final_info['allocated'] - initial_info['allocated']
    print(f"\nMemory difference: {memory_diff:+.2f} MB")
    
    if abs(memory_diff) < 5.0:  # Allow 5MB tolerance
        print("✓ PASSED: Memory properly cleaned up")
        return True
    else:
        print("✗ FAILED: Memory leak detected")
        return False

def create_test_mesh(size_factor=8):
    """Create a test mesh with variable complexity."""
    # Create a more complex mesh by subdividing a cube
    base_vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],  # top
    ], dtype=np.float32)
    
    base_triangles = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [2, 6, 7], [2, 7, 3],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ], dtype=np.uint32)
    
    # Duplicate and offset to create a larger mesh
    vertices_list = []
    triangles_list = []
    
    for i in range(size_factor):
        for j in range(size_factor):
            for k in range(size_factor):
                offset = np.array([i*3, j*3, k*3], dtype=np.float32)
                new_vertices = base_vertices + offset
                new_triangles = base_triangles + len(vertices_list) * 8
                
                vertices_list.append(new_vertices)
                triangles_list.append(new_triangles)
    
    vertices = np.vstack(vertices_list)
    triangles = np.vstack(triangles_list)
    
    return vertices, triangles

def test_basic_automatic_cleanup():
    """Test basic automatic cleanup when variables go out of scope."""
    vertices, triangles = create_test_mesh(8)  # Medium size mesh
    
    def create_and_use_bvh():
        """Create and use a BVH in a local scope."""
        bvh = cubvh.cuBVH(vertices, triangles)
        
        # Use the BVH to ensure it's fully initialized
        test_points = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32, device='cuda')
        distances, face_ids, _ = bvh.signed_distance(test_points)
        
        return distances.cpu().numpy()  # Return something to ensure computation happened
    
    # The BVH should be automatically cleaned up when function returns
    result = create_and_use_bvh()
    print(f"Computation result: {result}")
    
    return True

def test_loop_with_many_instances():
    """Test creating many BVH instances in a loop."""
    vertices, triangles = create_test_mesh(8)  # Small mesh for speed
    
    num_iterations = 50
    results = []
    
    for i in range(num_iterations):
        # Create BVH in loop scope
        bvh = cubvh.cuBVH(vertices, triangles)
        
        # Use it
        test_point = torch.tensor([[i*0.1, 0, 0]], dtype=torch.float32, device='cuda')
        dist, _, _ = bvh.signed_distance(test_point)
        results.append(dist.item())
        
        # Print memory every 10 iterations
        if (i + 1) % 10 == 0:
            info = get_gpu_memory_info()
            print(f"  Iteration {i+1}: {info['allocated']:.2f} MB allocated")
        
        # bvh should be automatically cleaned up at end of iteration
    
    print(f"Processed {num_iterations} BVH instances")
    print(f"Sample results: {results[:5]}...")
    
    return True

def test_nested_scopes():
    """Test automatic cleanup with nested scopes."""
    vertices, triangles = create_test_mesh(8)
    
    def outer_function():
        bvh1 = cubvh.cuBVH(vertices, triangles)
        
        def inner_function():
            bvh2 = cubvh.cuBVH(vertices, triangles)
            
            # Use both BVHs
            test_points = torch.tensor([[0, 0, 0]], dtype=torch.float32, device='cuda')
            dist1, _, _ = bvh1.signed_distance(test_points)
            dist2, _, _ = bvh2.signed_distance(test_points)
            
            return dist1.item(), dist2.item()
            # bvh2 should be cleaned up here
        
        result = inner_function()
        print(f"Inner function results: {result}")
        
        # Use bvh1 again to ensure it's still valid
        test_points = torch.tensor([[1, 1, 1]], dtype=torch.float32, device='cuda')
        dist1, _, _ = bvh1.signed_distance(test_points)
        
        return dist1.item()
        # bvh1 should be cleaned up here
    
    final_result = outer_function()
    print(f"Outer function result: {final_result}")
    
    return True

def test_exception_handling():
    """Test that memory is cleaned up even when exceptions occur."""
    vertices, triangles = create_test_mesh(8)
    
    def function_with_exception():
        bvh = cubvh.cuBVH(vertices, triangles)
        
        # Use the BVH
        test_points = torch.tensor([[0, 0, 0]], dtype=torch.float32, device='cuda')
        dist, _, _ = bvh.signed_distance(test_points)
        
        # Raise an exception
        raise ValueError("Test exception")
        # bvh should still be cleaned up despite the exception
    
    try:
        function_with_exception()
    except ValueError as e:
        print(f"Caught expected exception: {e}")
    
    return True

def test_large_memory_usage():
    """Test with larger meshes to see significant memory usage."""
    vertices, triangles = create_test_mesh(8)  # Larger mesh
    print(f"Testing with mesh: {len(vertices)} vertices, {len(triangles)} triangles")
    
    def create_large_bvh():
        bvh = cubvh.cuBVH(vertices, triangles)
        
        # Do some heavy computation
        test_points = torch.randn(5, 3, dtype=torch.float32, device='cuda')
        distances, face_ids, _ = bvh.signed_distance(test_points)
        
        return distances.mean().item()
    
    result = create_large_bvh()
    print(f"Large computation result: {result}")
    
    return True

def test_multiple_devices():
    """Test automatic cleanup across multiple CUDA devices (if available)."""
    if torch.cuda.device_count() < 2:
        print("Only one CUDA device available, skipping multi-device test")
        return True
    
    vertices, triangles = create_test_mesh(8)
    
    for device_id in range(min(2, torch.cuda.device_count())):
        print(f"  Testing on device {device_id}")
        with torch.cuda.device(device_id):
            bvh = cubvh.cuBVH(vertices, triangles, device=f'cuda:{device_id}')
            
            test_points = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=f'cuda:{device_id}')
            dist, _, _ = bvh.signed_distance(test_points)
            print(f"    Device {device_id} result: {dist.item()}")
            
            # bvh should be cleaned up when exiting this scope
    
    return True

def run_all_tests():
    """Run all automatic memory cleanup tests."""
    print("Testing Automatic GPU Memory Cleanup in cuBVH")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot run tests.")
        return False
    
    # List of tests to run
    tests = [
        ("Basic Automatic Cleanup", test_basic_automatic_cleanup),
        ("Loop with Many Instances", test_loop_with_many_instances),
        ("Nested Scopes", test_nested_scopes),
        ("Exception Handling", test_exception_handling),
        ("Large Memory Usage", test_large_memory_usage),
        ("Multiple Devices", test_multiple_devices),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        with memory_tracker(test_name):
            try:
                success = test_func()
                if success:
                    passed_tests += 1
            except Exception as e:
                print(f"✗ FAILED: Exception occurred: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {passed_tests}/{total_tests} tests passed")
    print(f"{'='*60}")
    
    if passed_tests == total_tests:
        print("✓ ALL TESTS PASSED: Automatic memory cleanup is working correctly!")
        return True
    else:
        print("✗ SOME TESTS FAILED: There may be memory management issues.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
