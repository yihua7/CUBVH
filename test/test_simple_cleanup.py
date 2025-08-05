#!/usr/bin/env python3
"""
Simple test to validate automatic GPU memory cleanup in cuBVH.
This script focuses specifically on testing that GPU memory is automatically 
freed when cuBVH variables go out of scope.
"""

import numpy as np
import torch
import cubvh
import gc
import time

def monitor_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0

def create_cube_mesh():
    """Create a simple cube mesh for testing."""
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=np.float32)
    
    triangles = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
    ], dtype=np.uint32)
    
    return vertices, triangles

def test_automatic_cleanup():
    """Test that cuBVH automatically cleans up GPU memory when variables die."""
    
    print("Testing Automatic GPU Memory Cleanup in cuBVH")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available - cannot test")
        return False
    
    # Clear initial memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    initial_memory = monitor_memory()
    print(f"Initial GPU memory: {initial_memory:.2f} MB")
    
    vertices, triangles = create_cube_mesh()
    print(f"Created test mesh: {len(vertices)} vertices, {len(triangles)} triangles")
    
    # Test 1: Single BVH creation and destruction
    print("\nTest 1: Single BVH automatic cleanup")
    print("-" * 30)
    
    def create_single_bvh():
        bvh = cubvh.cuBVH(vertices, triangles)
        
        # Use the BVH to ensure memory is allocated
        test_point = torch.tensor([[0, 0, 0]], dtype=torch.float32, device='cuda')
        dist, _, _ = bvh.signed_distance(test_point)
        
        during_memory = monitor_memory()
        print(f"Memory during BVH usage: {during_memory:.2f} MB")
        
        return dist.item()
    
    result = create_single_bvh()
    # Force garbage collection
    gc.collect()
    
    after_single_memory = monitor_memory()
    print(f"Memory after function return: {after_single_memory:.2f} MB")
    print(f"Result: {result:.4f}")
    
    single_cleanup_diff = after_single_memory - initial_memory
    print(f"Memory difference: {single_cleanup_diff:+.2f} MB")
    
    # Test 2: Multiple BVH instances in a loop
    print("\nTest 2: Multiple BVH instances in loop")
    print("-" * 30)
    
    memory_history = []
    num_iterations = 20
    
    for i in range(num_iterations):
        # Create BVH in loop scope - should be automatically cleaned up each iteration
        bvh = cubvh.cuBVH(vertices, triangles)
        
        # Use it
        test_point = torch.tensor([[i * 0.1, 0, 0]], dtype=torch.float32, device='cuda')
        dist, _, _ = bvh.signed_distance(test_point)
        
        current_memory = monitor_memory()
        memory_history.append(current_memory)
        
        if (i + 1) % 5 == 0:
            print(f"Iteration {i+1:2d}: {current_memory:.2f} MB")
        
        # bvh should be automatically destroyed here
    
    # Force cleanup
    gc.collect()
    final_memory = monitor_memory()
    
    print(f"\nFinal memory after loop: {final_memory:.2f} MB")
    loop_cleanup_diff = final_memory - initial_memory
    print(f"Total memory difference: {loop_cleanup_diff:+.2f} MB")
    
    # Test 3: Memory stability check
    print("\nTest 3: Memory stability analysis")
    print("-" * 30)
    
    # Check if memory usage is stable (not continuously growing)
    if len(memory_history) >= 10:
        first_half = memory_history[:len(memory_history)//2]
        second_half = memory_history[len(memory_history)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        print(f"First half average: {first_avg:.2f} MB")
        print(f"Second half average: {second_avg:.2f} MB")
        print(f"Growth rate: {second_avg - first_avg:+.2f} MB")
        
        stable = abs(second_avg - first_avg) < 10.0  # Allow 10MB variation
    else:
        stable = True
    
    # Test 4: Nested scope test
    print("\nTest 4: Nested scope cleanup")
    print("-" * 30)
    
    def outer_scope():
        bvh_outer = cubvh.cuBVH(vertices, triangles)
        outer_memory = monitor_memory()
        print(f"Outer scope memory: {outer_memory:.2f} MB")
        
        def inner_scope():
            bvh_inner = cubvh.cuBVH(vertices, triangles)
            inner_memory = monitor_memory()
            print(f"Inner scope memory: {inner_memory:.2f} MB")
            
            # Use both BVHs
            test_point = torch.tensor([[0, 0, 0]], dtype=torch.float32, device='cuda')
            dist1, _, _ = bvh_outer.signed_distance(test_point)
            dist2, _, _ = bvh_inner.signed_distance(test_point)
            
            return dist1.item(), dist2.item()
            # bvh_inner should be cleaned up here
        
        results = inner_scope()
        
        # Check memory after inner scope
        after_inner_memory = monitor_memory()
        print(f"After inner scope: {after_inner_memory:.2f} MB")
        
        return results
        # bvh_outer should be cleaned up here
    
    nested_results = outer_scope()
    gc.collect()
    
    after_nested_memory = monitor_memory()
    print(f"After nested test: {after_nested_memory:.2f} MB")
    nested_cleanup_diff = after_nested_memory - initial_memory
    print(f"Nested test memory difference: {nested_cleanup_diff:+.2f} MB")
    
    # Final assessment
    print("\n" + "=" * 50)
    print("AUTOMATIC CLEANUP TEST RESULTS")
    print("=" * 50)
    
    # Check if memory differences are reasonable (< 5MB tolerance)
    single_ok = abs(single_cleanup_diff) < 5.0
    loop_ok = abs(loop_cleanup_diff) < 5.0
    nested_ok = abs(nested_cleanup_diff) < 5.0
    
    print(f"Single BVH cleanup:     {'✓ PASS' if single_ok else '✗ FAIL'} ({single_cleanup_diff:+.2f} MB)")
    print(f"Loop BVH cleanup:       {'✓ PASS' if loop_ok else '✗ FAIL'} ({loop_cleanup_diff:+.2f} MB)")
    print(f"Nested scope cleanup:   {'✓ PASS' if nested_ok else '✗ FAIL'} ({nested_cleanup_diff:+.2f} MB)")
    print(f"Memory stability:       {'✓ PASS' if stable else '✗ FAIL'}")
    
    all_passed = single_ok and loop_ok and nested_ok and stable
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✓ cuBVH automatically cleans up GPU memory when variables die")
        print("✓ No memory leaks detected")
        print("✓ Memory usage is stable across multiple instances")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("✗ There may be memory management issues")
        print("✗ Manual cleanup might be necessary")
    
    return all_passed

if __name__ == "__main__":
    success = test_automatic_cleanup()
    
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)
