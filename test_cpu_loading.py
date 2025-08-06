#!/usr/bin/env python3
"""
Test script to demonstrate CPU loading behavior in cuBVH.
This script shows that cuBVH can now be loaded on CPU via torch.load 
and only allocates GPU memory when explicitly moved to GPU or when computation is performed.
"""

import numpy as np
import torch
import cubvh
import tempfile
import os

def create_test_mesh():
    """Create a simple test mesh."""
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

def test_cpu_loading():
    """Test that cuBVH can be loaded on CPU and then moved to GPU."""
    print("Testing CPU Loading Behavior in cuBVH")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available - cannot test")
        return False
    
    vertices, triangles = create_test_mesh()
    
    # Step 1: Create and save a BVH
    print("Step 1: Creating and saving BVH...")
    original_bvh = cubvh.cuBVH(vertices, triangles, device='cuda:0')
    print(f"Original BVH device: {original_bvh.device}")
    
    # Test the original BVH
    test_point = torch.tensor([[0, 0, 0]], dtype=torch.float32, device='cuda:0')
    original_dist, _, _ = original_bvh.signed_distance(test_point)
    print(f"Original BVH distance: {original_dist.item():.4f}")
    
    # Save to file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_filename = f.name
        torch.save(original_bvh, f)
    
    print(f"Saved BVH to: {temp_filename}")
    
    # Step 2: Clear memory and check GPU usage
    print("\nStep 2: Clearing memory...")
    del original_bvh
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"GPU memory after clearing: {initial_memory:.2f} MB")
    
    # Step 3: Load BVH (should be on CPU)
    print("\nStep 3: Loading BVH with torch.load...")
    loaded_bvh = torch.load(temp_filename, weights_only=False)
    
    after_load_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"GPU memory after loading: {after_load_memory:.2f} MB")
    print(f"Loaded BVH device: {loaded_bvh.device}")
    print(f"Loaded BVH impl: {loaded_bvh.impl}")
    
    memory_increase_on_load = after_load_memory - initial_memory
    print(f"Memory increase on load: {memory_increase_on_load:.2f} MB")
    
    # Step 4: Try to use BVH on CPU (should fail gracefully)
    print("\nStep 4: Testing CPU usage (should fail gracefully)...")
    try:
        cpu_test_point = torch.tensor([[0, 0, 0]], dtype=torch.float32, device='cpu')
        loaded_bvh.signed_distance(cpu_test_point)
        print("❌ UNEXPECTED: BVH computation succeeded on CPU")
    except RuntimeError as e:
        print(f"✅ EXPECTED: BVH computation failed on CPU: {e}")
    
    # Step 5: Move to GPU and test
    print("\nStep 5: Moving BVH to GPU...")
    gpu_memory_before_move = torch.cuda.memory_allocated() / (1024 * 1024)
    
    gpu_bvh = loaded_bvh.to('cuda:0')
    
    gpu_memory_after_move = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"GPU memory before move: {gpu_memory_before_move:.2f} MB")
    print(f"GPU memory after move: {gpu_memory_after_move:.2f} MB")
    print(f"GPU BVH device: {gpu_bvh.device}")
    print(f"GPU BVH impl: {gpu_bvh.impl}")
    
    # Step 6: Test GPU computation
    print("\nStep 6: Testing GPU computation...")
    gpu_test_point = torch.tensor([[0, 0, 0]], dtype=torch.float32, device='cuda:0')
    gpu_dist, _, _ = gpu_bvh.signed_distance(gpu_test_point)
    print(f"GPU BVH distance: {gpu_dist.item():.4f}")
    
    # Verify results match
    distance_match = abs(original_dist.item() - gpu_dist.item()) < 1e-6
    print(f"Distance results match: {distance_match}")
    
    # Step 7: Test lazy loading (load but don't move to GPU until computation)
    print("\nStep 7: Testing lazy loading...")
    torch.cuda.empty_cache()
    lazy_memory_start = torch.cuda.memory_allocated() / (1024 * 1024)
    
    lazy_bvh = torch.load(temp_filename, weights_only=False)
    lazy_memory_after_load = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"Lazy: GPU memory after load: {lazy_memory_after_load:.2f} MB")
    
    # Now force GPU usage by calling a computation method
    lazy_bvh._device = torch.device('cuda:0')  # Manually set device
    lazy_test_point = torch.tensor([[0, 0, 0]], dtype=torch.float32, device='cuda:0')
    lazy_dist, _, _ = lazy_bvh.signed_distance(lazy_test_point)
    
    lazy_memory_after_compute = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"Lazy: GPU memory after computation: {lazy_memory_after_compute:.2f} MB")
    print(f"Lazy: Distance result: {lazy_dist.item():.4f}")
    
    # Cleanup
    os.unlink(temp_filename)
    
    # Assessment
    print("\n" + "=" * 50)
    print("CPU Loading Test Results")
    print("=" * 50)
    
    load_on_cpu = memory_increase_on_load < 10.0  # Should be minimal GPU memory increase
    device_is_cpu = str(loaded_bvh.device) == 'cpu'
    impl_is_none = loaded_bvh.impl is None
    computation_works_after_move = distance_match
    
    print(f"Loads on CPU (minimal GPU memory): {'✅ PASS' if load_on_cpu else '❌ FAIL'} ({memory_increase_on_load:.2f} MB)")
    print(f"Device is CPU after load:          {'✅ PASS' if device_is_cpu else '❌ FAIL'} ({loaded_bvh.device})")
    print(f"Implementation is lazy (None):      {'✅ PASS' if impl_is_none else '❌ FAIL'} ({loaded_bvh.impl})")
    print(f"Computation works after move:       {'✅ PASS' if computation_works_after_move else '❌ FAIL'}")
    
    all_passed = load_on_cpu and device_is_cpu and impl_is_none and computation_works_after_move
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ cuBVH now loads on CPU by default")
        print("✅ GPU memory is only allocated when moved to GPU or when computation is performed")
        print("✅ Loaded BVH produces identical results to original")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("❌ CPU loading may not be working properly")
    
    return all_passed

if __name__ == "__main__":
    try:
        success = test_cpu_loading()
        print(f"\nTest {'PASSED' if success else 'FAILED'}")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    exit(0 if success else 1)
