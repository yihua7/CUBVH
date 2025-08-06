#!/usr/bin/env python3
"""
Enhanced test script for CPU loading behavior in cuBVH.
This version handles PyTorch 2.6+ changes to torch.load defaults and provides
comprehensive testing of the CPU loading functionality.
"""

import numpy as np
import torch
import cubvh
import tempfile
import os
import sys

def safe_torch_load(filename):
    """Safely load a torch file, handling both old and new PyTorch versions."""
    try:
        # Try with weights_only=False first (works with all versions)
        return torch.load(filename, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't have weights_only parameter
        return torch.load(filename)
    except Exception as e:
        print(f"Failed to load {filename}: {e}")
        raise

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

def test_pytorch_version_compatibility():
    """Test PyTorch version and torch.load behavior."""
    print(f"PyTorch version: {torch.__version__}")
    
    # Test torch.load signature
    import inspect
    sig = inspect.signature(torch.load)
    has_weights_only = 'weights_only' in sig.parameters
    print(f"torch.load has weights_only parameter: {has_weights_only}")
    
    if has_weights_only:
        default_weights_only = sig.parameters['weights_only'].default
        print(f"Default weights_only value: {default_weights_only}")
    
    return has_weights_only

def test_cpu_loading():
    """Test that cuBVH can be loaded on CPU and then moved to GPU."""
    print("Testing CPU Loading Behavior in cuBVH")
    print("=" * 50)
    
    # Check PyTorch compatibility
    has_weights_only = test_pytorch_version_compatibility()
    print()
    
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
    try:
        loaded_bvh = safe_torch_load(temp_filename)
        load_success = True
    except Exception as e:
        print(f"❌ Failed to load BVH: {e}")
        load_success = False
        loaded_bvh = None
    
    if not load_success:
        os.unlink(temp_filename)
        return False
    
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
        cpu_fails_correctly = False
    except RuntimeError as e:
        print(f"✅ EXPECTED: BVH computation failed on CPU: {e}")
        cpu_fails_correctly = True
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        cpu_fails_correctly = False
    
    # Step 5: Move to GPU and test
    print("\nStep 5: Moving BVH to GPU...")
    gpu_memory_before_move = torch.cuda.memory_allocated() / (1024 * 1024)
    
    try:
        gpu_bvh = loaded_bvh.to('cuda:0')
        move_success = True
    except Exception as e:
        print(f"❌ Failed to move BVH to GPU: {e}")
        move_success = False
        gpu_bvh = None
    
    if not move_success:
        os.unlink(temp_filename)
        return False
    
    gpu_memory_after_move = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"GPU memory before move: {gpu_memory_before_move:.2f} MB")
    print(f"GPU memory after move: {gpu_memory_after_move:.2f} MB")
    print(f"GPU BVH device: {gpu_bvh.device}")
    print(f"GPU BVH impl: {gpu_bvh.impl}")
    
    # Step 6: Test GPU computation
    print("\nStep 6: Testing GPU computation...")
    try:
        gpu_test_point = torch.tensor([[0, 0, 0]], dtype=torch.float32, device='cuda:0')
        gpu_dist, _, _ = gpu_bvh.signed_distance(gpu_test_point)
        print(f"GPU BVH distance: {gpu_dist.item():.4f}")
        
        # Verify results match
        distance_match = abs(original_dist.item() - gpu_dist.item()) < 1e-6
        print(f"Distance results match: {distance_match}")
        computation_success = True
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        distance_match = False
        computation_success = False
    
    # Step 7: Test lazy loading (load but don't move to GPU until computation)
    print("\nStep 7: Testing lazy loading...")
    torch.cuda.empty_cache()
    lazy_memory_start = torch.cuda.memory_allocated() / (1024 * 1024)
    
    try:
        lazy_bvh = safe_torch_load(temp_filename)
        lazy_memory_after_load = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"Lazy: GPU memory after load: {lazy_memory_after_load:.2f} MB")
        
        # Now force GPU usage by calling a computation method
        lazy_bvh._device = torch.device('cuda:0')  # Manually set device
        lazy_test_point = torch.tensor([[0, 0, 0]], dtype=torch.float32, device='cuda:0')
        lazy_dist, _, _ = lazy_bvh.signed_distance(lazy_test_point)
        
        lazy_memory_after_compute = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"Lazy: GPU memory after computation: {lazy_memory_after_compute:.2f} MB")
        print(f"Lazy: Distance result: {lazy_dist.item():.4f}")
        lazy_test_success = True
    except Exception as e:
        print(f"❌ Lazy loading test failed: {e}")
        lazy_test_success = False
    
    # Cleanup
    os.unlink(temp_filename)
    
    # Assessment
    print("\n" + "=" * 50)
    print("CPU Loading Test Results")
    print("=" * 50)
    
    load_on_cpu = memory_increase_on_load < 10.0  # Should be minimal GPU memory increase
    device_is_cpu = str(loaded_bvh.device) == 'cpu'
    impl_is_none = loaded_bvh.impl is None
    computation_works_after_move = distance_match and computation_success
    
    print(f"PyTorch compatibility:              {'✅ PASS' if has_weights_only else '⚠ OLD VERSION'}")
    print(f"BVH loading successful:             {'✅ PASS' if load_success else '❌ FAIL'}")
    print(f"Loads on CPU (minimal GPU memory):  {'✅ PASS' if load_on_cpu else '❌ FAIL'} ({memory_increase_on_load:.2f} MB)")
    print(f"Device is CPU after load:           {'✅ PASS' if device_is_cpu else '❌ FAIL'} ({loaded_bvh.device})")
    print(f"Implementation is lazy (None):       {'✅ PASS' if impl_is_none else '❌ FAIL'} ({loaded_bvh.impl})")
    print(f"CPU computation fails correctly:     {'✅ PASS' if cpu_fails_correctly else '❌ FAIL'}")
    print(f"GPU move successful:                 {'✅ PASS' if move_success else '❌ FAIL'}")
    print(f"Computation works after move:        {'✅ PASS' if computation_works_after_move else '❌ FAIL'}")
    print(f"Lazy loading works:                  {'✅ PASS' if lazy_test_success else '❌ FAIL'}")
    
    # Calculate overall success
    critical_tests = [
        load_success,
        load_on_cpu,
        device_is_cpu,
        impl_is_none,
        cpu_fails_correctly,
        move_success,
        computation_works_after_move
    ]
    
    all_passed = all(critical_tests)
    
    if all_passed:
        print("\n🎉 ALL CRITICAL TESTS PASSED!")
        print("✅ cuBVH now loads on CPU by default")
        print("✅ GPU memory is only allocated when moved to GPU or when computation is performed")
        print("✅ Loaded BVH produces identical results to original")
        print("✅ Error handling works correctly")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("❌ CPU loading may not be working properly")
        
        # Print specific failure information
        failed_tests = []
        if not load_success: failed_tests.append("BVH loading")
        if not load_on_cpu: failed_tests.append("CPU loading")
        if not device_is_cpu: failed_tests.append("CPU device")
        if not impl_is_none: failed_tests.append("lazy implementation")
        if not cpu_fails_correctly: failed_tests.append("CPU error handling")
        if not move_success: failed_tests.append("GPU move")
        if not computation_works_after_move: failed_tests.append("GPU computation")
        
        print(f"Failed: {', '.join(failed_tests)}")
    
    return all_passed

def test_safe_loading_patterns():
    """Test various safe loading patterns."""
    print("\n" + "=" * 50)
    print("Testing Safe Loading Patterns")
    print("=" * 50)
    
    vertices, triangles = create_test_mesh()
    
    # Create and save a BVH
    bvh = cubvh.cuBVH(vertices, triangles, device='cuda:0')
    
    patterns = [
        ("Standard save/load", lambda f: torch.save(bvh, f)),
        ("With protocol", lambda f: torch.save(bvh, f, pickle_protocol=4)),
    ]
    
    for pattern_name, save_func in patterns:
        print(f"\nTesting {pattern_name}...")
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_filename = f.name
            try:
                save_func(f)
                print(f"  Save: ✅ Success")
                
                # Test loading
                loaded = safe_torch_load(temp_filename)
                print(f"  Load: ✅ Success (device: {loaded.device})")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
            finally:
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)

if __name__ == "__main__":
    try:
        # Run main CPU loading test
        success = test_cpu_loading()
        
        # Run additional pattern tests
        test_safe_loading_patterns()
        
        print(f"\nOverall test result: {'PASSED' if success else 'FAILED'}")
        
        if not success:
            print("\nTroubleshooting tips:")
            print("1. Ensure you're using a compatible PyTorch version")
            print("2. Check that CUDA is available and working")
            print("3. Verify cuBVH is properly compiled and installed")
            print("4. Try updating to the latest PyTorch version")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    sys.exit(0 if success else 1)
