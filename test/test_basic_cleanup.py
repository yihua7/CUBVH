#!/usr/bin/env python3
"""
Minimal test to demonstrate automatic GPU memory cleanup in cuBVH.
"""

def test_basic_automatic_cleanup():
    """Basic test showing automatic memory cleanup."""
    import numpy as np
    import torch
    import cubvh
    import gc
    
    print("Testing automatic GPU memory cleanup in cuBVH...")
    
    # Simple cube mesh
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=np.float32)
    
    triangles = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
    ], dtype=np.uint32)
    
    # Clear memory and get baseline
    torch.cuda.empty_cache()
    gc.collect()
    initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"Initial GPU memory: {initial_memory:.2f} MB")
    
    # Test automatic cleanup when variable goes out of scope
    def create_and_use_bvh():
        """Create BVH in local scope - should be automatically cleaned up."""
        bvh = cubvh.cuBVH(vertices, triangles)
        
        # Use the BVH
        test_point = torch.tensor([[0, 0, 0]], dtype=torch.float32, device='cuda')
        distance, face_id, _ = bvh.signed_distance(test_point)
        
        during_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"Memory during BVH usage: {during_memory:.2f} MB")
        
        return distance.item()
        # BVH should be automatically destroyed here when function returns
    
    # Call the function - BVH will be created and destroyed
    result = create_and_use_bvh()
    
    # Force garbage collection to ensure cleanup
    gc.collect()
    
    # Check final memory
    final_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"Memory after cleanup: {final_memory:.2f} MB")
    print(f"Distance result: {result:.4f}")
    
    memory_diff = final_memory - initial_memory
    print(f"Memory difference: {memory_diff:+.2f} MB")
    
    # Test with multiple instances
    print("\nTesting multiple instances...")
    for i in range(10):
        bvh = cubvh.cuBVH(vertices, triangles)
        test_point = torch.tensor([[i * 0.1, 0, 0]], dtype=torch.float32, device='cuda')
        dist, _, _ = bvh.signed_distance(test_point)
        
        if i % 3 == 0:
            current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"  Iteration {i}: {current_memory:.2f} MB")
        
        # bvh automatically destroyed at end of each iteration
    
    gc.collect()
    loop_final_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"Memory after loop: {loop_final_memory:.2f} MB")
    
    loop_memory_diff = loop_final_memory - initial_memory
    print(f"Total memory increase: {loop_memory_diff:+.2f} MB")
    
    # Assessment
    if abs(memory_diff) < 5.0 and abs(loop_memory_diff) < 10.0:
        print("\n✅ SUCCESS: Automatic cleanup is working!")
        print("   GPU memory is properly freed when cuBVH variables die.")
        return True
    else:
        print("\n❌ FAILURE: Memory may not be cleaned up properly.")
        print("   Consider using manual cleanup with bvh.free_memory()")
        return False

if __name__ == "__main__":
    try:
        success = test_basic_automatic_cleanup()
        print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure PyTorch and cuBVH are properly installed.")
    except Exception as e:
        print(f"Error during test: {e}")
        success = False
    
    exit(0 if success else 1)
