# CPU Loading in cuBVH

## Overview

cuBVH now supports **CPU loading** when using `torch.load()`. This means that when you load a saved cuBVH from disk, it will initially be on CPU and only allocate GPU memory when explicitly moved to GPU or when computation is performed.

## New Behavior

### Before (Previous Implementation)
```python
# Loading would immediately allocate GPU memory
bvh = torch.load('my_bvh.pkl')  # ❌ Immediate GPU allocation
print(bvh.device)  # cuda:0
```

### After (New Implementation)
```python
# Loading keeps BVH on CPU
bvh = torch.load('my_bvh.pkl')  # ✅ No GPU allocation
print(bvh.device)  # cpu
print(bvh.impl)    # None (lazy initialization)

# GPU memory only allocated when needed
bvh_gpu = bvh.to('cuda:0')  # Move to GPU when ready
# OR
result = bvh.signed_distance(points.cuda())  # Auto-creates GPU impl
```

## Benefits

1. **Memory Efficiency**: No immediate GPU memory allocation when loading
2. **Flexibility**: Load many BVHs without hitting GPU memory limits
3. **Lazy Initialization**: GPU resources only used when actually needed
4. **Better Resource Management**: Explicit control over when GPU memory is allocated

## Usage Patterns

### Pattern 1: Load and Explicitly Move to GPU
```python
# Load on CPU
bvh = torch.load('my_bvh.pkl')
print(f"Device after load: {bvh.device}")  # cpu

# Move to GPU when ready
bvh_gpu = bvh.to('cuda:0')
print(f"Device after move: {bvh_gpu.device}")  # cuda:0

# Use normally
distances = bvh_gpu.signed_distance(query_points)
```

### Pattern 2: Lazy GPU Allocation
```python
# Load on CPU
bvh = torch.load('my_bvh.pkl')  # No GPU memory used

# Set target device
bvh._device = torch.device('cuda:0')  # Still no GPU memory

# GPU implementation created on first use
query_points = torch.randn(100, 3, device='cuda:0')
distances = bvh.signed_distance(query_points)  # NOW GPU memory is allocated
```

### Pattern 3: Load Multiple BVHs Efficiently
```python
# Load multiple BVHs without GPU memory pressure
bvhs = []
for filename in bvh_files:
    bvh = torch.load(filename)  # All on CPU
    bvhs.append(bvh)

# Use them one at a time
for i, bvh in enumerate(bvhs):
    bvh_gpu = bvh.to('cuda:0')
    # Process with bvh_gpu...
    bvh_gpu.free_memory()  # Free GPU memory before next iteration
```

## Implementation Details

### Lazy Initialization
When a cuBVH is loaded via `torch.load()`:

1. **Device**: Set to `cpu`
2. **Implementation**: `impl = None` (lazy)
3. **Data**: BVH nodes and triangles kept on CPU
4. **GPU Allocation**: Deferred until first GPU operation

### Automatic GPU Activation
The GPU implementation is automatically created when:

1. **Explicit move**: `bvh.to('cuda:0')`
2. **Computation method called**: `ray_trace()`, `signed_distance()`, `unsigned_distance()`
3. **Target device is CUDA**: Methods check if device is CUDA and create impl

### Error Handling
```python
bvh = torch.load('my_bvh.pkl')  # Device is 'cpu'

# This will raise an informative error
try:
    result = bvh.signed_distance(cpu_points)
except RuntimeError as e:
    print(e)  # "cuBVH implementation requires a CUDA device. Use .to('cuda') to move to GPU first."
```

## Migration Guide

### For Existing Code
Most existing code will work without changes:

```python
# This still works fine
bvh = torch.load('my_bvh.pkl')
bvh_gpu = bvh.to('cuda:0')  # Explicit move
result = bvh_gpu.signed_distance(points)
```

### For New Code
Take advantage of the improved memory management:

```python
# Load without GPU allocation
bvh = torch.load('my_bvh.pkl')

# Check properties without GPU memory
print(f"Device: {bvh.device}")
print(f"BVH nodes shape: {bvh.get_bvh_nodes().shape}")

# Move to GPU only when needed
if need_gpu_computation:
    bvh = bvh.to('cuda:0')
```

## Testing

Use the provided test script to verify CPU loading behavior:

```bash
python test_cpu_loading.py
```

This script validates:
- ✅ BVH loads on CPU by default
- ✅ No immediate GPU memory allocation
- ✅ Lazy GPU implementation creation
- ✅ Correct computation results after GPU move
- ✅ Memory efficiency

## Backward Compatibility

The changes are fully backward compatible:

- **Old pickle files**: Can still be loaded (with automatic CPU loading)
- **Existing code**: Continues to work without modification
- **API**: No breaking changes to public methods

## Performance Considerations

- **Loading time**: Slightly faster (no immediate GPU allocation)
- **Memory usage**: Much more efficient for loading multiple BVHs
- **First computation**: Slightly slower (lazy initialization overhead)
- **Subsequent computations**: Identical performance

## Best Practices

1. **Load on CPU**: Let `torch.load()` put BVH on CPU initially
2. **Move explicitly**: Use `.to('cuda:0')` when ready for GPU computation
3. **Free memory**: Use `.free_memory()` to clean up when done
4. **Monitor usage**: Use `torch.cuda.memory_allocated()` to track GPU memory
