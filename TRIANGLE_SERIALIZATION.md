# Triangle Serialization in cuBVH

This document describes the new triangle serialization functionality added to cuBVH that allows you to save and restore the sorted triangle data along with BVH nodes.

## Background

When cuBVH builds a BVH (Bounding Volume Hierarchy), it reorders the triangles in memory to optimize spatial locality. This reordering is crucial for BVH performance, but it means that the triangle data after building is different from the original input.

Previously, when saving/loading a BVH, only the BVH nodes were serialized, and the triangles were reconstructed from the original vertex/triangle data. This worked but lost the optimized triangle ordering.

## New Functionality

### C++ API

The `cuBVH` class now has additional methods:

```cpp
// Get sorted triangles as a tensor [N, 10] where each row is:
// [ax, ay, az, bx, by, bz, cx, cy, cz, original_id]
virtual at::Tensor get_triangles() = 0;

// Set triangles from a tensor with the same format
virtual void set_triangles(at::Tensor triangles_tensor) = 0;
```

### Python API

The `cuBVH` class constructor now accepts an optional `triangles_data` parameter:

```python
bvh = cuBVH(vertices, triangles, bvh_nodes=None, triangles_data=None, device=None)
```

New methods:

```python
# Get the sorted triangles data as a tensor [N, 10]
triangles_data = bvh.get_triangles_data()

# Get the sorted triangles in a more usable format
vertices, triangles, triangle_ids = bvh.get_sorted_triangles()
```

### Tensor Format

The triangles data tensor has shape `[N, 10]` where each row contains:
- `[0:3]`: First vertex (ax, ay, az) 
- `[3:6]`: Second vertex (bx, by, bz)
- `[6:9]`: Third vertex (cx, cy, cz)
- `[9]`: Original triangle ID

## Usage Examples

### Saving and Loading BVH State

```python
import numpy as np
import torch
import cubvh

# Create original BVH
vertices = np.array([...])  # Your vertex data
triangles = np.array([...]) # Your triangle indices
bvh = cubvh.cuBVH(vertices, triangles)

# Save BVH state
bvh_nodes = bvh.get_bvh_nodes()
triangles_data = bvh.get_triangles_data()

# Later, restore BVH from saved state
bvh_restored = cubvh.cuBVH(vertices, triangles, 
                          bvh_nodes=bvh_nodes, 
                          triangles_data=triangles_data)
```

### Working with Sorted Triangles

```python
# Get the triangles in their BVH-optimized order
sorted_vertices, sorted_triangles, triangle_ids = bvh.get_sorted_triangles()

print(f"Original triangle order: {np.arange(len(triangles))}")
print(f"BVH-optimized order: {triangle_ids}")
```

### Pickling Support

The triangle data is automatically included in pickle serialization:

```python
import pickle

# Save BVH to file
with open('bvh.pkl', 'wb') as f:
    pickle.dump(bvh, f)

# Load BVH from file
with open('bvh.pkl', 'rb') as f:
    bvh_loaded = pickle.load(f)

# The loaded BVH will have the same optimized triangle ordering
```

## Performance Benefits

1. **Faster Loading**: When restoring a BVH, you avoid the triangle reordering step since the triangles are already in their optimized order.

2. **Exact Reconstruction**: The restored BVH will have exactly the same triangle ordering and memory layout as the original, ensuring identical results.

3. **Better Serialization**: The complete BVH state can be saved and restored without any loss of optimization.

## Backward Compatibility

The new functionality is fully backward compatible:
- Existing code will continue to work without changes
- Old pickle files can still be loaded (they just won't have the optimized triangle ordering)
- The `triangles_data` parameter is optional

## Testing

A test script is provided in `test_triangle_serialization.py` that demonstrates the functionality and verifies that restored BVHs produce identical results to the originals.
