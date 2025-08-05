# Memory Management Fix for cuBVH

This document describes the memory leak fixes applied to cuBVH to prevent GPU memory accumulation when creating many BVH instances.

## Problem Description

The original cuBVH implementation had memory management issues where:

1. **GPUMemory class issues**: 
   - Move assignment operator didn't properly handle the `m_owned` flag
   - Missing copy assignment operator 
   - Destructor logic could be bypassed in some cases

2. **No explicit cleanup**: 
   - No way to manually free GPU memory when needed
   - Reliance only on destructors which might not be called immediately

3. **Reference counting issues**: 
   - Multiple references to GPU memory without proper ownership tracking

## Fixes Applied

### 1. GPUMemory Class Improvements

#### Fixed Move Assignment Operator
```cpp
GPUMemory<T>& operator=(GPUMemory<T>&& other) {
    if (this != &other) {
        // Free our current memory if we own it
        if (m_owned && m_data) {
            free_memory();
        }
        
        // Take ownership of other's data
        std::swap(m_data, other.m_data);
        std::swap(m_size, other.m_size);
        std::swap(m_owned, other.m_owned);
    }
    return *this;
}
```

#### Added Copy Assignment Operator
```cpp
GPUMemory<T>& operator=(const GPUMemory<T> &other) {
    if (this != &other) {
        // Free our current memory if we own it
        if (m_owned && m_data) {
            free_memory();
        }
        
        // Create a non-owning view of other's data
        m_data = other.m_data;
        m_size = other.m_size;
        m_owned = false;
    }
    return *this;
}
```

#### Improved Destructor
```cpp
~GPUMemory() {
#ifndef __CUDA_ARCH__
    if (!m_owned || !m_data) {
        return;
    }
    
    try {
        free_memory();
        m_size = 0;
    } catch (std::runtime_error error) {
        // Error handling...
    }
#endif
}
```

#### Added Explicit Free Method
```cpp
void free() {
    if (m_owned && m_data) {
        free_memory();
        m_size = 0;
    }
}
```

### 2. cuBVH Class Improvements

#### Added Memory Cleanup Interface
```cpp
class cuBVH {
    // ... existing methods ...
    virtual void free_memory() = 0;
};
```

#### Implemented in cuBVHImpl
```cpp
void free_memory() override {
    // Explicitly free GPU memory
    triangles_gpu.free();
    
    // Free BVH nodes GPU memory if available
    if (triangle_bvh) {
        triangle_bvh->free_gpu_memory();
    }
}
```

### 3. TriangleBvh Class Improvements

#### Added GPU Memory Cleanup
```cpp
class TriangleBvh {
    // ... existing methods ...
    void free_gpu_memory() {
        m_nodes_gpu.free();
    }
};
```

### 4. Python API Improvements

#### Added Explicit Cleanup Method
```python
def free_memory(self):
    """Explicitly free GPU memory used by this BVH."""
    if hasattr(self, 'impl') and self.impl is not None:
        self.impl.free_memory()
```

#### Added Destructor for Automatic Cleanup
```python
def __del__(self):
    """Destructor to ensure GPU memory is freed."""
    try:
        self.free_memory()
    except:
        # Ignore errors during destruction
        pass
```

## Usage Examples

### Automatic Memory Management
```python
import cubvh

# Memory is automatically freed when BVH goes out of scope
for i in range(1000):
    bvh = cubvh.cuBVH(vertices, triangles)
    # Use bvh...
    # Memory automatically freed when bvh is destroyed
```

### Explicit Memory Management
```python
import cubvh

# Manual control over memory cleanup
bvh = cubvh.cuBVH(vertices, triangles)
# Use bvh...

# Explicitly free memory when done
bvh.free_memory()
```

### Best Practices for Large-Scale Usage
```python
import cubvh
import gc
import torch

def process_many_meshes(mesh_list):
    for i, (vertices, triangles) in enumerate(mesh_list):
        bvh = cubvh.cuBVH(vertices, triangles)
        
        # Process the mesh...
        result = bvh.signed_distance(query_points)
        
        # Explicitly clean up every N iterations
        if i % 100 == 0:
            bvh.free_memory()
            gc.collect()
            torch.cuda.empty_cache()
```

## Performance Impact

The memory management fixes have minimal performance impact:

- **Memory allocation/deallocation**: Slightly more overhead due to proper ownership tracking
- **Explicit cleanup**: No performance impact on normal operations
- **Automatic cleanup**: Only affects object destruction, which should be rare in performance-critical code

## Testing

Use the provided test script to verify memory management:

```bash
python test_memory_fix.py
```

This script creates many BVH instances and monitors GPU memory usage to ensure no leaks occur.

## Backward Compatibility

All changes are backward compatible:
- Existing code continues to work without modifications
- New cleanup methods are optional
- Automatic cleanup happens transparently
