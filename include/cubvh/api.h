#pragma once

#include <Eigen/Dense>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <cuda_runtime.h>

using namespace Eigen;

using Verts = Matrix<float, Dynamic, 3, RowMajor>;
using Trigs = Matrix<uint32_t, Dynamic, 3, RowMajor>;

namespace cubvh {

// abstract class of raytracer
class cuBVH {
public:
    cuBVH() {}
    virtual ~cuBVH() {}

    virtual void ray_trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor positions, at::Tensor face_id, at::Tensor depth) = 0;
    virtual void unsigned_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw) = 0;
    virtual void signed_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw, uint32_t mode) = 0;
    
    // Methods for BVH node serialization
    virtual at::Tensor get_bvh_nodes() = 0;
    virtual void set_bvh_nodes(at::Tensor nodes_tensor) = 0;
};

// function to create an implementation of cuBVH
cuBVH* create_cuBVH(Ref<const Verts> vertices, Ref<const Trigs> triangles);

// function to create an implementation of cuBVH without building BVH (for loading from saved state)
cuBVH* create_cuBVH_no_build(Ref<const Verts> vertices, Ref<const Trigs> triangles);

// floodfill
at::Tensor floodfill(at::Tensor grid);

// sparse marching cubes
std::tuple<at::Tensor, at::Tensor> sparse_marching_cubes(at::Tensor coords, at::Tensor corners, double iso_d);
    
} // namespace cubvh