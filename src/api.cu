#include <cubvh/api.h>
#include <cubvh/common.h>
#include <cubvh/bvh.cuh>
#include <cubvh/floodfill.cuh>
#include <cubvh/spcumc.cuh>

#include <Eigen/Dense>

using namespace Eigen;

using Verts = Matrix<float, Dynamic, 3, RowMajor>;
using Trigs = Matrix<uint32_t, Dynamic, 3, RowMajor>;

namespace cubvh {

class cuBVHImpl : public cuBVH {
public:

    // accept numpy array (cpu) to init 
    cuBVHImpl(Ref<const Verts> vertices, Ref<const Trigs> triangles, bool build_bvh = true) : cuBVH() {

        const size_t n_vertices = vertices.rows();
        const size_t n_triangles = triangles.rows();

        triangles_cpu.resize(n_triangles);

        for (size_t i = 0; i < n_triangles; i++) {
            triangles_cpu[i] = {vertices.row(triangles(i, 0)), vertices.row(triangles(i, 1)), vertices.row(triangles(i, 2)), (int64_t)i};
        }

        if (build_bvh) {
            if (!triangle_bvh) {
                triangle_bvh = TriangleBvh::make();
            }

            triangle_bvh->build(triangles_cpu, 8);
        } else {
            // Just create the BVH object without building
            triangle_bvh = TriangleBvh::make();
        }
        triangles_gpu.resize_and_copy_from_host(triangles_cpu);

    }

    void ray_trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor positions, at::Tensor face_id, at::Tensor depth) {

        const uint32_t n_elements = rays_o.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        triangle_bvh->ray_trace_gpu(n_elements, rays_o.data_ptr<float>(), rays_d.data_ptr<float>(), positions.data_ptr<float>(), face_id.data_ptr<int64_t>(), depth.data_ptr<float>(), triangles_gpu.data(), stream);
    }

    void unsigned_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw) {

        const uint32_t n_elements = positions.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        triangle_bvh->unsigned_distance_gpu(n_elements, positions.data_ptr<float>(), distances.data_ptr<float>(), face_id.data_ptr<int64_t>(), uvw.has_value() ? uvw.value().data_ptr<float>() : nullptr, triangles_gpu.data(), stream);

    }

    void signed_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw, uint32_t mode) {

        const uint32_t n_elements = positions.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        triangle_bvh->signed_distance_gpu(n_elements, mode, positions.data_ptr<float>(), distances.data_ptr<float>(), face_id.data_ptr<int64_t>(), uvw.has_value() ? uvw.value().data_ptr<float>() : nullptr, triangles_gpu.data(), stream);
    }

    at::Tensor get_bvh_nodes() override {
        const auto& nodes = triangle_bvh->get_nodes();
        size_t n_nodes = nodes.size();
        
        // Each node: [min_x, min_y, min_z, max_x, max_y, max_z, left_idx, right_idx]
        at::Tensor nodes_tensor = at::zeros({(int64_t)n_nodes, 8}, at::TensorOptions().dtype(torch::kFloat32));
        float* data = nodes_tensor.data_ptr<float>();
        
        for (size_t i = 0; i < n_nodes; ++i) {
            const auto& node = nodes[i];
            data[i * 8 + 0] = node.bb.min.x();
            data[i * 8 + 1] = node.bb.min.y();
            data[i * 8 + 2] = node.bb.min.z();
            data[i * 8 + 3] = node.bb.max.x();
            data[i * 8 + 4] = node.bb.max.y();
            data[i * 8 + 5] = node.bb.max.z();
            data[i * 8 + 6] = static_cast<float>(node.left_idx);
            data[i * 8 + 7] = static_cast<float>(node.right_idx);
        }
        
        return nodes_tensor;
    }
    
    void set_bvh_nodes(at::Tensor nodes_tensor) override {
        TORCH_CHECK(nodes_tensor.dim() == 2 && nodes_tensor.size(1) == 8, 
                   "BVH nodes tensor must have shape [N, 8]");
        
        nodes_tensor = nodes_tensor.contiguous().cpu();
        float* data = nodes_tensor.data_ptr<float>();
        size_t n_nodes = nodes_tensor.size(0);
        
        std::vector<TriangleBvhNode> nodes(n_nodes);
        for (size_t i = 0; i < n_nodes; ++i) {
            auto& node = nodes[i];
            node.bb.min.x() = data[i * 8 + 0];
            node.bb.min.y() = data[i * 8 + 1];
            node.bb.min.z() = data[i * 8 + 2];
            node.bb.max.x() = data[i * 8 + 3];
            node.bb.max.y() = data[i * 8 + 4];
            node.bb.max.z() = data[i * 8 + 5];
            node.left_idx = static_cast<int>(data[i * 8 + 6]);
            node.right_idx = static_cast<int>(data[i * 8 + 7]);
        }
        
        triangle_bvh->set_nodes(nodes);
    }

    at::Tensor get_triangles() override {
        size_t n_triangles = triangles_cpu.size();
        
        // Each triangle: [ax, ay, az, bx, by, bz, cx, cy, cz, id]
        at::Tensor triangles_tensor = at::zeros({(int64_t)n_triangles, 10}, at::TensorOptions().dtype(torch::kFloat32));
        float* data = triangles_tensor.data_ptr<float>();
        
        for (size_t i = 0; i < n_triangles; ++i) {
            const auto& triangle = triangles_cpu[i];
            data[i * 10 + 0] = triangle.a.x();
            data[i * 10 + 1] = triangle.a.y();
            data[i * 10 + 2] = triangle.a.z();
            data[i * 10 + 3] = triangle.b.x();
            data[i * 10 + 4] = triangle.b.y();
            data[i * 10 + 5] = triangle.b.z();
            data[i * 10 + 6] = triangle.c.x();
            data[i * 10 + 7] = triangle.c.y();
            data[i * 10 + 8] = triangle.c.z();
            data[i * 10 + 9] = static_cast<float>(triangle.id);
        }
        
        return triangles_tensor;
    }
    
    void set_triangles(at::Tensor triangles_tensor) override {
        TORCH_CHECK(triangles_tensor.dim() == 2 && triangles_tensor.size(1) == 10, 
                   "Triangles tensor must have shape [N, 10]");
        
        triangles_tensor = triangles_tensor.contiguous().cpu();
        float* data = triangles_tensor.data_ptr<float>();
        size_t n_triangles = triangles_tensor.size(0);
        
        triangles_cpu.resize(n_triangles);
        for (size_t i = 0; i < n_triangles; ++i) {
            auto& triangle = triangles_cpu[i];
            triangle.a.x() = data[i * 10 + 0];
            triangle.a.y() = data[i * 10 + 1];
            triangle.a.z() = data[i * 10 + 2];
            triangle.b.x() = data[i * 10 + 3];
            triangle.b.y() = data[i * 10 + 4];
            triangle.b.z() = data[i * 10 + 5];
            triangle.c.x() = data[i * 10 + 6];
            triangle.c.y() = data[i * 10 + 7];
            triangle.c.z() = data[i * 10 + 8];
            triangle.id = static_cast<int64_t>(data[i * 10 + 9]);
        }
        
        // Update GPU memory with new triangles
        triangles_gpu.resize_and_copy_from_host(triangles_cpu);
    }

    std::vector<Triangle> triangles_cpu;
    GPUMemory<Triangle> triangles_gpu;
    std::shared_ptr<TriangleBvh> triangle_bvh;
};
    
cuBVH* create_cuBVH(Ref<const Verts> vertices, Ref<const Trigs> triangles) {
    return new cuBVHImpl{vertices, triangles, true};
}

cuBVH* create_cuBVH_no_build(Ref<const Verts> vertices, Ref<const Trigs> triangles) {
    return new cuBVHImpl{vertices, triangles, false};
}

cuBVH* create_cuBVH_from_data(Ref<const Verts> vertices, Ref<const Trigs> triangles, at::Tensor bvh_nodes, at::Tensor triangles_data) {
    cuBVHImpl* impl = new cuBVHImpl{vertices, triangles, false};
    impl->set_bvh_nodes(bvh_nodes);
    impl->set_triangles(triangles_data);
    return impl;
}

at::Tensor floodfill(at::Tensor grid) {

    // assert grid is uint8_t
    assert(grid.dtype() == at::ScalarType::Bool);

    const int B = grid.size(0);
    const int H = grid.size(1);
    const int W = grid.size(2);
    const int D = grid.size(3);

    // allocate mask
    at::Tensor mask = at::zeros({B, H, W, D}, at::device(grid.device()).dtype(at::ScalarType::Int));

    _floodfill_batch(grid.data_ptr<bool>(), B, H, W, D, mask.data_ptr<int32_t>());

    return mask;
}

std::tuple<at::Tensor, at::Tensor> sparse_marching_cubes(
    at::Tensor coords,        // [N,3] int32, cuda
    at::Tensor corners,       // [N,8] float32, cuda
    double iso_d)             // (PyTorch passes double ⇒ cast to float)
{
    TORCH_CHECK(coords.is_cuda(),  "coords must reside on CUDA");
    TORCH_CHECK(corners.is_cuda(), "corners must reside on CUDA");
    TORCH_CHECK(coords.dtype()  == torch::kInt32,   "coords must be int32");
    TORCH_CHECK(corners.dtype() == torch::kFloat32, "corners must be float32");
    TORCH_CHECK(coords.sizes().size()  == 2 && coords.size(1)  == 3,
                "coords must be of shape [N,3]");
    TORCH_CHECK(corners.sizes().size() == 2 && corners.size(1) == 8,
                "corners must be of shape [N,8]");
    TORCH_CHECK(coords.size(0) == corners.size(0),
                "coords and corners must have the same first-dim (N)");

    // Ensure contiguous memory - PyTorch extensions expect this.
    coords  = coords.contiguous();
    corners = corners.contiguous();
    const int    N   = static_cast<int>(coords.size(0));
    const int   *d_coords  = coords.data_ptr<int>();
    const float *d_corners = corners.data_ptr<float>();
    const float  iso       = static_cast<float>(iso_d);

    // Use the current PyTorch CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // --- call the CUDA sparse MC core (header we wrote earlier) -------------------
    auto mesh = _sparse_marching_cubes(d_coords, d_corners, N, iso, stream);
    thrust::device_vector<V3f> &verts_vec = mesh.first;
    thrust::device_vector<Tri> &tris_vec  = mesh.second;
    const int64_t M = static_cast<int64_t>(verts_vec.size());
    const int64_t T = static_cast<int64_t>(tris_vec.size());

    // --- create output tensors ----------------------------------------------------
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(coords.device());
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(coords.device());

    at::Tensor verts = at::empty({M, 3}, opts_f);
    at::Tensor tris  = at::empty({T, 3}, opts_i);

    // Copy GPU→GPU (same stream ⇒ async & cheap)
    cudaMemcpyAsync(verts.data_ptr<float>(),
                    thrust::raw_pointer_cast(verts_vec.data()),
                    M * 3 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    cudaMemcpyAsync(tris.data_ptr<int>(),
                    thrust::raw_pointer_cast(tris_vec.data()),
                    T * 3 * sizeof(int),
                    cudaMemcpyDeviceToDevice, stream);

    // Make sure copies finish before we free device_vectors
    cudaStreamSynchronize(stream);

    return {verts, tris};
}


} // namespace cubvh