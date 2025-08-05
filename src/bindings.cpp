// #include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <cubvh/api.h>


namespace py = pybind11;
using namespace cubvh;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

py::class_<cuBVH>(m, "cuBVH")
    .def("ray_trace", &cuBVH::ray_trace)
    .def("unsigned_distance", &cuBVH::unsigned_distance)
    .def("signed_distance", &cuBVH::signed_distance)
    .def("get_bvh_nodes", &cuBVH::get_bvh_nodes)
    .def("set_bvh_nodes", &cuBVH::set_bvh_nodes)
    .def("get_triangles", &cuBVH::get_triangles)
    .def("set_triangles", &cuBVH::set_triangles)
    .def("free_memory", &cuBVH::free_memory);

m.def("create_cuBVH", &create_cuBVH);

m.def("create_cuBVH_no_build", &create_cuBVH_no_build);

m.def("create_cuBVH_from_data", &create_cuBVH_from_data);

m.def("floodfill", &floodfill);

m.def("sparse_marching_cubes", &sparse_marching_cubes);

}