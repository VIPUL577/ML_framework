#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_fp16.h>
#include "seera_engine_cuda.hpp"



namespace py = pybind11;
using arr_f = py::array_t<float, py::array::c_style | py::array::forcecast>;
using arr_i = py::array_t<int32_t, py::array::c_style>;

PYBIND11_MODULE(seera_cuda, m) {
    m.doc() = "Seera CUDA ENGINE ACTIVATED!!!!";
    
}