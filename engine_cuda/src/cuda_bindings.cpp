#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <seera_engine_cuda.hpp>

namespace py = pybind11;

using arr_f = py::array_t<float, py::array::c_style | py::array::forcecast>;
using arr_i = py::array_t<int32_t, py::array::c_style>;
using arr_s = py::array_t<int16_t, py::array::c_style>;
PYBIND11_MODULE(seera_cuda, m)
{
      m.doc() = "Seera CUDA ENGINE ACTIVATED!!!!";

      // ══════════════════════════════════════════════════════════════════
      // Memory Management
      // ══════════════════════════════════════════════════════════════════

      // cuda_malloc: allocate n_bytes on GPU, return GPU address
      m.def("cuda_malloc", [](size_t n_bytes) -> uintptr_t
            {
        void *ptr;
        cudaMalloc(&ptr, n_bytes);
        return reinterpret_cast<uintptr_t>(ptr); }, "Allocate GPU memory (bytes), returns GPU address");

      // cuda_malloc_f16: allocate n_elements * sizeof(float) on GPU (now float)
      m.def("cuda_malloc_f16", [](int n_elements) -> uintptr_t
            {
        float *ptr;
        cudaMalloc(&ptr, (size_t)n_elements * sizeof(float));
        return reinterpret_cast<uintptr_t>(ptr); }, "Allocate n float elements on GPU, returns GPU address");

      // cuda_malloc_f32: allocate n_elements * sizeof(float) on GPU
      m.def("cuda_malloc_f32", [](int n_elements) -> uintptr_t
            {
        float *ptr;
        cudaMalloc(&ptr, (size_t)n_elements * sizeof(float));
        return reinterpret_cast<uintptr_t>(ptr); }, "Allocate n float elements on GPU, returns GPU address");

      // cuda_malloc_i32: allocate n_elements * sizeof(int) on GPU
      m.def("cuda_malloc_i32", [](int n_elements) -> uintptr_t
            {
        int *ptr;
        cudaMalloc(&ptr, (size_t)n_elements * sizeof(int));
        return reinterpret_cast<uintptr_t>(ptr); }, "Allocate n int32 elements on GPU, returns GPU address");

      // cuda_malloc_i16: allocate n_elements * sizeof(short) on GPU
      m.def("cuda_malloc_i16", [](int n_elements) -> uintptr_t
            {
        short *ptr;
        cudaMalloc(&ptr, (size_t)n_elements * sizeof(short));
        return reinterpret_cast<uintptr_t>(ptr); }, "Allocate n int16 elements on GPU, returns GPU address");

      // cuda_free: free GPU memory
      m.def("cuda_free", [](uintptr_t ptr)
            { cudaFree(reinterpret_cast<void *>(ptr)); }, "Free GPU memory");

      m.def("cuda_memcopy_devicetodevice", [](uintptr_t ptr1, uintptr_t ptr2, int num_elem)
            { cudaMemcpy(reinterpret_cast<void *>(ptr1), reinterpret_cast<void *>(ptr2), num_elem * sizeof(float), cudaMemcpyDeviceToDevice); }, "for concatenate");

      // cuda_memset: zero-fill GPU memory
      m.def("cuda_memset", [](uintptr_t ptr, int value, size_t n_bytes)
            { cudaMemset(reinterpret_cast<void *>(ptr), value, n_bytes); }, "Memset GPU memory");

      // ══════════════════════════════════════════════════════════════════
      // Data Transfer: CPU ↔ GPU  (float32 numpy ↔ float GPU)
      // ══════════════════════════════════════════════════════════════════

      // to_device_f16: float32 numpy → float* on GPU (now uses float instead of half)
      m.def("to_device_f16", [](arr_f A) -> uintptr_t
            {
        int size = A.size();
        float *d_ptr;
        cudaMalloc(&d_ptr, (size_t)size * sizeof(float));
        cudaMemcpy(d_ptr, A.data(), (size_t)size * sizeof(float), cudaMemcpyHostToDevice);
        return reinterpret_cast<uintptr_t>(d_ptr); }, "Upload float32 numpy → float GPU, returns GPU address");

      // to_host_f16: float* GPU → float32 numpy (now uses float instead of half)
      m.def("to_host_f16", [](uintptr_t ptr, py::tuple shape) -> arr_f
            {
        int size = 1;
        std::vector<ssize_t> dims;
        for (auto s : shape) {
            int d = s.cast<int>();
            dims.push_back((ssize_t)d);
            size *= d;
        }

        arr_f out(dims);
        cudaMemcpy(out.mutable_data(), reinterpret_cast<float *>(ptr),
                   (size_t)size * sizeof(float), cudaMemcpyDeviceToHost);
        return out; }, "Download float GPU → float32 numpy with shape");

      // to_device_f32: float32 numpy → float* on GPU
      m.def("to_device_f32", [](arr_f A) -> uintptr_t
            {
        int size = A.size();
        float *d_ptr;
        cudaMalloc(&d_ptr, (size_t)size * sizeof(float));
        cudaMemcpy(d_ptr, A.data(), (size_t)size * sizeof(float), cudaMemcpyHostToDevice);
        return reinterpret_cast<uintptr_t>(d_ptr); }, "Upload float32 numpy → float32 GPU, returns GPU address");

      // to_host_f32: float* GPU → float32 numpy
      m.def("to_host_f32", [](uintptr_t ptr, py::tuple shape) -> arr_f
            {
        int size = 1;
        std::vector<ssize_t> dims;
        for (auto s : shape) {
            int d = s.cast<int>();
            dims.push_back((ssize_t)d);
            size *= d;
        }

        arr_f out(dims);
        cudaMemcpy(out.mutable_data(), reinterpret_cast<float *>(ptr),
                   (size_t)size * sizeof(float), cudaMemcpyDeviceToHost);
        return out; }, "Download float32 GPU → float32 numpy with shape");

      // to_host_i32: int* GPU → int32 numpy
      m.def("to_host_i32", [](uintptr_t ptr, py::tuple shape) -> arr_i
            {
        int size = 1;
        std::vector<ssize_t> dims;
        for (auto s : shape) {
            int d = s.cast<int>();
            dims.push_back((ssize_t)d);
            size *= d;
        }

        arr_i out(dims);
        cudaMemcpy(out.mutable_data(), reinterpret_cast<int *>(ptr),
                   (size_t)size * sizeof(int), cudaMemcpyDeviceToHost);
        return out; }, "Download int32 GPU → int32 numpy with shape");

      // to_host_i16: short* GPU → int16 numpy
      m.def("to_host_i16", [](uintptr_t ptr, py::tuple shape) -> arr_s
            {
        int size = 1;
        std::vector<ssize_t> dims;
        for (auto s : shape) {
            int d = s.cast<int>();
            dims.push_back((ssize_t)d);
            size *= d;
        }

        arr_s out(dims);
        cudaMemcpy(out.mutable_data(), reinterpret_cast<short *>(ptr),
                   (size_t)size * sizeof(short), cudaMemcpyDeviceToHost);
        return out; }, "Download int16 GPU → int16 numpy with shape");

      // to_device_i16: int16 numpy → short* on GPU
      m.def("to_device_i16", [](arr_s A) -> uintptr_t
            {
        int size = A.size();
        short *d_ptr;
        cudaMalloc(&d_ptr, (size_t)size * sizeof(short));
        cudaMemcpy(d_ptr, A.data(), (size_t)size * sizeof(short), cudaMemcpyHostToDevice);
        return reinterpret_cast<uintptr_t>(d_ptr); }, "Upload int16 numpy → int16 GPU, returns GPU address");

      // to_device_i32: int32 numpy → int* on GPU
      m.def("to_device_i32", [](arr_i A) -> uintptr_t
            {
        int size = A.size();
        int *d_ptr;
        cudaMalloc(&d_ptr, (size_t)size * sizeof(int));
        cudaMemcpy(d_ptr, A.data(), (size_t)size * sizeof(int), cudaMemcpyHostToDevice);
        return reinterpret_cast<uintptr_t>(d_ptr); }, "Upload int32 numpy → int32 GPU, returns GPU address");
      // ══════════════════════════════════════════════════════════════════
      // Activations  (all GPU ptrs as uintptr_t)
      // ══════════════════════════════════════════════════════════════════

      // cuda_relu_fwd(x_gpu, out_gpu, grad_gpu, size)
      m.def("cuda_relu_fwd", [](uintptr_t x, uintptr_t out, uintptr_t grad, int size)
            { seera_cuda::cuda_relu_fwd(
                  reinterpret_cast<const float *>(x),
                  reinterpret_cast<float *>(out),
                  reinterpret_cast<float *>(grad), size); }, "ReLU forward: x→out, grad (GPU ptrs)");

      // cuda_sigmoid_fwd
      m.def("cuda_sigmoid_fwd", [](uintptr_t x, uintptr_t out, uintptr_t grad, int size)
            { seera_cuda::cuda_sigmoid_fwd(
                  reinterpret_cast<const float *>(x),
                  reinterpret_cast<float *>(out),
                  reinterpret_cast<float *>(grad), size); }, "Sigmoid forward (GPU ptrs)");

      // cuda_tanh_fwd
      m.def("cuda_tanh_fwd", [](uintptr_t x, uintptr_t out, uintptr_t grad, int size)
            { seera_cuda::cuda_tanh_fwd(
                  reinterpret_cast<const float *>(x),
                  reinterpret_cast<float *>(out),
                  reinterpret_cast<float *>(grad), size); }, "Tanh forward (GPU ptrs)");

      // cuda_log_fwd
      m.def("cuda_log_fwd", [](uintptr_t x, uintptr_t out, uintptr_t grad, int size)
            { seera_cuda::cuda_log_fwd(
                  reinterpret_cast<const float *>(x),
                  reinterpret_cast<float *>(out),
                  reinterpret_cast<float *>(grad), size); }, "Log forward (GPU ptrs)");

      // cuda_exp_fwd
      m.def("cuda_exp_fwd", [](uintptr_t x, uintptr_t out, uintptr_t grad, int size)
            { seera_cuda::cuda_exp_fwd(
                  reinterpret_cast<const float *>(x),
                  reinterpret_cast<float *>(out),
                  reinterpret_cast<float *>(grad), size); }, "Exp forward (GPU ptrs)");

      // cuda_abs_fwd
      m.def("cuda_abs_fwd", [](uintptr_t x, uintptr_t out, uintptr_t grad, int size)
            { seera_cuda::cuda_abs_fwd(
                  reinterpret_cast<const float *>(x),
                  reinterpret_cast<float *>(out),
                  reinterpret_cast<float *>(grad), size); }, "Abs forward (GPU ptrs)");

      // cuda_sqrt_fwd
      m.def("cuda_sqrt_fwd", [](uintptr_t x, uintptr_t out, uintptr_t grad, int size)
            { seera_cuda::cuda_sqrt_fwd(
                  reinterpret_cast<const float *>(x),
                  reinterpret_cast<float *>(out),
                  reinterpret_cast<float *>(grad), size); }, "Sqrt forward (GPU ptrs)");

      // cuda_pow_fwd
      m.def("cuda_pow_fwd", [](uintptr_t x, float exponent, uintptr_t out, uintptr_t grad, int size)
            { seera_cuda::cuda_pow_fwd(
                  reinterpret_cast<const float *>(x), exponent,
                  reinterpret_cast<float *>(out),
                  reinterpret_cast<float *>(grad), size); }, "Pow forward (GPU ptrs)");

      // cuda_clip_fwd
      m.def("cuda_clip_fwd", [](uintptr_t x, float lo, float hi, uintptr_t out, uintptr_t grad, int size)
            { seera_cuda::cuda_clip_fwd(
                  reinterpret_cast<const float *>(x), lo, hi,
                  reinterpret_cast<float *>(out),
                  reinterpret_cast<float *>(grad), size); }, "Clip forward (GPU ptrs)");

      // ══════════════════════════════════════════════════════════════════
      // Softmax
      // ══════════════════════════════════════════════════════════════════

      // cuda_softmax_fwd
      m.def("cuda_softmax_fwd", [](uintptr_t x, uintptr_t out, int N, int C)
            { seera_cuda::cuda_softmax_fwd(
                  reinterpret_cast<const float *>(x),
                  reinterpret_cast<float *>(out), N, C); }, "Softmax forward: x[N,C]→out[N,C] (GPU ptrs)");

      // cuda_softmax_vjp
      m.def("cuda_softmax_vjp", [](uintptr_t s, uintptr_t dout, uintptr_t dx, int N, int C)
            { seera_cuda::cuda_softmax_vjp(
                  reinterpret_cast<const float *>(s),
                  reinterpret_cast<const float *>(dout),
                  reinterpret_cast<float *>(dx), N, C); }, "Softmax VJP backward (GPU ptrs)");

      // ══════════════════════════════════════════════════════════════════
      // Matmul
      // ══════════════════════════════════════════════════════════════════

      // cuda_matmul: C[Nbatch x M x N] = A[M x K] @ B[Nbatch x K x N]
      m.def("cuda_matmul", [](uintptr_t A, uintptr_t B, uintptr_t C, int M, int N, int K, int Nbatch)
            { seera_cuda::cuda_matmul(
                  reinterpret_cast<float *>(A),
                  reinterpret_cast<float *>(B),
                  reinterpret_cast<float *>(C), M, N, K, Nbatch); }, "Matmul: A[M,K] @ B[Nbatch,K,N] → C[Nbatch,M,N] (GPU ptrs)");

      // cuda_matmul_bwd
      m.def("cuda_matmul_bwd", [](uintptr_t A, uintptr_t B, uintptr_t dC, uintptr_t dA, uintptr_t dB, int M, int N, int K, int Nbatch)
            { seera_cuda::cuda_matmul_bwd(
                  reinterpret_cast<float *>(A),
                  reinterpret_cast<float *>(B),
                  reinterpret_cast<float *>(dC),
                  reinterpret_cast<float *>(dA),
                  reinterpret_cast<float *>(dB), M, N, K, Nbatch); }, "Matmul backward (GPU ptrs)");

      // cuda_transpose_2d: in[rows x cols] → out[cols x rows]
      m.def("cuda_transpose_2d", [](uintptr_t in, uintptr_t out, int rows, int cols)
            { seera_cuda::cuda_transpose_2d(
                  reinterpret_cast<float *>(in),
                  reinterpret_cast<float *>(out),
                  rows, cols); }, "Transpose 2D: in[rows,cols] → out[cols,rows] (GPU ptrs)");

      // cuda_transpose_3d: in[Nbatch, M, K] → out[Nbatch, K, M]
      m.def("cuda_transpose_3d", [](uintptr_t in, uintptr_t out, int Nbatch, int M, int K)
            { seera_cuda::cuda_transpose_3d(
                  reinterpret_cast<float *>(in),
                  reinterpret_cast<float *>(out),
                  Nbatch, M, K); }, "Transpose 3D batched: in[Nbatch,M,K] → out[Nbatch,K,M] (GPU ptrs)");

      // ══════════════════════════════════════════════════════════════════
      // Conv2D
      // ══════════════════════════════════════════════════════════════════

      // cuda_conv2d_fwd
      m.def("cuda_conv2d_fwd", [](uintptr_t image, uintptr_t kernel, uintptr_t conv, int batchN, int C, int H, int W, int N, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w)
            { seera_cuda::cuda_conv2d_fwd(
                  reinterpret_cast<float *>(image),
                  reinterpret_cast<float *>(kernel),
                  reinterpret_cast<float *>(conv),
                  batchN, C, H, W, N, R, S, pad_h, pad_w, stride_h, stride_w); }, "Conv2D forward (GPU ptrs)");

      // cuda_conv2d_bwd
      m.def("cuda_conv2d_bwd", [](uintptr_t W, uintptr_t X, uintptr_t dY, uintptr_t dX, uintptr_t dW, int batch, int C, int H, int W_in, int N, int R, int S, int strideh, int stridew, int padh, int padw)
            { seera_cuda::cuda_conv2d_bwd(
                  reinterpret_cast<float *>(W),
                  reinterpret_cast<float *>(X),
                  reinterpret_cast<float *>(dY),
                  reinterpret_cast<float *>(dX),
                  reinterpret_cast<float *>(dW),
                  batch, C, H, W_in, N, R, S, strideh, stridew, padh, padw); }, "Conv2D backward (GPU ptrs)");

      // ══════════════════════════════════════════════════════════════════
      // ConvTranspose2D
      // ══════════════════════════════════════════════════════════════════

      // cuda_conv2DTranpose_fwd
      m.def("cuda_conv2DTranpose_fwd", [](uintptr_t X, uintptr_t W, uintptr_t out, int batch, int Cin, int Hin, int Win, int Cout, int KH, int KW, int strideh, int stridew, int padh, int padw)
            { seera_cuda::cuda_conv2DTranpose_fwd(
                  reinterpret_cast<float *>(X),
                  reinterpret_cast<float *>(W),
                  reinterpret_cast<float *>(out),
                  batch, Cin, Hin, Win, Cout, KH, KW, strideh, stridew, padh, padw); }, "ConvTranspose2D forward (GPU ptrs)");

      // cuda_conv2DTranspose_bwd
      m.def("cuda_conv2DTranspose_bwd", [](uintptr_t W, uintptr_t X, uintptr_t dY, uintptr_t dX, uintptr_t dW, int batch, int Cin, int Hin, int Win, int Cout, int KH, int KW, int strideh, int stridew, int padh, int padw)
            { seera_cuda::cuda_conv2DTranspose_bwd(
                  reinterpret_cast<float *>(W),
                  reinterpret_cast<float *>(X),
                  reinterpret_cast<float *>(dY),
                  reinterpret_cast<float *>(dX),
                  reinterpret_cast<float *>(dW),
                  batch, Cin, Hin, Win, Cout, KH, KW, strideh, stridew, padh, padw); }, "ConvTranspose2D backward (GPU ptrs)");

      // ══════════════════════════════════════════════════════════════════
      // MaxPool2D
      // ══════════════════════════════════════════════════════════════════

      // cuda_maxpool_fwd  (mask is short* GPU)
      m.def("cuda_maxpool_fwd", [](uintptr_t image, uintptr_t out, uintptr_t mask, int batchN, int C, int H, int W, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w)
            { seera_cuda::cuda_maxpool_fwd(
                  reinterpret_cast<float *>(image),
                  reinterpret_cast<float *>(out),
                  reinterpret_cast<int *>(mask),
                  batchN, C, H, W, R, S, pad_h, pad_w, stride_h, stride_w); }, "MaxPool2D forward (GPU ptrs, mask is int16 GPU)");

      // cuda_maxpool_bwd
      m.def("cuda_maxpool_bwd", [](uintptr_t dout, uintptr_t mask, uintptr_t dX, int batchN, int C, int H, int W, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w)
            { seera_cuda::cuda_maxpool_bwd(
                  reinterpret_cast<float *>(dout),
                  reinterpret_cast<int *>(mask),
                  reinterpret_cast<float *>(dX),
                  batchN, C, H, W, R, S, pad_h, pad_w, stride_h, stride_w); }, "MaxPool2D backward (GPU ptrs)");

      // ══════════════════════════════════════════════════════════════════
      // Unpooling (Nearest-Neighbor Upsample)
      // ══════════════════════════════════════════════════════════════════

      // cuda_unpooling_fwd
      m.def("cuda_unpooling_fwd", [](uintptr_t inp, uintptr_t out, int batchN, int C, int H, int W, int sh, int sw)
            { seera_cuda::cuda_unpooling_fwd(
                  reinterpret_cast<float *>(inp),
                  reinterpret_cast<float *>(out),
                  batchN, C, H, W, sh, sw); }, "Unpooling (nearest upsample) forward (GPU ptrs)");

      // cuda_unpooling_bwd
      m.def("cuda_unpooling_bwd", [](uintptr_t dout, uintptr_t dx, int batchN, int C, int H, int W, int sh, int sw)
            { seera_cuda::cuda_unpooling_bwd(
                  reinterpret_cast<float *>(dout),
                  reinterpret_cast<float *>(dx),
                  batchN, C, H, W, sh, sw); }, "Unpooling backward (GPU ptrs)");

      // ══════════════════════════════════════════════════════════════════
      // Reduction Ops  (dimarr is HOST int*, passed as numpy)
      // ══════════════════════════════════════════════════════════════════

      // cuda_sum_fwd
      m.def("cuda_sum_fwd", [](uintptr_t A, uintptr_t out, int ndims, int dim, arr_i dimarr)
            { seera_cuda::cuda_sum_fwd(
                  reinterpret_cast<float *>(A),
                  reinterpret_cast<float *>(out),
                  ndims, dim,
                  const_cast<int *>(dimarr.data())); }, "Sum reduction forward (A, out: GPU ptrs; dimarr: numpy)");

      // cuda_mean_fwd
      m.def("cuda_mean_fwd", [](uintptr_t A, uintptr_t out, int ndims, int dim, arr_i dimarr)
            { seera_cuda::cuda_mean_fwd(
                  reinterpret_cast<float *>(A),
                  reinterpret_cast<float *>(out),
                  ndims, dim,
                  const_cast<int *>(dimarr.data())); }, "Mean reduction forward (A, out: GPU ptrs; dimarr: numpy)");

      // cuda_max_fwd
      m.def("cuda_max_fwd", [](uintptr_t A, uintptr_t out, int ndims, int dim, arr_i dimarr)
            { seera_cuda::cuda_max_fwd(
                  reinterpret_cast<float *>(A),
                  reinterpret_cast<float *>(out),
                  ndims, dim,
                  const_cast<int *>(dimarr.data())); }, "Max reduction forward (A, out: GPU ptrs; dimarr: numpy)");

      // cuda_min_fwd
      m.def("cuda_min_fwd", [](uintptr_t A, uintptr_t out, int ndims, int dim, arr_i dimarr)
            { seera_cuda::cuda_min_fwd(
                  reinterpret_cast<float *>(A),
                  reinterpret_cast<float *>(out),
                  ndims, dim,
                  const_cast<int *>(dimarr.data())); }, "Min reduction forward (A, out: GPU ptrs; dimarr: numpy)");

      // cuda_argmax_fwd  (out is int* GPU)
      m.def("cuda_argmax_fwd", [](uintptr_t A, uintptr_t out, int ndims, int dim, arr_i dimarr)
            { seera_cuda::cuda_argmax_fwd(
                  reinterpret_cast<float *>(A),
                  reinterpret_cast<int *>(out),
                  ndims, dim,
                  const_cast<int *>(dimarr.data())); }, "Argmax reduction forward (A: float GPU, out: int GPU; dimarr: numpy)");

      // cuda_argmin_fwd  (out is int* GPU)
      m.def("cuda_argmin_fwd", [](uintptr_t A, uintptr_t out, int ndims, int dim, arr_i dimarr)
            { seera_cuda::cuda_argmin_fwd(
                  reinterpret_cast<float *>(A),
                  reinterpret_cast<int *>(out),
                  ndims, dim,
                  const_cast<int *>(dimarr.data())); }, "Argmin reduction forward (A: float GPU, out: int GPU; dimarr: numpy)");

      // ── Reduction Backward ──────────────────────────────────────────

      // cuda_sum_bwd
      m.def("cuda_sum_bwd", [](uintptr_t dOut, uintptr_t dA, int ndims, int dim, arr_i dimarr)
            { seera_cuda::cuda_sum_bwd(
                  reinterpret_cast<float *>(dOut),
                  reinterpret_cast<float *>(dA),
                  ndims, dim,
                  const_cast<int *>(dimarr.data())); }, "Sum reduction backward (GPU ptrs; dimarr: numpy)");

      // cuda_mean_bwd
      m.def("cuda_mean_bwd", [](uintptr_t dOut, uintptr_t dA, int ndims, int dim, arr_i dimarr)
            { seera_cuda::cuda_mean_bwd(
                  reinterpret_cast<float *>(dOut),
                  reinterpret_cast<float *>(dA),
                  ndims, dim,
                  const_cast<int *>(dimarr.data())); }, "Mean reduction backward (GPU ptrs; dimarr: numpy)");

      // cuda_max_bwd  (needs fwdInput, fwdOutput for sparse grad routing)
      m.def("cuda_max_bwd", [](uintptr_t dOut, uintptr_t fwdInput, uintptr_t fwdOutput, uintptr_t dA, int ndims, int dim, arr_i dimarr)
            { seera_cuda::cuda_max_bwd(
                  reinterpret_cast<float *>(dOut),
                  reinterpret_cast<float *>(fwdInput),
                  reinterpret_cast<float *>(fwdOutput),
                  reinterpret_cast<float *>(dA),
                  ndims, dim,
                  const_cast<int *>(dimarr.data())); }, "Max reduction backward (GPU ptrs; dimarr: numpy)");

      // cuda_min_bwd
      m.def("cuda_min_bwd", [](uintptr_t dOut, uintptr_t fwdInput, uintptr_t fwdOutput, uintptr_t dA, int ndims, int dim, arr_i dimarr)
            { seera_cuda::cuda_min_bwd(
                  reinterpret_cast<float *>(dOut),
                  reinterpret_cast<float *>(fwdInput),
                  reinterpret_cast<float *>(fwdOutput),
                  reinterpret_cast<float *>(dA),
                  ndims, dim,
                  const_cast<int *>(dimarr.data())); }, "Min reduction backward (GPU ptrs; dimarr: numpy)");

      m.def("cuda_scaler_multiply_f", [](uintptr_t arr, float k, int total_elements)
            { seera_cuda::cuda_scaler_multiply_f(
                  reinterpret_cast<float *>(arr), k, total_elements); }, "Scalar multiply float (GPU ptr)");
      m.def("cuda_scaler_add_f", [](uintptr_t arr, float k, int total_elements)
            { seera_cuda::cuda_scaler_add_f(
                  reinterpret_cast<float *>(arr), k, total_elements); }, "Scalar multiply float (GPU ptr)");
      m.def("cuda_ones_f", [](int total_elements)
            {     float* ones; 
                  cudaMalloc(&ones, total_elements*sizeof(float));
                  seera_cuda::cuda_ones_f(ones, total_elements); 
                  return reinterpret_cast<uintptr_t>(ones); }, "Fill ones float (GPU ptr)");

      m.def("cuda_zeros_f", [](int total_elements)
            {     float* zeros; 
                  cudaMalloc(&zeros, total_elements*sizeof(float));
                  seera_cuda::cuda_zeros_f(zeros, total_elements); 
                  return reinterpret_cast<uintptr_t>(zeros); }, "Fill zeros float (GPU ptr)");

      m.def("cuda_power_of", [](uintptr_t arr, float k, int total_elements)
            { seera_cuda::cuda_scaler_power_f(
                  reinterpret_cast<float *>(arr), k, total_elements); }, "Fill zeros float (GPU ptr)");

      // ══════════════════════════════════════════════════════════════════
      // Elemops
      // ══════════════════════════════════════════════════════════════════
      m.def("cuda_elemadd", [](uintptr_t A, uintptr_t B, uintptr_t C, int size)
            { seera_cuda::cuda_elemadd_gputogpu(
                  reinterpret_cast<float *>(A),
                  reinterpret_cast<float *>(B),
                  reinterpret_cast<float *>(C),
                  size); }, "Elementwise add (float)");

      m.def("cuda_elemsub", [](uintptr_t A, uintptr_t B, uintptr_t C, int size)
            { seera_cuda::cuda_elemsub_gputogpu(
                  reinterpret_cast<float *>(A),
                  reinterpret_cast<float *>(B),
                  reinterpret_cast<float *>(C),
                  size); }, "Elementwise sub (float)");

      m.def("cuda_elemdiv", [](uintptr_t A, uintptr_t B, uintptr_t C, int size)
            { seera_cuda::cuda_elemdiv_gputogpu(
                  reinterpret_cast<float *>(A),
                  reinterpret_cast<float *>(B),
                  reinterpret_cast<float *>(C),
                  size); }, "Elementwise div (float)");

      m.def("cuda_elemmult", [](uintptr_t A, uintptr_t B, uintptr_t C, int size)
            { seera_cuda::cuda_elemmult_gputogpu(
                  reinterpret_cast<float *>(A),
                  reinterpret_cast<float *>(B),
                  reinterpret_cast<float *>(C),
                  size); }, "Elementwise mul (float)");

      // ══════════════════════════════════════════════════════════════════
      // Broadcasting Ops (4-D)
      // ══════════════════════════════════════════════════════════════════

      m.def("broadcast_add_4d", [](uintptr_t A, uintptr_t B, uintptr_t C, int aN, int aC, int aH, int aW, int bN, int bC, int bH, int bW)
            { seera_cuda::broadcast_add_4d(
                  reinterpret_cast<const float *>(A),
                  reinterpret_cast<const float *>(B),
                  reinterpret_cast<float *>(C),
                  aN, aC, aH, aW,
                  bN, bC, bH, bW); }, "Broadcast add 4D (GPU ptrs)");

      m.def("broadcast_mul_4d", [](uintptr_t A, uintptr_t B, uintptr_t C, int aN, int aC, int aH, int aW, int bN, int bC, int bH, int bW)
            { seera_cuda::broadcast_mul_4d(
                  reinterpret_cast<const float *>(A),
                  reinterpret_cast<const float *>(B),
                  reinterpret_cast<float *>(C),
                  aN, aC, aH, aW,
                  bN, bC, bH, bW); }, "Broadcast mul 4D (GPU ptrs)");
}