#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>
#include <math.h>
#include "seera_engine_cuda.hpp"
#include <stdio.h>
#include <time.h>
#include <vector>

// git ls-files | xargs wc -l to count the number of lines
namespace seera_cuda
{
  __host__ __device__ inline void float2halff(float *A, half *B)
  {
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    B[globalid] = __float2half(A[globalid]);
  }
  __host__ __device__ inline void half2float(half *A, float *B)
  {
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    B[globalid] = __half2float(A[globalid]);
  }

  __global__ void Reductionsum(half *arr, half *output, int limit, int stride,
                               half divisor)
  {
    half temp = __float2half(0.0f);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;

    for (int i = 0; i < limit; i += stride)
    {
      temp = __hadd(temp, arr[base + i]);
    }

    output[tid] = __hdiv(temp, divisor);
  }

  __global__ void Reductionmax(half *arr, half *output, int limit, int stride,
                               half dummy)
  {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;
    half temp = arr[base];

    for (int i = 0; i < limit; i += stride)
    {
      temp = __hmax(temp, arr[base + i]);
    }

    output[tid] = temp;
  }
  __global__ void Reductionmin(half *arr, half *output, int limit, int stride,
                               half dummy)
  {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;
    half temp = arr[base];

    for (int i = 0; i < limit; i += stride)
    {
      temp = __hmin(temp, arr[base + i]);
    }

    output[tid] = temp;
  }
  __global__ void Reductionargmin(half *arr, int *output, int limit, int stride)
  {
    int arg = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;
    half temp = arr[base];

    for (int i = 0; i < limit; i += stride)
    {
      if (__hgt(arr[base + i], temp) == 0)
      {
        temp = arr[base + i];
        arg = i / stride;
      }
    }

    output[tid] = arg;
  }
  __global__ void Reductionargmax(half *arr, int *output, int limit, int stride)
  {
    int arg = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;
    half temp = arr[base];

    for (int i = 0; i < limit; i += stride)
    {
      if (__hlt(arr[base + i], temp) == 0)
      {
        temp = arr[base + i];
        arg = i / stride;
      }
    }

    output[tid] = arg;
  }

  template <typename Kernel>
  void _cuda_reduce_gputogpu(half *A, half *out, int ndims, int dim, int *dimarr,
                             half divisor, Kernel kernel)
  {
    int prod = 1;
    int stride = 1;
    int limit = 1;
    int totalthreads = 1;

    for (int i = ndims - 1; i >= 0; i--)
    {
      prod *= dimarr[i];
      if (i == dim)
        limit = prod;
      else
        totalthreads *= dimarr[i];
    }

    stride = limit / dimarr[dim];

    // out must already be allocated: size = totalthreads

    int threadsPerBlock = 256;
    int blocks = (totalthreads + threadsPerBlock - 1) / threadsPerBlock;

    kernel<<<blocks, threadsPerBlock>>>(A, out, limit, stride, divisor);

    cudaDeviceSynchronize();
  }

  template <typename Kernel>
  void _cuda_reduce_arg_gputogpu(half *A, int *out, int ndims, int dim,
                                 int *dimarr, Kernel kernel)
  {
    int prod = 1, stride = 1, limit = 1, totalthreads = 1;

    for (int i = ndims - 1; i >= 0; i--)
    {
      prod *= dimarr[i];
      if (i == dim)
        limit = prod;
      else
        totalthreads *= dimarr[i];
    }

    stride = limit / dimarr[dim];

    // out must already be allocated: size = totalthreads

    int threads = 256;
    int blocks = (totalthreads + threads - 1) / threads;

    kernel<<<blocks, threads>>>(A, out, limit, stride);

    cudaDeviceSynchronize();
  }

  void cuda_sum_fwd(half *A, half *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_gputogpu(A, out, ndims, dim, dimarr, __float2half(1.0f),
                          Reductionsum);
  }

  void cuda_mean_fwd(half *A, half *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_gputogpu(A, out, ndims, dim, dimarr,
                          __float2half((float)dimarr[dim]), Reductionsum);
  }

  void cuda_max_fwd(half *A, half *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_gputogpu(A, out, ndims, dim, dimarr, __float2half(0.0f),
                          Reductionmax);
  }

  void cuda_min_fwd(half *A, half *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_gputogpu(A, out, ndims, dim, dimarr, __float2half(0.0f),
                          Reductionmin);
  }

  void cuda_argmax_fwd(half *A, int *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_arg_gputogpu(A, out, ndims, dim, dimarr, Reductionargmax);
  }

  void cuda_argmin_fwd(half *A, int *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_arg_gputogpu(A, out, ndims, dim, dimarr, Reductionargmin);
  }

  // ======================== BACKWARD KERNELS ========================

  // sum backward: broadcast upstream gradient back along reduced dimension
  // dA[outer * limit + inner + i*stride] = dOut[tid]  for i in [0, dimarr[dim])
  __global__ void Reductionsum_bwd(half *dOut, half *dA, int limit, int stride,
                                   half divisor)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;

    half grad = __hdiv(dOut[tid], divisor);

    for (int i = 0; i < limit; i += stride)
    {
      dA[base + i] = grad;
    }
  }

  // max/min backward: gradient flows only to the position matching the forward output
  // dA[position_of_max] = dOut[tid], all others = 0
  __global__ void Reductionmax_bwd(half *dOut, half *fwdInput, half *fwdOutput,
                                   half *dA, int limit, int stride)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;

    half out_val = fwdOutput[tid];
    half grad = dOut[tid];
    int found = 0;

    for (int i = 0; i < limit; i += stride)
    {
      // Route gradient to the first position that matches the max value
      if (!found && __heq(fwdInput[base + i], out_val))
      {
        dA[base + i] = grad;
        found = 1;
      }
      else
      {
        dA[base + i] = __float2half(0.0f);
      }
    }
  }

  __global__ void Reductionmin_bwd(half *dOut, half *fwdInput, half *fwdOutput,
                                   half *dA, int limit, int stride)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;

    half out_val = fwdOutput[tid];
    half grad = dOut[tid];
    int found = 0;

    for (int i = 0; i < limit; i += stride)
    {
      if (!found && __heq(fwdInput[base + i], out_val))
      {
        dA[base + i] = grad;
        found = 1;
      }
      else
      {
        dA[base + i] = __float2half(0.0f);
      }
    }
  }

  // ======================== BACKWARD WRAPPERS ========================

  // Template wrapper for sum/mean backward (same kernel signature as forward)
  template <typename Kernel>
  void _cuda_reduce_bwd_gputogpu(half *dOut, half *dA, int ndims, int dim,
                                 int *dimarr, half divisor, Kernel kernel)
  {
    int prod = 1;
    int stride = 1;
    int limit = 1;
    int totalthreads = 1;

    for (int i = ndims - 1; i >= 0; i--)
    {
      prod *= dimarr[i];
      if (i == dim)
        limit = prod;
      else
        totalthreads *= dimarr[i];
    }

    stride = limit / dimarr[dim];

    int threadsPerBlock = 256;
    int blocks = (totalthreads + threadsPerBlock - 1) / threadsPerBlock;

    kernel<<<blocks, threadsPerBlock>>>(dOut, dA, limit, stride, divisor);

    cudaDeviceSynchronize();
  }

  // Template wrapper for max/min backward (sparse gradient, extra saved-tensor args)
  template <typename Kernel>
  void _cuda_reduce_bwd_sparse_gputogpu(half *dOut, half *fwdInput,
                                        half *fwdOutput, half *dA, int ndims,
                                        int dim, int *dimarr, Kernel kernel)
  {
    int prod = 1;
    int stride = 1;
    int limit = 1;
    int totalthreads = 1;

    for (int i = ndims - 1; i >= 0; i--)
    {
      prod *= dimarr[i];
      if (i == dim)
        limit = prod;
      else
        totalthreads *= dimarr[i];
    }

    stride = limit / dimarr[dim];

    int threadsPerBlock = 256;
    int blocks = (totalthreads + threadsPerBlock - 1) / threadsPerBlock;

    kernel<<<blocks, threadsPerBlock>>>(dOut, fwdInput, fwdOutput, dA, limit,
                                        stride);

    cudaDeviceSynchronize();
  }

  // sum backward: dA[i] = dOut[reduced_idx] (broadcast, no scaling)
  void cuda_sum_bwd(half *dOut, half *dA, int ndims, int dim,
                    int *dimarr)
  {
    _cuda_reduce_bwd_gputogpu(dOut, dA, ndims, dim, dimarr, __float2half(1.0f),
                              Reductionsum_bwd);
  }

  // mean backward: dA[i] = dOut[reduced_idx] / dimarr[dim]
  void cuda_mean_bwd(half *dOut, half *dA, int ndims, int dim,
                     int *dimarr)
  {
    _cuda_reduce_bwd_gputogpu(dOut, dA, ndims, dim, dimarr,
                              __float2half((float)dimarr[dim]),
                              Reductionsum_bwd);
  }

  // max backward: dA[argmax_pos] = dOut, rest = 0 (requires saved fwd input/output)
  void cuda_max_bwd(half *dOut, half *fwdInput, half *fwdOutput,
                    half *dA, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_bwd_sparse_gputogpu(dOut, fwdInput, fwdOutput, dA, ndims, dim,
                                     dimarr, Reductionmax_bwd);
  }

  // min backward: dA[argmin_pos] = dOut, rest = 0 (requires saved fwd input/output)
  void cuda_min_bwd(half *dOut, half *fwdInput, half *fwdOutput,
                    half *dA, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_bwd_sparse_gputogpu(dOut, fwdInput, fwdOutput, dA, ndims, dim,
                                     dimarr, Reductionmin_bwd);
  }

} // namespace seera_cuda