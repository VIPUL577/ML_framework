#include <cuda.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <vector>

// git ls-files | xargs wc -l to count the number of lines
namespace seera_cuda
{

  __global__ void Reductionsum(float *arr, float *output, int limit, int stride,
                               float divisor, int totalthreads)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalthreads) return;

    float temp = 0.0f;
    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;

    for (int i = 0; i < limit; i += stride)
    {
      temp = temp + arr[base + i];
    }

    output[tid] = temp / divisor;
  }

  __global__ void Reductionmax(float *arr, float *output, int limit, int stride,
                               float dummy, int totalthreads)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalthreads) return;

    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;
    float temp = arr[base];

    for (int i = 0; i < limit; i += stride)
    {
      temp = fmaxf(temp, arr[base + i]);
    }

    output[tid] = temp;
  }
  __global__ void Reductionmin(float *arr, float *output, int limit, int stride,
                               float dummy, int totalthreads)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalthreads) return;

    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;
    float temp = arr[base];

    for (int i = 0; i < limit; i += stride)
    {
      temp = fminf(temp, arr[base + i]);
    }

    output[tid] = temp;
  }
  __global__ void Reductionargmin(float *arr, int *output, int limit, int stride,
                                  int totalthreads)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalthreads) return;

    int arg = 0;
    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;
    float temp = arr[base];

    for (int i = 0; i < limit; i += stride)
    {
      if (arr[base + i] <= temp)
      {
        temp = arr[base + i];
        arg = i / stride;
      }
    }

    output[tid] = arg;
  }
  __global__ void Reductionargmax(float *arr, int *output, int limit, int stride,
                                  int totalthreads)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalthreads) return;

    int arg = 0;
    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;
    float temp = arr[base];

    for (int i = 0; i < limit; i += stride)
    {
      if (arr[base + i] >= temp)
      {
        temp = arr[base + i];
        arg = i / stride;
      }
    }

    output[tid] = arg;
  }

  template <typename Kernel>
  void _cuda_reduce_gputogpu(float *A, float *out, int ndims, int dim, int *dimarr,
                             float divisor, Kernel kernel)
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

    kernel<<<blocks, threadsPerBlock>>>(A, out, limit, stride, divisor, totalthreads);

    cudaDeviceSynchronize();
  }

  template <typename Kernel>
  void _cuda_reduce_arg_gputogpu(float *A, int *out, int ndims, int dim,
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

    kernel<<<blocks, threads>>>(A, out, limit, stride, totalthreads);

    cudaDeviceSynchronize();
  }

  void cuda_sum_fwd(float *A, float *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_gputogpu(A, out, ndims, dim, dimarr, 1.0f,
                          Reductionsum);
  }

  void cuda_mean_fwd(float *A, float *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_gputogpu(A, out, ndims, dim, dimarr,
                          (float)dimarr[dim], Reductionsum);
  }

  void cuda_max_fwd(float *A, float *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_gputogpu(A, out, ndims, dim, dimarr, 0.0f,
                          Reductionmax);
  }

  void cuda_min_fwd(float *A, float *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_gputogpu(A, out, ndims, dim, dimarr, 0.0f,
                          Reductionmin);
  }

  void cuda_argmax_fwd(float *A, int *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_arg_gputogpu(A, out, ndims, dim, dimarr, Reductionargmax);
  }

  void cuda_argmin_fwd(float *A, int *out, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_arg_gputogpu(A, out, ndims, dim, dimarr, Reductionargmin);
  }

  // ======================== BACKWARD KERNELS ========================

  // sum backward: broadcast upstream gradient back along reduced dimension
  // dA[outer * limit + inner + i*stride] = dOut[tid]  for i in [0, dimarr[dim])
  __global__ void Reductionsum_bwd(float *dOut, float *dA, int limit, int stride,
                                   float divisor, int totalthreads)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalthreads) return;

    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;

    float grad = dOut[tid] / divisor;

    for (int i = 0; i < limit; i += stride)
    {
      dA[base + i] = grad;
    }
  }

  // max/min backward: gradient flows only to the position matching the forward output
  // dA[position_of_max] = dOut[tid], all others = 0
  __global__ void Reductionmax_bwd(float *dOut, float *fwdInput, float *fwdOutput,
                                   float *dA, int limit, int stride, int totalthreads)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalthreads) return;

    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;

    float out_val = fwdOutput[tid];
    float grad = dOut[tid];
    int found = 0;

    for (int i = 0; i < limit; i += stride)
    {
      if (!found && fwdInput[base + i] == out_val)
      {
        dA[base + i] = grad;
        found = 1;
      }
      else
      {
        dA[base + i] = 0.0f;
      }
    }
  }

  __global__ void Reductionmin_bwd(float *dOut, float *fwdInput, float *fwdOutput,
                                   float *dA, int limit, int stride, int totalthreads)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalthreads) return;

    int inner = tid % stride;
    int outer = tid / stride;
    int base = outer * limit + inner;

    float out_val = fwdOutput[tid];
    float grad = dOut[tid];
    int found = 0;

    for (int i = 0; i < limit; i += stride)
    {
      if (!found && fwdInput[base + i] == out_val)
      {
        dA[base + i] = grad;
        found = 1;
      }
      else
      {
        dA[base + i] = 0.0f;
      }
    }
  }

  // ======================== BACKWARD WRAPPERS ========================

  // Template wrapper for sum/mean backward (same kernel signature as forward)
  template <typename Kernel>
  void _cuda_reduce_bwd_gputogpu(float *dOut, float *dA, int ndims, int dim,
                                 int *dimarr, float divisor, Kernel kernel)
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

    kernel<<<blocks, threadsPerBlock>>>(dOut, dA, limit, stride, divisor, totalthreads);

    cudaDeviceSynchronize();
  }

  // Template wrapper for max/min backward (sparse gradient, extra saved-tensor args)
  template <typename Kernel>
  void _cuda_reduce_bwd_sparse_gputogpu(float *dOut, float *fwdInput,
                                        float *fwdOutput, float *dA, int ndims,
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
                                        stride, totalthreads);

    cudaDeviceSynchronize();
  }

  // sum backward: dA[i] = dOut[reduced_idx] (broadcast, no scaling)
  void cuda_sum_bwd(float *dOut, float *dA, int ndims, int dim,
                    int *dimarr)
  {
    _cuda_reduce_bwd_gputogpu(dOut, dA, ndims, dim, dimarr, 1.0f,
                              Reductionsum_bwd);
  }

  // mean backward: dA[i] = dOut[reduced_idx] / dimarr[dim]
  void cuda_mean_bwd(float *dOut, float *dA, int ndims, int dim,
                     int *dimarr)
  {
    _cuda_reduce_bwd_gputogpu(dOut, dA, ndims, dim, dimarr,
                              (float)dimarr[dim],
                              Reductionsum_bwd);
  }

  // max backward: dA[argmax_pos] = dOut, rest = 0 (requires saved fwd input/output)
  void cuda_max_bwd(float *dOut, float *fwdInput, float *fwdOutput,
                    float *dA, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_bwd_sparse_gputogpu(dOut, fwdInput, fwdOutput, dA, ndims, dim,
                                     dimarr, Reductionmax_bwd);
  }

  // min backward: dA[argmin_pos] = dOut, rest = 0 (requires saved fwd input/output)
  void cuda_min_bwd(float *dOut, float *fwdInput, float *fwdOutput,
                    float *dA, int ndims, int dim, int *dimarr)
  {
    _cuda_reduce_bwd_sparse_gputogpu(dOut, fwdInput, fwdOutput, dA, ndims, dim,
                                     dimarr, Reductionmin_bwd);
  }

} // namespace seera_cuda