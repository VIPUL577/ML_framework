#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

__global__ void float2halff(float *A, half *B)
{
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    B[globalid] = __float2half(A[globalid]);
}
__global__ void half2float(half *A, float *B)
{
    int globalid = blockIdx.x * blockDim.x + threadIdx.x;
    B[globalid] = __half2float(A[globalid]);
}

__global__ void Reductionsum(half *arr, half *output, int limit, int stride, half divisor)
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

__global__ void Reductionmax(half *arr, half *output, int limit, int stride, half dummy)
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
__global__ void Reductionmin(half *arr, half *output, int limit, int stride, half dummy)
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

    for (int i = 1; i < limit; i += stride)
    {
        if (temp > arr[base + i])
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

    for (int i = 1; i < limit; i += stride)
    {
        if (temp < arr[base + i])
        {
            temp = arr[base + i];
            arg = i / stride;
        }
    }

    output[tid] = arg;
}

float *_cuda_elemwise(float *hA,
                      int ndims, int dim, int *dimarr, half divisor = __float2half(0.0f),
                      void (*kernel)(half *, half *, int, int, half))
{
    int prod = 1;
    int stride = 1;
    int limit = 1;
    int totalthreads = 1;

    for (int i = ndims - 1; i > -1; i--)
    {
        prod *= dimarr[i];
        if (i == dim)
        {
            limit = prod;
        }
        else
        {
            totalthreads *= dimarr[i];
        }
    }

    stride = limit / dimarr[dim];
    int total_elems = prod;

    float *fA, *fB;
    half *dA, *dB;

    float *hrA = new float[totalthreads];
    cudaMalloc(&dA, total_elems * sizeof(half));
    cudaMalloc(&dB, total_elems * sizeof(half));

    cudaMalloc(&fA, total_elems * sizeof(float));
    cudaMalloc(&fB, total_elems * sizeof(float));

    cudaMemcpy(fA, hA, total_elems * sizeof(float), cudaMemcpyHostToDevice);

    float2halff<<<ceil((float)total_elems / 512), 512>>>(fA, dA);

    int threadsPerBlock = 256;
    int blocks = (totalthreads + threadsPerBlock - 1) / threadsPerBlock;

    kernel<<<blocks, threadsPerBlock>>>(dA, dB, limit, stride, divisor);
    half2float<<<ceil((float)totalthreads / 512), 512>>>(dB, fB);
    cudaMemcpy(hrA, fB, totalthreads * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(fA);
    cudaFree(fB);

    return hrA;
}

half *_cuda_elemwise_gputogpu(half *hA,
                              int ndims, int dim, int *dimarr, half divisor = __float2half(0.0f),
                              void (*kernel)(half *, half *, int, int,  half))
{
    int prod = 1;
    int stride = 1;
    int limit = 1;
    int totalthreads = 1;

    for (int i = ndims - 1; i > -1; i--)
    {
        prod *= dimarr[i];
        if (i == dim)
        {
            limit = prod;
        }
        else
        {
            totalthreads *= dimarr[i];
        }
    }

    stride = limit / dimarr[dim];
    int total_elems = prod;

    // float *fA,*fB;
    half *dA;

    cudaMalloc(&dA, totalthreads * sizeof(half));

    int threadsPerBlock = 256;
    int blocks = (totalthreads + threadsPerBlock - 1) / threadsPerBlock;

    kernel<<<blocks, threadsPerBlock>>>(hA, dA, limit, stride, divisor);

    cudaDeviceSynchronize();

    return dA;
}
int *_cuda_reduce_arg(float *hA,
                      int ndims, int dim, int *dimarr,
                      void (*kernel)(half *, int *, int, int))
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
    int total_elems = prod;

    float *dA_f;
    half *dA;
    int *dB;

    cudaMalloc(&dA_f, total_elems * sizeof(float));
    cudaMalloc(&dA, total_elems * sizeof(half));
    cudaMalloc(&dB, totalthreads * sizeof(int));

    cudaMemcpy(dA_f, hA, total_elems * sizeof(float), cudaMemcpyHostToDevice);

    float2halff<<<(total_elems + 511) / 512, 512>>>(dA_f, dA);

    int threads = 256;
    int blocks = (totalthreads + threads - 1) / threads;

    kernel<<<blocks, threads>>>(dA, dB, limit, stride);

    int *h_out = new int[totalthreads];
    cudaMemcpy(h_out, dB, totalthreads * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(dA_f);
    cudaFree(dA);
    cudaFree(dB);

    return h_out;
}

int *_cuda_reduce_arg_gputogpu(half *dA,
                               int ndims, int dim, int *dimarr,
                               void (*kernel)(half *, int *, int, int))
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

    int *dB;
    cudaMalloc(&dB, totalthreads * sizeof(int));

    int threads = 256;
    int blocks = (totalthreads + threads - 1) / threads;

    kernel<<<blocks, threads>>>(dA, dB, limit, stride);

    cudaDeviceSynchronize();
    return dB;
}

float *cuda_sum(float *hA, int ndims, int dim, int *dimarr)
{
    return _cuda_elemwise(hA, ndims, dim, dimarr, __float2half(0.0f), Reductionsum);
}

float *cuda_mean(float *hA, int ndims, int dim, int *dimarr)
{
    return _cuda_elemwise(hA, ndims, dim, dimarr, __float2half((float)dimarr[dim]), Reductionsum);
}

float *cuda_max(float *hA, int ndims, int dim, int *dimarr)
{
    return _cuda_elemwise(hA, ndims, dim, dimarr, __float2half(0.0f), Reductionmax);
}

float *cuda_min(float *hA, int ndims, int dim, int *dimarr)
{
    return _cuda_elemwise(hA, ndims, dim, dimarr, __float2half(0.0f), Reductionmin);
}

int *cuda_argmax(float *hA, int ndims, int dim, int *dimarr)
{
    return _cuda_reduce_arg(hA, ndims, dim, dimarr, Reductionargmax);
}

int *cuda_argmin(float *hA, int ndims, int dim, int *dimarr)
{
    return _cuda_reduce_arg(hA, ndims, dim, dimarr, Reductionargmin);
}


half* cuda_sum_gputogpu(half* dA, int ndims, int dim, int* dimarr)
{
    return _cuda_elemwise_gputogpu(dA, ndims, dim, dimarr,__float2half(0.0f), Reductionsum);
}

half* cuda_mean_gputogpu(half* dA, int ndims, int dim, int* dimarr)
{
    return _cuda_elemwise_gputogpu(dA, ndims, dim, dimarr,__float2half((float)dimarr[dim]), Reductionsum);
}

half* cuda_max_gputogpu(half* dA, int ndims, int dim, int* dimarr)
{
    return _cuda_elemwise_gputogpu(dA, ndims, dim, dimarr, __float2half(0.0f),Reductionmax);
}

half* cuda_min_gputogpu(half* dA, int ndims, int dim, int* dimarr)
{
    return _cuda_elemwise_gputogpu(dA, ndims, dim, dimarr, __float2half(0.0f), Reductionmin);
}

int* cuda_argmax_gputogpu(half* dA, int ndims, int dim, int* dimarr)
{
    return _cuda_reduce_arg_gputogpu(dA, ndims, dim, dimarr, Reductionargmax);
}

int* cuda_argmin_gputogpu(half* dA, int ndims, int dim, int* dimarr)
{
    return _cuda_reduce_arg_gputogpu(dA, ndims, dim, dimarr, Reductionargmin);
}