#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <cuda_fp16.h>

#include <stdlib.h>

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

__global__ void elemadd(half *A, half *B, half *C, int n, int m)
{
    __shared__ half tiledA[32 * 32];
    __shared__ half tiledB[32 * 32];

    int tile_width = blockDim.x;
    int id = tile_width * threadIdx.x + threadIdx.y;
    int global_id = (blockIdx.z * m + blockIdx.x * blockDim.x + threadIdx.x) * n + (blockIdx.y * blockDim.y + threadIdx.y);

    tiledA[id] = A[global_id];
    tiledB[id] = B[global_id];

    __syncthreads();

    C[global_id] = tiledA[id] + tiledB[id];
}
__global__ void elemsub(half *A, half *B, half *C, int n, int m)
{
    __shared__ half tiledA[32 * 32];
    __shared__ half tiledB[32 * 32];

    int tile_width = blockDim.x;
    int id = tile_width * threadIdx.x + threadIdx.y;
    int global_id = (blockIdx.z * m + blockIdx.x * blockDim.x + threadIdx.x) * n + (blockIdx.y * blockDim.y + threadIdx.y);

    tiledA[id] = A[global_id];
    tiledB[id] = B[global_id];

    __syncthreads();

    C[global_id] = tiledA[id] - tiledB[id];
}
__global__ void elemdiv(half *A, half *B, half *C, int n, int m)
{
    __shared__ half tiledA[32 * 32];
    __shared__ half tiledB[32 * 32];

    int tile_width = blockDim.x;
    int id = tile_width * threadIdx.x + threadIdx.y;
    int global_id = (blockIdx.z * m + blockIdx.x * blockDim.x + threadIdx.x) * n + (blockIdx.y * blockDim.y + threadIdx.y);

    tiledA[id] = A[global_id];
    tiledB[id] = B[global_id];

    __syncthreads();

    C[global_id] = tiledA[id] / tiledB[id];
}
__global__ void elemmult(half *A, half *B, half *C, int n, int m)
{
    __shared__ half tiledA[32 * 32];
    __shared__ half tiledB[32 * 32];

    int tile_width = blockDim.x;
    int id = tile_width * threadIdx.x + threadIdx.y;
    int global_id = (blockIdx.z * m + blockIdx.x * blockDim.x + threadIdx.x) * n + (blockIdx.y * blockDim.y + threadIdx.y);

    tiledA[id] = A[global_id];
    tiledB[id] = B[global_id];

    __syncthreads();

    C[global_id] = tiledA[id] * tiledB[id];
}

float *_cuda_elemwise(float *hA, float *hB,
                      int N, int M, int batchN,
                      void (*kernel)(half *, half *, half *, int, int))
{
    size_t SZ = (size_t)N * M * sizeof(half);
    size_t SZf = (size_t)N * M * sizeof(float);

    float *fA, *fB,*fC; 

    half *dA, *dB, *dC;
    cudaMalloc(&dA, SZ);
    cudaMalloc(&dB, SZ);
    cudaMalloc(&dC, SZ);
    cudaMalloc(&fA, SZf);
    cudaMalloc(&fB, SZf);
    cudaMalloc(&fC, SZf);

    cudaMemcpy(fA, hA, SZf, cudaMemcpyHostToDevice);
    cudaMemcpy(fB, hB, SZf, cudaMemcpyHostToDevice);

    float2halff<<<ceil((float)batchN * M * N / 512), 512>>>(fA, dA);
    float2halff<<<ceil((float)batchN * M * N / 512), 512>>>(fB, dB);

    dim3 block(32, 32, batchN);
    dim3 grid(N / 32, M / 32);

    kernel<<<grid, block>>>(dA, dB, dC, N, M);
    
    float *hC = new float[(size_t)N * M];
    half2float<<<ceil((float)batchN * M * N / 512), 512>>>(dC, fC);
    cudaDeviceSynchronize();
    cudaMemcpy(hC, fC, SZf, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(fA);
    cudaFree(fB);
    cudaFree(fC);

    return hC;
}

half *_cuda_elemwise_gputogpu(half *hA, half *hB,
                              int N, int M, int batchN,
                              void (*kernel)(half *, half *, half *, int, int))
{
    size_t SZ = (size_t)N * M * sizeof(float);

    half *dC;

    cudaMalloc(&dC, SZ);

    dim3 block(32, 32, batchN);
    dim3 grid(N / 32, M / 32);

    kernel<<<grid, block>>>(hA, hB, dC, N, M);
    cudaDeviceSynchronize();

    return dC;
}

float *cuda_elemadd(float *hA, float *hB, int N, int M, int batchN = 1)
{
    return _cuda_elemwise(hA, hB, N, M, batchN, elemadd);
}

float *cuda_elemsub(float *hA, float *hB, int N, int M, int batchN = 1)
{
    return _cuda_elemwise(hA, hB, N, M, batchN, elemsub);
}

float *cuda_elemmult(float *hA, float *hB, int N, int M, int batchN = 1)
{
    return _cuda_elemwise(hA, hB, N, M, batchN, elemmult);
}

float *cuda_elemdiv(float *hA, float *hB, int N, int M, int batchN = 1)
{
    return _cuda_elemwise(hA, hB, N, M, batchN, elemdiv);
}

half *cuda_elemadd_gputogpu(half *hA, half *hB, int N, int M, int batchN = 1)
{
    return _cuda_elemwise_gputogpu(hA, hB, N, M, batchN, elemadd);
}

half *cuda_elemsub_gputogpu(half *hA, half *hB, int N, int M, int batchN = 1)
{
    return _cuda_elemwise_gputogpu(hA, hB, N, M, batchN, elemsub);
}

half *cuda_elemmult_gputogpu(half *hA, half *hB, int N, int M, int batchN = 1)
{
    return _cuda_elemwise_gputogpu(hA, hB, N, M, batchN, elemmult);
}

half *cuda_elemdiv_gputogpu(half *hA, half *hB, int N, int M, int batchN = 1)
{
    return _cuda_elemwise_gputogpu(hA, hB, N, M, batchN, elemdiv);
}