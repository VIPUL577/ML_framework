#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include "seera_engine_cuda.hpp"


#include <stdlib.h>
namespace seera_cuda {
__global__ void float2halff(float *A, half *B) {
  int globalid = blockIdx.x * blockDim.x + threadIdx.x;
  B[globalid] = __float2half(A[globalid]);
}
__global__ void half2float(half *A, float *B) {
  int globalid = blockIdx.x * blockDim.x + threadIdx.x;
  B[globalid] = __half2float(A[globalid]);
}

__global__ void elemadd(half *A, half *B, half *C, int n, int m) {
  __shared__ half tiledA[32 * 32];
  __shared__ half tiledB[32 * 32];

  int tile_width = blockDim.x;
  int id = tile_width * threadIdx.x + threadIdx.y;
  int global_id = (blockIdx.z * m + blockIdx.x * blockDim.x + threadIdx.x) * n +
                  (blockIdx.y * blockDim.y + threadIdx.y);

  tiledA[id] = A[global_id];
  tiledB[id] = B[global_id];

  __syncthreads();

  C[global_id] = tiledA[id] + tiledB[id];
}
__global__ void elemsub(half *A, half *B, half *C, int n, int m) {
  __shared__ half tiledA[32 * 32];
  __shared__ half tiledB[32 * 32];

  int tile_width = blockDim.x;
  int id = tile_width * threadIdx.x + threadIdx.y;
  int global_id = (blockIdx.z * m + blockIdx.x * blockDim.x + threadIdx.x) * n +
                  (blockIdx.y * blockDim.y + threadIdx.y);

  tiledA[id] = A[global_id];
  tiledB[id] = B[global_id];

  __syncthreads();

  C[global_id] = tiledA[id] - tiledB[id];
}
__global__ void elemdiv(half *A, half *B, half *C, int n, int m) {
  __shared__ half tiledA[32 * 32];
  __shared__ half tiledB[32 * 32];

  int tile_width = blockDim.x;
  int id = tile_width * threadIdx.x + threadIdx.y;
  int global_id = (blockIdx.z * m + blockIdx.x * blockDim.x + threadIdx.x) * n +
                  (blockIdx.y * blockDim.y + threadIdx.y);

  tiledA[id] = A[global_id];
  tiledB[id] = B[global_id];

  __syncthreads();

  C[global_id] = tiledA[id] / tiledB[id];
}
__global__ void elemmult(half *A, half *B, half *C, int n, int m) {
  __shared__ half tiledA[32 * 32];
  __shared__ half tiledB[32 * 32];

  int tile_width = blockDim.x;
  int id = tile_width * threadIdx.x + threadIdx.y;
  int global_id = (blockIdx.z * m + blockIdx.x * blockDim.x + threadIdx.x) * n +
                  (blockIdx.y * blockDim.y + threadIdx.y);

  tiledA[id] = A[global_id];
  tiledB[id] = B[global_id];

  __syncthreads();

  C[global_id] = tiledA[id] * tiledB[id];
}

// ======================== BACKWARD KERNELS ========================

// elemadd backward: dA = dC, dB = dC  (just copy upstream gradient)
__global__ void elemadd_bwd(half *dC, half *dA, half *dB, int n, int m) {
  int tile_width = blockDim.x;
  int id = tile_width * threadIdx.x + threadIdx.y;
  int global_id = (blockIdx.z * m + blockIdx.x * blockDim.x + threadIdx.x) * n +
                  (blockIdx.y * blockDim.y + threadIdx.y);

  __shared__ half tiledDC[32 * 32];
  tiledDC[id] = dC[global_id];
  __syncthreads();

  dA[global_id] = tiledDC[id];
  dB[global_id] = tiledDC[id];
}

// elemsub backward: dA = dC, dB = -dC
__global__ void elemsub_bwd(half *dC, half *dA, half *dB, int n, int m) {
  int tile_width = blockDim.x;
  int id = tile_width * threadIdx.x + threadIdx.y;
  int global_id = (blockIdx.z * m + blockIdx.x * blockDim.x + threadIdx.x) * n +
                  (blockIdx.y * blockDim.y + threadIdx.y);

  __shared__ half tiledDC[32 * 32];
  tiledDC[id] = dC[global_id];
  __syncthreads();

  dA[global_id] = tiledDC[id];
  half zero = __float2half(0.0f);
  dB[global_id] = zero - tiledDC[id];
}

// elemmult backward: dA = dC * B, dB = dC * A
__global__ void elemmult_bwd(half *dC, half *A, half *B, half *dA, half *dB,
                             int n, int m) {
  int tile_width = blockDim.x;
  int id = tile_width * threadIdx.x + threadIdx.y;
  int global_id = (blockIdx.z * m + blockIdx.x * blockDim.x + threadIdx.x) * n +
                  (blockIdx.y * blockDim.y + threadIdx.y);

  __shared__ half tDC[32 * 32];
  __shared__ half tA[32 * 32];
  __shared__ half tB[32 * 32];

  tDC[id] = dC[global_id];
  tA[id] = A[global_id];
  tB[id] = B[global_id];
  __syncthreads();

  dA[global_id] = tDC[id] * tB[id];
  dB[global_id] = tDC[id] * tA[id];
}

// elemdiv backward: dA = dC / B, dB = -dC * A / (B * B)
__global__ void elemdiv_bwd(half *dC, half *A, half *B, half *dA, half *dB,
                            int n, int m) {
  int tile_width = blockDim.x;
  int id = tile_width * threadIdx.x + threadIdx.y;
  int global_id = (blockIdx.z * m + blockIdx.x * blockDim.x + threadIdx.x) * n +
                  (blockIdx.y * blockDim.y + threadIdx.y);

  __shared__ half tDC[32 * 32];
  __shared__ half tA[32 * 32];
  __shared__ half tB[32 * 32];

  tDC[id] = dC[global_id];
  tA[id] = A[global_id];
  tB[id] = B[global_id];
  __syncthreads();

  dA[global_id] = tDC[id] / tB[id];
  // dB = -dC * A / (B * B)
  half b_sq = tB[id] * tB[id];
  half zero = __float2half(0.0f);
  dB[global_id] = zero - (tDC[id] * tA[id] / b_sq);
}

// ======================== FORWARD WRAPPERS ========================

void _cuda_elemwise_gputogpu(half *A, half *B, half *C, int N, int M,
                             int batchN,
                             void (*kernel)(half *, half *, half *, int, int)) {
  size_t SZ = (size_t)N * M * batchN * sizeof(half);

  // Assume C is already allocated by caller
  // If not, caller must cudaMalloc before passing

  dim3 block(32, 32);
  dim3 grid((N + 31) / 32, (M + 31) / 32, batchN);

  kernel<<<grid, block>>>(A, B, C, N, M);
  cudaDeviceSynchronize();
}

void cuda_elemadd_gputogpu(half *A, half *B, half *C, int N, int M,
                           int batchN = 1) {
  _cuda_elemwise_gputogpu(A, B, C, N, M, batchN, elemadd);
}

void cuda_elemsub_gputogpu(half *A, half *B, half *C, int N, int M,
                           int batchN = 1) {
  _cuda_elemwise_gputogpu(A, B, C, N, M, batchN, elemsub);
}

void cuda_elemmult_gputogpu(half *A, half *B, half *C, int N, int M,
                            int batchN = 1) {
  _cuda_elemwise_gputogpu(A, B, C, N, M, batchN, elemmult);
}

void cuda_elemdiv_gputogpu(half *A, half *B, half *C, int N, int M,
                           int batchN = 1) {
  _cuda_elemwise_gputogpu(A, B, C, N, M, batchN, elemdiv);
}

// ======================== BACKWARD WRAPPERS ========================

// elemadd backward: dA = dC, dB = dC
void cuda_elemadd_bwd_gputogpu(half *dC, half *dA, half *dB, int N, int M,
                               int batchN = 1) {
  dim3 block(32, 32);
  dim3 grid((N + 31) / 32, (M + 31) / 32, batchN);
  elemadd_bwd<<<grid, block>>>(dC, dA, dB, N, M);
  cudaDeviceSynchronize();
}

// elemsub backward: dA = dC, dB = -dC
void cuda_elemsub_bwd_gputogpu(half *dC, half *dA, half *dB, int N, int M,
                               int batchN = 1) {
  dim3 block(32, 32);
  dim3 grid((N + 31) / 32, (M + 31) / 32, batchN);
  elemsub_bwd<<<grid, block>>>(dC, dA, dB, N, M);
  cudaDeviceSynchronize();
}

// elemmult backward: dA = dC * B, dB = dC * A  (requires saved fwd inputs)
void cuda_elemmult_bwd_gputogpu(half *dC, half *A, half *B, half *dA, half *dB,
                                int N, int M, int batchN = 1) {
  dim3 block(32, 32);
  dim3 grid((N + 31) / 32, (M + 31) / 32, batchN);
  elemmult_bwd<<<grid, block>>>(dC, A, B, dA, dB, N, M);
  cudaDeviceSynchronize();
}

// elemdiv backward: dA = dC / B, dB = -dC * A / (B^2)  (requires saved fwd inputs)
void cuda_elemdiv_bwd_gputogpu(half *dC, half *A, half *B, half *dA, half *dB,
                               int N, int M, int batchN = 1) {
  dim3 block(32, 32);
  dim3 grid((N + 31) / 32, (M + 31) / 32, batchN);
  elemdiv_bwd<<<grid, block>>>(dC, A, B, dA, dB, N, M);
  cudaDeviceSynchronize();
}

} // namespace seera_cuda