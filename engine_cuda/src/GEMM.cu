#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include "seera_engine_cuda.hpp"

#include <time.h>
namespace seera_cuda {
using namespace nvcuda;

__host__ __device__ inline  void float2halff(float *A, half *B) {
  int globalid = blockIdx.x * blockDim.x + threadIdx.x;
  B[globalid] = __float2half(A[globalid]);
}
__host__ __device__ inline void half2float(half *A, float *B) {
  int globalid = blockIdx.x * blockDim.x + threadIdx.x;
  B[globalid] = __half2float(A[globalid]);
}
__global__ void matmul_wmma_bound(half *A, half *B, half *C, int M, int N,
                                  int K) {
  int warpM = blockIdx.y * 16;
  int warpN = blockIdx.x * 16;
  int batchno = blockIdx.z;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  __shared__ half shA[16 * 16];
  __shared__ half shB[16 * 16];

  for (int p = 0; p < K; p += 16) {
    for (int i = 0; i < 8; ++i) {
      int linear_idx = threadIdx.x + i * 32;
      int row = linear_idx / 16;
      int col = linear_idx % 16;

      int global_row_A = warpM + row;
      int global_col_A = p + col;
      if (global_row_A < M && global_col_A < K) {
        shA[linear_idx] = A[(global_row_A)*K + global_col_A];
      } else {
        shA[linear_idx] = __float2half(0.0f);
      }

      int global_row_B = p + row;
      int global_col_B = warpN + col;
      if (global_row_B < K && global_col_B < N) {
        shB[linear_idx] = B[(batchno * K + global_row_B) * N + global_col_B];
      } else {
        shB[linear_idx] = __float2half(0.0f);
      }
    }

    __syncthreads();

    wmma::load_matrix_sync(a_frag, shA, 16);
    wmma::load_matrix_sync(b_frag, shB, 16);

    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    __syncthreads();
  }

  // 3. SAFE STORE
  __shared__ float shC[16 * 16];
  wmma::store_matrix_sync(shC, acc_frag, 16, wmma::mem_row_major);
  __syncthreads();

  // The 32 threads cooperate to write the 256 elements back to global memory,
  // respecting bounds
  for (int i = 0; i < 8; ++i) {
    int linear_idx = threadIdx.x + i * 32;
    int row = linear_idx / 16;
    int col = linear_idx % 16;

    int global_row_C = warpM + row;
    int global_col_C = warpN + col;

    if (global_row_C < M && global_col_C < N) {
      C[(batchno * M + global_row_C) * N + global_col_C] = shC[linear_idx];
    }
  }
}

// ======================== FORWARD WRAPPER ========================

void cuda_matmul(half *hA, half *hB, half *hC, int M, int N, int K,
                 int Nbatch) {

  dim3 block(32);
  dim3 grid((N + 15) / 16, (M + 15) / 16, Nbatch);

  matmul_wmma_bound<<<grid, block>>>(hA, hB, hC, M, N, K);
  cudaDeviceSynchronize();
}

// ======================== UTILITY KERNELS ========================

// Transpose a 2D matrix: in[rows x cols] → out[cols x rows]
__global__ void transpose_2d(half *in, half *out, int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows * cols) {
    int r = idx / cols;
    int c = idx % cols;
    out[c * rows + r] = in[r * cols + c];
  }
}

// Elementwise accumulate: dst[i] += src[i]
__global__ void elemwise_accumulate(half *dst, half *src, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = __hadd(dst[idx], src[idx]);
  }
}

// ======================== BACKWARD WRAPPER ========================

// Forward: C[batch x M x N] = A[M x K] @ B[batch x K x N]
// Backward:
//   dB[batch x K x N] = A_T[K x M] @ dC[batch x M x N]
//   dA[M x K]         = sum_b( dC_b[M x N] @ B_T_b[N x K] )
//
// Both computed by reusing matmul_wmma_bound with transposed inputs.

void cuda_matmul_bwd(half *A, half *B, half *dC, half *dA, half *dB,
                     int M, int N, int K, int Nbatch) {

  int threads = 256;

  // --- Allocate temporaries ---
  half *A_T, *B_T, *dA_temp;
  cudaMalloc(&A_T, (size_t)K * M * sizeof(half));
  cudaMalloc(&B_T, (size_t)Nbatch * N * K * sizeof(half));
  cudaMalloc(&dA_temp, (size_t)M * K * sizeof(half));

  // --- 1. Transpose A (M x K) → A_T (K x M) ---
  int total_A = M * K;
  transpose_2d<<<(total_A + threads - 1) / threads, threads>>>(A, A_T, M, K);

  // --- 2. Transpose each batch of B (K x N) → B_T (N x K) ---
  int slice = K * N;
  for (int b = 0; b < Nbatch; b++) {
    transpose_2d<<<(slice + threads - 1) / threads, threads>>>(
        B + b * slice, B_T + b * slice, K, N);
  }
  cudaDeviceSynchronize();

  // --- 3. dB = A_T @ dC  (reuse matmul_wmma_bound) ---
  // A_T[K x M] is shared (non-batched), dC[batch x M x N] is batched
  // Result: dB[batch x K x N]
  // matmul_wmma_bound(A_T, dC, dB, K_rows, N_cols, M_inner, Nbatch)
  {
    dim3 block(32);
    dim3 grid((N + 15) / 16, (K + 15) / 16, Nbatch);
    matmul_wmma_bound<<<grid, block>>>(A_T, dC, dB, K, N, M);
    cudaDeviceSynchronize();
  }

  // --- 4. dA = sum_b ( dC_b @ B_T_b )  (reuse matmul_wmma_bound per-batch) ---
  // For each batch: dC_b[M x N] @ B_T_b[N x K] = dA_b[M x K]
  // matmul_wmma_bound treats first arg as non-batched, second as batched.
  // With Nbatch=1: first arg = dC_b, second arg = B_T_b, result = dA_temp
  cudaMemset(dA, 0, (size_t)M * K * sizeof(half));

  {
    dim3 block(32);
    dim3 grid((K + 15) / 16, (M + 15) / 16, 1);
    int total_dA = M * K;

    for (int b = 0; b < Nbatch; b++) {
      matmul_wmma_bound<<<grid, block>>>(
          dC + b * M * N,    // dC_b as "A" (non-batched) [M x N]
          B_T + b * N * K,   // B_T_b as "B" (Nbatch=1)  [N x K]
          dA_temp,           // result [M x K]
          M, K, N);
      cudaDeviceSynchronize();

      // Accumulate: dA += dA_temp
      elemwise_accumulate<<<(total_dA + threads - 1) / threads, threads>>>(
          dA, dA_temp, total_dA);
      cudaDeviceSynchronize();
    }
  }

  // --- Cleanup ---
  cudaFree(A_T);
  cudaFree(B_T);
  cudaFree(dA_temp);
}

} // namespace seera_cuda
