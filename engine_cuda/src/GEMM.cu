#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <time.h>
#include <mma.h>

using namespace nvcuda;

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
__global__ void matmul_wmma_bound(half *A, half *B, half *C, int M, int N, int K)
{
    int warpM = blockIdx.y * 16;
    int warpN = blockIdx.x * 16;
    int batchno = blockIdx.z;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    __shared__ half shA[16 * 16];
    __shared__ half shB[16 * 16];

    for (int p = 0; p < K; p += 16)
    {
        for (int i = 0; i < 8; ++i)
        {
            int linear_idx = threadIdx.x + i * 32;
            int row = linear_idx / 16;
            int col = linear_idx % 16;

            int global_row_A = warpM + row;
            int global_col_A = p + col;
            if (global_row_A < M && global_col_A < K)
            {
                shA[linear_idx] = A[(batchno * M + global_row_A) * K + global_col_A];
            }
            else
            {
                shA[linear_idx] = __float2half(0.0f);
            }

            int global_row_B = p + row;
            int global_col_B = warpN + col;
            if (global_row_B < K && global_col_B < N)
            {
                shB[linear_idx] = B[global_row_B * N + global_col_B];
            }
            else
            {
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

    // The 32 threads cooperate to write the 256 elements back to global memory, respecting bounds
    for (int i = 0; i < 8; ++i)
    {
        int linear_idx = threadIdx.x + i * 32;
        int row = linear_idx / 16;
        int col = linear_idx % 16;

        int global_row_C = warpM + row;
        int global_col_C = warpN + col;

        if (global_row_C < M && global_col_C < N)
        {
            C[(batchno * M + global_row_C) * N + global_col_C] = shC[linear_idx];
        }
    }
}

float *cuda_matmul(float *A, float *B, int M, int N, int K, int Nbatch)
{
    float *fA, *fB, *fC;
    half *hA, *hB, *hC;

    float C[M * N * Nbatch];

    cudaMalloc(&fA, sizeof(float) * M * K * Nbatch);
    cudaMalloc(&fB, sizeof(float) * N * K * Nbatch);
    cudaMalloc(&hA, sizeof(half) * M * K * Nbatch);
    cudaMalloc(&hB, sizeof(half) * N * K * Nbatch);
    cudaMalloc(&hC, sizeof(half) * N * M * Nbatch);
    cudaMemcpy(fA, A, sizeof(float) * M * K * Nbatch, cudaMemcpyHostToDevice);
    cudaMemcpy(fB, B, sizeof(float) * N * K * Nbatch, cudaMemcpyHostToDevice);

    float2halff<<<ceil((float)Nbatch * M * K / 512), 512>>>(fA, hA);
    float2halff<<<ceil((float)Nbatch * K * N / 512), 512>>>(fB, hB);

    dim3 block(32);
    dim3 grid((N + 15) / 16, (M + 15) / 16, Nbatch);

    matmul_wmma_bound<<<grid, block>>>(hA, hB, hC, M, N, K);
    half2float<<<ceil((float)Nbatch * M * N / 512), 512>>>(hC, fC);
    cudaDeviceSynchronize();

    cudaMemcpy(C, fC, sizeof(float) * N * M * Nbatch, cudaMemcpyDeviceToHost);

    cudaFree(fA);
    cudaFree(fB);
    cudaFree(fC);

    return C;
}

half *cuda_matmul(half *A, half *B, int M, int N, int K, int Nbatch)
{

    half *hA, *hB, *hC;
    half C[M * N * Nbatch];

    cudaMalloc(&hA, sizeof(half) * M * K * Nbatch);
    cudaMalloc(&hB, sizeof(half) * N * K * Nbatch);
    cudaMalloc(&hC, sizeof(half) * N * M * Nbatch);
    cudaMemcpy(hA, A, sizeof(half) * M * K * Nbatch, cudaMemcpyHostToDevice);
    cudaMemcpy(hB, B, sizeof(half) * N * K * Nbatch, cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid((N + 15) / 16, (M + 15) / 16, Nbatch);

    matmul_wmma_bound<<<grid, block>>>(hA, hB, hC, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(C, hC, sizeof(half) * N * M * Nbatch, cudaMemcpyDeviceToHost);

    return C;
}

half *cuda_matmul_gputogpu(half *hA, half *hB, int M, int N, int K, int Nbatch)
{

    half *hC;

    dim3 block(32);
    dim3 grid((N + 15) / 16, (M + 15) / 16, Nbatch);

    matmul_wmma_bound<<<grid, block>>>(hA, hB, hC, M, N, K);
    cudaDeviceSynchronize();

    return hC;
}
