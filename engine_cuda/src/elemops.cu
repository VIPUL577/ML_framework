#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include <stdlib.h>
namespace seera_cuda
{

    // =======================
    // Kernels
    // =======================

    __global__ void elemadd(float *A, float *B, float *C, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
            C[idx] = A[idx] + B[idx];
    }

    __global__ void elemsub(float *A, float *B, float *C, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
            C[idx] = A[idx] - B[idx];
    }

    __global__ void elemdiv(float *A, float *B, float *C, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
            C[idx] = A[idx] / B[idx];
    }

    __global__ void elemmult(float *A, float *B, float *C, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
            C[idx] = A[idx] * B[idx];
    }

    // =======================
    // Wrappers (NO kernel passing)
    // =======================

    void cuda_elemadd_gputogpu(float *A, float *B, float *C, int size)
    {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        elemadd<<<blocks, threads>>>(A, B, C, size);
        cudaDeviceSynchronize();
    }

    void cuda_elemsub_gputogpu(float *A, float *B, float *C, int size)
    {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        elemsub<<<blocks, threads>>>(A, B, C, size);
        cudaDeviceSynchronize();
    }

    void cuda_elemdiv_gputogpu(float *A, float *B, float *C, int size)
    {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        elemdiv<<<blocks, threads>>>(A, B, C, size);
        cudaDeviceSynchronize();
    }

    void cuda_elemmult_gputogpu(float *A, float *B, float *C, int size)
    {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        elemmult<<<blocks, threads>>>(A, B, C, size);
        cudaDeviceSynchronize();
    }

    // ======================== BACKWARD WRAPPERS ========================

    // elemadd backward: dA = dC, dB = dC
    //   void cuda_elemadd_bwd_gputogpu(float *dC, float *dA, float *dB, int N, int M,
    //                                  int batchN)
    //   {
    //     dim3 block(32, 32);
    //     dim3 grid((N + 31) / 32, (M + 31) / 32, batchN);
    //     elemadd_bwd<<<grid, block>>>(dC, dA, dB, N, M);
    //     cudaDeviceSynchronize();
    //   }

    //   // elemsub backward: dA = dC, dB = -dC
    //   void cuda_elemsub_bwd_gputogpu(float *dC, float *dA, float *dB, int N, int M,
    //                                  int batchN)
    //   {
    //     dim3 block(32, 32);
    //     dim3 grid((N + 31) / 32, (M + 31) / 32, batchN);
    //     elemsub_bwd<<<grid, block>>>(dC, dA, dB, N, M);
    //     cudaDeviceSynchronize();
    //   }

    //   // elemmult backward: dA = dC * B, dB = dC * A  (requires saved fwd inputs)
    //   void cuda_elemmult_bwd_gputogpu(float *dC, float *A, float *B, float *dA, float *dB,
    //                                   int N, int M, int batchN)
    //   {
    //     dim3 block(32, 32);
    //     dim3 grid((N + 31) / 32, (M + 31) / 32, batchN);
    //     elemmult_bwd<<<grid, block>>>(dC, A, B, dA, dB, N, M);
    //     cudaDeviceSynchronize();
    //   }

    //   // elemdiv backward: dA = dC / B, dB = -dC * A / (B^2)  (requires saved fwd inputs)
    //   void cuda_elemdiv_bwd_gputogpu(float *dC, float *A, float *B, float *dA, float *dB,
    //                                  int N, int M, int batchN)
    //   {
    //     dim3 block(32, 32);
    //     dim3 grid((N + 31) / 32, (M + 31) / 32, batchN);
    //     elemdiv_bwd<<<grid, block>>>(dC, A, B, dA, dB, N, M);
    //     cudaDeviceSynchronize();
    //   }

} // namespace seera_cuda