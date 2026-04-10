#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

#include <time.h>
namespace seera_cuda
{
  using namespace nvcuda;

   __global__ void matmul_wmma_bound(float *A, float *B, float *C, int M, int N,
                                    int K)
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
          shA[linear_idx] = __float2half(A[(batchno * M + global_row_A) * K + global_col_A]);
        }
        else
        {
          shA[linear_idx] = __float2half(0.0f);
        }

        int global_row_B = p + row;
        int global_col_B = warpN + col;
        if (global_row_B < K && global_col_B < N)
        {
          shB[linear_idx] = __float2half(B[(global_row_B)*N + global_col_B]);
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

    // The 32 threads cooperate to write the 256 elements back to global memory,
    // respecting bounds
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

  // ======================== FORWARD WRAPPER ========================

  void cuda_matmul(float *hA, float *hB, float *hC, int M, int N, int K,
                   int Nbatch)
  {

    dim3 block(32);
    dim3 grid((N + 15) / 16, (M + 15) / 16, Nbatch);
    matmul_wmma_bound<<<grid, block>>>(hA, hB, hC, M, N, K);
    cudaDeviceSynchronize();
  }

  // ======================== UTILITY KERNELS ========================

  // Transpose a 2D matrix: in[rows x cols] → out[cols x rows]
  __global__ void transpose_2d(float *in, float *out, int rows, int cols)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
    {
      int r = idx / cols;
      int c = idx % cols;
      out[c * rows + r] = in[r * cols + c];
    }
  }

  // Elementwise accumulate: dst[i] += src[i]
  __global__ void elemwise_accumulate(float *dst, float *src, int n)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
      dst[idx] = dst[idx] + src[idx];
    }
  }

  __global__ void transpose_3d_batch(float *in, float *out,
                                     int M, int K, int Nbatch)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // linear within slice
    int b = blockIdx.z;                              // batch index
    int slice = M * K;
    if (b < Nbatch && idx < slice)
    {
      int r = idx / K;
      int c = idx % K;
      out[b * slice + c * M + r] = in[b * slice + r * K + c];
    }
  }

  void cuda_transpose_3d(float *in, float *out, int Nbatch, int M, int K)
  {
    int slice = M * K;
    int threads = 256;
    dim3 grid((slice + threads - 1) / threads, 1, Nbatch);
    transpose_3d_batch<<<grid, threads>>>(in, out, M, K, Nbatch);
    cudaDeviceSynchronize();
  }
 // ======================== BACKWARD WRAPPER ========================

// Forward: C[batch x M x N] = A[batch x M x K] @ B[K x N]
//   (kernel batches the first operand; B is shared across the batch)
//
// Backward:
//   dA[batch x M x K] = dC[batch x M x N] @ B_T[N x K]
//       → single batched kernel call (dC is the batched "A" operand,
//         B_T is the shared non-batched "B" operand)
//
//   dB[K x N]         = sum_b( A_T_b[K x M] @ dC_b[M x N] )
//       → per-batch kernel calls then accumulate into dB

void cuda_matmul_bwd(float *A, float *B, float *dC, float *dA, float *dB,
                     int M, int N, int K, int Nbatch)
{
  int threads = 256;

  // --- Allocate temporaries ---
  float *B_T;                                              // [N x K]        non-batched
  float *A_T;                                              // [batch x K x M]
  float *dB_temp;                                          // [K x N]        accumulation scratch
  cudaMalloc(&B_T,     (size_t)N * K          * sizeof(float));
  cudaMalloc(&A_T,     (size_t)Nbatch * K * M * sizeof(float));
  cudaMalloc(&dB_temp, (size_t)K * N          * sizeof(float));

  // --- 1. Transpose B (K x N) → B_T (N x K)  [non-batched] ---
  int total_B = K * N;
  transpose_2d<<<(total_B + threads - 1) / threads, threads>>>(B, B_T, K, N);

  // --- 2. Transpose every batch slice of A: (Nbatch, M, K) → A_T (Nbatch, K, M) ---
  cuda_transpose_3d(A, A_T, Nbatch, M, K);  // includes cudaDeviceSynchronize()

  cudaDeviceSynchronize();  // ensure B_T is also ready

  // --- 3. dA[batch x M x K] = dC[batch x M x N] @ B_T[N x K] ---
  //    dC   → batched "A" operand  (grid.z = Nbatch)
  //    B_T  → shared non-batched "B" operand
  {
    dim3 block(32);
    dim3 grid((K + 15) / 16, (M + 15) / 16, Nbatch);
    matmul_wmma_bound<<<grid, block>>>(dC, B_T, dA, M, K, N);
    cudaDeviceSynchronize();
  }

  // --- 4. dB[K x N] = sum_b( A_T_b[K x M] @ dC_b[M x N] ) ---
  //    Per-batch: A_T_b is the single-slice "A" (Nbatch=1 → batchno always 0),
  //               dC_b  is the non-batched "B".
  //    Accumulate each result into dB.
  cudaMemset(dB, 0, (size_t)K * N * sizeof(float));
  {
    dim3 block(32);
    dim3 grid((N + 15) / 16, (K + 15) / 16, 1);
    int total_dB = K * N;

    for (int b = 0; b < Nbatch; b++)
    {
      matmul_wmma_bound<<<grid, block>>>(
          A_T + b * K * M,   // A_T_b as "A" (Nbatch=1) [K x M]
          dC  + b * M * N,   // dC_b  as "B" (non-batched) [M x N]
          dB_temp,           // result [K x N]
          K, N, M);
      cudaDeviceSynchronize();

      elemwise_accumulate<<<(total_dB + threads - 1) / threads, threads>>>(
          dB, dB_temp, total_dB);
      cudaDeviceSynchronize();
    }
  }

  // --- Cleanup ---
  cudaFree(B_T);
  cudaFree(A_T);
  cudaFree(dB_temp);
}

  // ======================== TRANSPOSE WRAPPER ========================

  void cuda_transpose_2d(float *in, float *out, int rows, int cols)
  {
    int total = rows * cols;
    int threads = 256;
    transpose_2d<<<(total + threads - 1) / threads, threads>>>(in, out, rows, cols);
    cudaDeviceSynchronize();
  }

  // ======================== BATCHED 3-D TRANSPOSE ========================
  // (Nbatch, M, K) → (Nbatch, K, M)
  // Each thread transposes one element; blockIdx.z selects the batch slice.
}
// ======================== HELPER FUNCTIONS ========================

void fill_rand_float(float *arr, int size) {
    for (int i = 0; i < size; i++) {
        // Generating numbers between 0.0 and 1.0
        arr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

// Helper function to print a matrix
void print_matrix(const char* name, float* mat, int batch, int rows, int cols) {
    printf("\n=== %s ===\n", name);
    for (int b = 0; b < batch; b++) {
        if (batch > 1) printf("Batch %d:\n", b);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                // (batch * rows * cols) + (r * cols) + c
                printf("%6.2f ", mat[(b * rows + r) * cols + c]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

// ======================== MAIN TESTING FUNCTION ========================

int main() {
    // Seed random generation
    srand(time(NULL));

    // NOTE: Lowered dimensions here so the terminal doesn't get flooded. 
    // Change these back to 64 when doing performance testing!
    int M = 2;
    int N = 2;
    int K = 2;
    int Nbatch = 3;

    // Calculate sizes
    int size_A  = Nbatch * M * K;
    int size_B  = K * N;          // B is shared across batches
    int size_C  = Nbatch * M * N;

    printf("Allocating memory for M=%d, N=%d, K=%d, Batch=%d...\n", M, N, K, Nbatch);

    // Host memory allocation
    float *h_A  = (float *)malloc(size_A * sizeof(float));
    float *h_B  = (float *)malloc(size_B * sizeof(float));
    float *h_C  = (float *)malloc(size_C * sizeof(float));
    float *h_dC = (float *)malloc(size_C * sizeof(float));
    float *h_dA = (float *)malloc(size_A * sizeof(float));
    float *h_dB = (float *)malloc(size_B * sizeof(float));

    // Initialize host data with random floats [0.0, 1.0]
    fill_rand_float(h_A, size_A);
    fill_rand_float(h_B, size_B);
    fill_rand_float(h_dC, size_C);

    // Device memory allocation
    float *d_A, *d_B, *d_C, *d_dC, *d_dA, *d_dB;
    cudaMalloc((void **)&d_A,  size_A * sizeof(float));
    cudaMalloc((void **)&d_B,  size_B * sizeof(float));
    cudaMalloc((void **)&d_C,  size_C * sizeof(float));
    cudaMalloc((void **)&d_dC, size_C * sizeof(float));
    cudaMalloc((void **)&d_dA, size_A * sizeof(float));
    cudaMalloc((void **)&d_dB, size_B * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dC, h_dC, size_C * sizeof(float), cudaMemcpyHostToDevice);

    // Testing Forward Pass
    printf("Running Forward Pass (Batched MatMul)...\n");
    seera_cuda::cuda_matmul(d_A, d_B, d_C, M, N, K, Nbatch);

    // Testing Backward Pass
    printf("Running Backward Pass (Gradients)...\n");
    seera_cuda::cuda_matmul_bwd(d_A, d_B, d_dC, d_dA, d_dB, M, N, K, Nbatch);

    // Copy results back to host for inspection
    cudaMemcpy(h_C,  d_C,  size_C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dA, d_dA, size_A * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dB, d_dB, size_B * sizeof(float), cudaMemcpyDeviceToHost);

    // --- PRINT THE MATRICES ---
    print_matrix("Matrix A (Input)", h_A, Nbatch, M, K);
    print_matrix("Matrix B (Input, shared across batches)", h_B, 1, K, N);
    print_matrix("Matrix C (Forward Output)", h_C, Nbatch, M, N);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_dC);
    free(h_dA);
    free(h_dB);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_dC);
    cudaFree(d_dA);
    cudaFree(d_dB);

    return 0;
}