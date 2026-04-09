#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
namespace seera_cuda {
using namespace nvcuda;

// ============================================================================
// FORWARD PASS: Conv2dTranspose
// ============================================================================
// X(Batch,Cin,Hin,Win) * W(Cin,Cout,R,S) -> (Batch,Cout,Hout,Wout)
__global__ void
conv2dTransmatmul(float *X, float *W, float *C, int M, int N, int K, int Cout,
                  int R, int S, int BATCH, int Hin,
                  int Win) { // k-> Cin, M-> Cout*R*S, N-> Batch*H_in*W_in
  int warpM = blockIdx.y * 16;
  int warpN = blockIdx.x * 16;

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
      int A_s = global_row_A % S;
      int A_r = (global_row_A / S) % R;
      int A_cout = global_row_A / (S * R);

      int A_cin = p + col;
      if (global_row_A < M && A_cin < K) {
        shA[linear_idx] = __float2half(
            W[A_cin * (Cout * R * S) + A_cout * (R * S) + A_r * S + A_s]);
      } else {
        shA[linear_idx] = __float2half(0.0f);
      }

      int B_cin = p + row;
      int global_col_B = warpN + col;
      int B_w = global_col_B % Win;
      int B_h = (global_col_B / Win) % Hin;
      int B_b = global_col_B / (Win * Hin);
      if (B_cin < K && global_col_B < N) {
        shB[linear_idx] = __float2half(
            X[B_b * K * Hin * Win + B_cin * Hin * Win + B_h * Win + B_w]);
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

  __shared__ float shC[16 * 16];
  wmma::store_matrix_sync(shC, acc_frag, 16, wmma::mem_row_major);
  __syncthreads();

  for (int i = 0; i < 8; ++i) {
    int linear_idx = threadIdx.x + i * 32;
    int row = linear_idx / 16;
    int col = linear_idx % 16;

    int global_row_C = warpM + row;
    int global_col_C = warpN + col;

    if (global_row_C < M && global_col_C < N) {
      C[(global_row_C)*N + global_col_C] = shC[linear_idx];
    }
  }
}

__global__ void col2im(float *d_out, float *d_in, int C, int R, int S, int H_in,
                       int W_in, int H_out, int W_out, int stridew, int strideh,
                       int padh, int padw, int total_N) {
  int globalid = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y;
  int batchN = blockIdx.z;
  int sr = S * R;
  int w_in = globalid % W_in;
  int h_in = globalid / W_in;

  if (h_in >= H_in || w_in >= W_in || c >= C)
    return;

  float temp = 0.0f;
  for (int row = 0; row < sr; row++) {
    int s = row % S;
    int r = row / S;

    int h_num = h_in + padh - r;
    int w_num = w_in + padw - s;

    if (h_num % strideh != 0 || w_num % stridew != 0)
      continue;

    int h_out = h_num / strideh;
    int w_out = w_num / stridew;

    if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
      int gemm_row = c * sr + row;
      int gemm_col = batchN * H_out * W_out + h_out * W_out + w_out;
      temp += d_in[gemm_row * total_N + gemm_col];
    }
  }

  d_out[(batchN * C + c) * H_in * W_in + globalid] = temp;
}

__global__ void convulution_eff_bwd(const float *input_image, float *conv,
                                    float *kernel, int N, int C, int H, int W,
                                    int R, int S, int pad_h, int pad_w,
                                    int stride_h, int stride_w, int H_out,
                                    int W_out) {
  __shared__ half iw2col[16 * 16];
  __shared__ half krl[16 * 16];

  int batchno = blockIdx.z;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> im2col_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> krl_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

  wmma::fill_fragment(acc_frag, 0.0f);
  for (int p = 0; p < C * S * R; p += 16) {
    for (int i = 0; i < 8; i++) {
      int tid = threadIdx.x + i * 32;
      int local_row = tid / 16;
      int local_col = tid % 16;

      int global_image = blockIdx.x * 16 + local_row;
      int ni = blockIdx.y * 16 + local_col;
      int global_kernel_ = p + local_row;
      int global_kernel = p + local_col;

      int iy_w_out = global_image % W_out;
      int iy_h_out = global_image / W_out;

      int ky_is = global_kernel % S;
      int index_ = global_kernel / S;
      int ky_ir = index_ % R;
      int ky_ic = index_ / R;

      int h_in = iy_h_out * stride_h - pad_h + ky_ir;
      int w_in = iy_w_out * stride_w - pad_w + ky_is;

      if (ni < N && ky_is < S && ky_ir < R) {
        int kernel_idx = ni * (C * S * R) + global_kernel_;
        krl[tid] = __float2half(kernel[kernel_idx]);
      } else {
        krl[tid] = __float2half(0.0f);
      }

      if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W && ky_ic < C) {
        int input_idx = ((batchno * C + ky_ic) * H + h_in) * W + w_in;
        iw2col[tid] = __float2half(input_image[input_idx]);
      } else {
        iw2col[tid] = __float2half(0.0f);
      }
    }
    __syncthreads();
    wmma::load_matrix_sync(im2col_frag, iw2col, 16);
    wmma::load_matrix_sync(krl_frag, krl, 16);
    wmma::mma_sync(acc_frag, im2col_frag, krl_frag, acc_frag);
    __syncthreads();
  }

  __shared__ float sha_conv[16 * 16];
  wmma::store_matrix_sync(sha_conv, acc_frag, 16, wmma::mem_row_major);

  for (int i = 0; i < 8; i++) {
    int tid = threadIdx.x + i * 32;
    int index_ = blockIdx.x * 16 + (tid / 16);
    int ni = blockIdx.y * 16 + (tid % 16);

    if (ni < N && index_ < H_out * W_out)
      conv[(batchno * N + ni) * H_out * W_out + index_] = sha_conv[tid];
  }
}

// ============================================================================
// dW WMMA kernel: per-batch fused im2col + GEMM
// ============================================================================
__global__ void conv2dTrans_dW_kernel(float *X, float *dY, float *dW_batch,
                                      int Cin, int Hin, int Win, int Cout,
                                      int KH, int KW, int Hout, int Wout,
                                      int strideh, int stridew, int padh,
                                      int padw) {
  int warpM = blockIdx.y * 16; // Cin dimension
  int warpN = blockIdx.x * 16; // Cout*KH*KW dimension
  int batchIdx = blockIdx.z;

  int spatial = Hin * Win; // K dimension for the GEMM (contraction dim)
  int CRS = Cout * KH * KW;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  __shared__ half shA[16 * 16]; // X tile
  __shared__ half shB[16 * 16]; // im2col(dY) tile

  for (int p = 0; p < spatial; p += 16) {
    for (int i = 0; i < 8; ++i) {
      int linear_idx = threadIdx.x + i * 32;
      int row = linear_idx / 16;
      int col = linear_idx % 16;

      int cin = warpM + row;
      int hw_idx = p + col;
      if (cin < Cin && hw_idx < spatial) {
        shA[linear_idx] =
            __float2half(X[batchIdx * Cin * spatial + cin * spatial + hw_idx]);
      } else {
        shA[linear_idx] = __float2half(0.0f);
      }

      int crs_idx = warpN + row;
      int hw_idx_b = p + col;

      if (crs_idx < CRS && hw_idx_b < spatial) {
        // Decode crs_idx -> cout, kh, kw
        int kw_ = crs_idx % KW;
        int kh_ = (crs_idx / KW) % KH;
        int cout_ = crs_idx / (KH * KW);

        // Decode hw_idx_b -> h_in, w_in (position in input X)
        int w_in = hw_idx_b % Win;
        int h_in = hw_idx_b / Win;

        // Corresponding position in dY
        int h_out = h_in * strideh - padh + kh_;
        int w_out = w_in * stridew - padw + kw_;

        if (h_out >= 0 && h_out < Hout && w_out >= 0 && w_out < Wout) {
          shB[linear_idx] =
              __float2half(dY[batchIdx * Cout * Hout * Wout +
                              cout_ * Hout * Wout + h_out * Wout + w_out]);
        } else {
          shB[linear_idx] = __float2half(0.0f);
        }
      } else {
        shB[linear_idx] = __float2half(0.0f);
      }
    }

    __syncthreads();
    wmma::load_matrix_sync(a_frag, shA, 16);
    wmma::load_matrix_sync(b_frag, shB, 16);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // Store dW_b[Cin, Cout*KH*KW] for this batch element
  __shared__ float shC[16 * 16];
  wmma::store_matrix_sync(shC, acc_frag, 16, wmma::mem_row_major);

  for (int i = 0; i < 8; ++i) {
    int linear_idx = threadIdx.x + i * 32;
    int row = linear_idx / 16;
    int col = linear_idx % 16;

    int gRow = warpM + row;
    int gCol = warpN + col;
    if (gRow < Cin && gCol < CRS) {
      dW_batch[batchIdx * Cin * CRS + gRow * CRS + gCol] = shC[linear_idx];
    }
  }
}

// Reduce dW across batch dimension: dW[i] = sum_b dW_batch[b * total + i]
__global__ void dW_batch_reduce(float *dW, float *dW_batch, int BatchN,
                                int total) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  float temp = 0.0f;
  for (int b = 0; b < BatchN; b++)
    temp += dW_batch[b * total + index];
  dW[index] = temp;
}
void cuda_col2im_gputogpu(float *d_in, float *d_out, int batchN, int C,
                          int H_in, int W_in, int R, int S, int pad_h,
                          int pad_w, int stride_h, int stride_w, int total_N) {
  int H_out = (H_in + 2 * pad_h - R) / stride_h + 1;
  int W_out = (W_in + 2 * pad_w - S) / stride_w + 1;

  int tpb = 256;

  dim3 grid((H_in * W_in + tpb - 1) / tpb, C, batchN);
  dim3 block(tpb);

  col2im<<<grid, block>>>(d_out, d_in, C, R, S, H_in, W_in, H_out, W_out,
                          stride_w, stride_h, pad_h, pad_w, total_N);

  cudaDeviceSynchronize();
}

// ============================================================================
// Backward pass entry point
// ============================================================================
void cuda_conv2DTranspose_bwd(float *W, float *X, float *dY, float *dX,
                              float *dW, int batch, int Cin, int Hin, int Win,
                              int Cout, int KH, int KW, int strideh,
                              int stridew, int padh, int padw) {
  int Hout = (Hin - 1) * strideh - 2 * padh + KH;
  int Wout = (Win - 1) * stridew - 2 * padw + KW;
  int CRS = Cout * KH * KW;

  // ================================================================
  // Step 1: dX = Conv2d(dY, W)
  //   dY: [batch, Cout, Hout, Wout] = conv input  (C_input = Cout)
  //   W:  [Cin, Cout, KH, KW]       = conv kernel (N_output = Cin)
  //   dX: [batch, Cin, Hin, Win]     = conv output
  //   H_out_conv = (Hout + 2*padh - KH)/strideh + 1 = Hin
  // ================================================================
  {
    int aa1 = (Hin * Win + 15) / 16;
    int aa2 = (Cin + 15) / 16;
    dim3 tpb(32, 1);
    dim3 grid(aa1, aa2, batch);

    convulution_eff_bwd<<<grid, tpb>>>(dY, dX, W, Cin, Cout, Hout, Wout, KH, KW,
                                       padh, padw, strideh, stridew, Hin, Win);
    cudaDeviceSynchronize();
  }

  // ================================================================
  // Step 2: dW via per-batch fused WMMA kernel + reduce
  //   For each batch b:
  //     dW_b[Cin, CRS] = X_b[Cin, spatial] @ im2col(dY_b)[CRS, spatial]^T
  //   Then: dW = sum_b dW_b
  // ================================================================
  {
    float *dW_batch;
    cudaMalloc(&dW_batch, batch * Cin * CRS * sizeof(float));

    dim3 block_dw(32);
    dim3 grid_dw((CRS + 15) / 16, (Cin + 15) / 16, batch);

    conv2dTrans_dW_kernel<<<grid_dw, block_dw>>>(X, dY, dW_batch, Cin, Hin, Win,
                                                 Cout, KH, KW, Hout, Wout,
                                                 strideh, stridew, padh, padw);
    cudaDeviceSynchronize();

    // Reduce across batch
    int dw_total = Cin * CRS;
    int tpb = 256;
    dW_batch_reduce<<<(dw_total + tpb - 1) / tpb, tpb>>>(dW, dW_batch, batch,
                                                         dw_total);
    cudaDeviceSynchronize();

    cudaFree(dW_batch);
  }
}
void cuda_conv2DTranpose_fwd(float *hA, float *hB, float *hC, int batch,
                             int Cin, int Hin, int Win, int Cout, int KH,
                             int KW, int strideh, int stridew, int padh,
                             int padw) {
  int Hout = (Hin - 1) * strideh - 2 * padh + KH;
  int Wout = (Win - 1) * stridew - 2 * padw + KW;

  int N = batch * Hin * Win;
  int M = Cout * KH * KW;
  int K = Cin;

  float *intermediate;
  cudaMalloc(&intermediate, M * N * sizeof(float));

  dim3 block(32);
  dim3 grid((N + 15) / 16, (M + 15) / 16);

  conv2dTransmatmul<<<grid, block>>>(hA, hB, intermediate, M, N, K, Cout, KH,
                                     KW, batch, Hin, Win);
  cudaDeviceSynchronize();

  cuda_col2im_gputogpu(intermediate, hC, batch, Cout, Hout, Wout, KH, KW, padh,
                       padw, strideh, stridew, N);

  cudaFree(intermediate);
}

} // namespace seera_cuda