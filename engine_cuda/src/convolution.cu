

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <mma.h>
#include <time.h>
#include <vector>
namespace seera_cuda
{
    using namespace nvcuda;

    __global__ void convulution_eff(const float *input_image, float *conv,
                                    float *kernel, int N, int C, int H, int W,
                                    int R, int S, int pad_h, int pad_w,
                                    int stride_h, int stride_w, int H_out,
                                    int W_out)
    {
        __shared__ half iw2col[16 * 16];
        __shared__ half krl[16 * 16];

        int batchno = blockIdx.z;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> im2col_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> krl_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

        wmma::fill_fragment(acc_frag, 0.0f);
        for (int p = 0; p < C * S * R; p += 16)
        {
            for (int i = 0; i < 8; i++)
            {
                int tid = threadIdx.x + i * 32; //-> 255 - unique

                int local_row = (tid / 16); // for row divide hota hai
                int local_col = (tid % 16); // for column mod hota hai

                int global_image = blockIdx.x * 16 + local_row; // -> H_out*W_out
                int ni = blockIdx.y * 16 + local_col;           // -> N
                int global_kernel_ = p + local_row;             // -> C*S*R
                int global_kernel = p + local_col;

                int iy_w_out = (global_image % W_out);
                int iy_h_out = global_image / W_out;

                int ky_is = global_kernel % S;
                int index_ = global_kernel / S;
                int ky_ir = index_ % R;
                int ky_ic = (index_ / R);

                int h_in = iy_h_out * stride_h - pad_h + ky_ir;
                int w_in = iy_w_out * stride_w - pad_w + ky_is;

                if (ni < N && ky_is < S && ky_ir < R)
                {
                    int kernel_idx = (ni) * (C * S * R) +
                                     global_kernel_; // isiliye since kernel ka row is used
                    krl[tid] = __float2half(kernel[kernel_idx]);
                }
                else
                {
                    krl[tid] = __float2half(0.0f);
                }
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W && ky_ic < C)
                {
                    int input_idx = ((batchno * C + ky_ic) * H + h_in) * W + w_in;
                    iw2col[tid] = __float2half(
                        input_image[input_idx]); //- half(1); // and subsequntly im2col
                                                 // matrix ka column use ho raha hai.
                }
                else
                {
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

        for (int i = 0; i < 8; i++)
        {
            int tid = threadIdx.x + i * 32; //-> 255
            int index_ = blockIdx.x * 16 + (tid / 16);

            int ni = blockIdx.y * 16 + (tid % 16); // -> N

            if (ni < N && index_ < H_out * W_out)
                conv[(batchno * N + ni) * H_out * W_out + index_] =
                    sha_conv[tid];
        }
    }

    __global__ void _weight_reduce(float *dW, float *dwn, int BatchN, int Cin,
                                   int Cout, int R, int S)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x; //-> Cout,Cin,R,S
        int s = index % S;
        int r = (index / S) % R;
        int cout = (index / (S * R)) % Cout;
        int cin = index / (S * R * Cout);
        float temp = 0.0f;
        if (s < S && r < R && cin < Cin && cout < Cout)
        {
            for (int ii = 0; ii < BatchN; ii++)
            {
                temp += dwn[ii * Cout * Cin * R * S + index];
            }
            dW[index] = temp;
        }
    }

    // namespace seera_cuda

    // ============================================================================
    // BACKWARD PASS for Conv2d
    // ============================================================================
    // Forward: Y[b,n,h_out,w_out] = sum_{c,r,s} W[n,c,r,s] *
    // X[b,c,h_out*s_h-p_h+r,h_out*s_w-p_w+s]
    //
    // dX = ConvTranspose2d(dY, W)
    //   dY: [batch, N, H_out, W_out],  W: [N, C, R, S]
    //   dX: [batch, C, H, W]
    //   ConvTranspose with Cin=N, Cout=C, Hin=H_out, Win=W_out
    //
    // dW[n,c,r,s] = sum_{b,h_out,w_out} dY[b,n,h_out,w_out] *
    // X[b,c,h_out*s-p+r,w_out*s-p+s]
    //   Per-batch: dW_b[N, C*R*S] = dY_b[N, H_out*W_out] @ im2col(X_b)[C*R*S,
    //   H_out*W_out]^T Then reduce: dW = sum_b dW_b
    // ============================================================================

    // --- dX kernels: ConvTranspose(dY, W) → dX ---
    // Reusing the same WMMA conv-transpose matmul pattern from upsampling.cu

    __global__ void conv2d_bwd_transmatmul(float *W, float *X, float *C_out, int M,   //-> WROTE THIS KERNEL EVEN AFTER 48 HOURS OF NO SLEEP!!!! PEAK 
                                           int N, int K, int Cout, int R, int S,
                                           int BATCH, int Hin, int Win)
    {
        int warpM = blockIdx.y * 16;
        int warpN = blockIdx.x * 16;

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
                int A_s = global_row_A % S;
                int A_r = (global_row_A / S) % R;
                int A_cout = global_row_A / (S * R);

                int A_cin = p + col;
                if (global_row_A < M && A_cin < K)
                {
                    shA[linear_idx] = __float2half(
                        W[A_cin * (Cout * R * S) + A_cout * (R * S) + A_r * S + A_s]);
                }
                else
                {
                    shA[linear_idx] = __float2half(0.0f);
                }

                int B_cin = p + row;
                int global_col_B = warpN + col;
                int B_w = global_col_B % Win;
                int B_h = (global_col_B / Win) % Hin;
                int B_b = global_col_B / (Win * Hin);
                if (B_cin < K && global_col_B < N)
                {
                    shB[linear_idx] = __float2half(
                        X[B_b * K * Hin * Win + B_cin * Hin * Win + B_h * Win + B_w]);
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

        __shared__ float shC[16 * 16];
        wmma::store_matrix_sync(shC, acc_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (int i = 0; i < 8; ++i)
        {
            int linear_idx = threadIdx.x + i * 32;
            int row = linear_idx / 16;
            int col = linear_idx % 16;
            int gRow = warpM + row;
            int gCol = warpN + col;
            if (gRow < M && gCol < N)
            {
                C_out[gRow * N + gCol] = shC[linear_idx];
            }
        }
    }

    // col2im kernel for dX reconstruction
    __global__ void conv2d_bwd_col2im(float *d_out, float *d_in, int C, int R,
                                      int S, int H_in, int W_in, int H_out,
                                      int W_out, int stridew, int strideh, int padh,
                                      int padw, int total_N)
    {
        int globalid = blockIdx.x * blockDim.x + threadIdx.x;
        int c = blockIdx.y;
        int batchN = blockIdx.z;
        int sr = S * R;
        int w_in = globalid % W_in;
        int h_in = globalid / W_in;

        if (h_in >= H_in || w_in >= W_in || c >= C)
            return;

        float temp = 0.0f;
        for (int row = 0; row < sr; row++)
        {
            int s = row % S;
            int r = row / S;

            int h_num = h_in + padh - r;
            int w_num = w_in + padw - s;

            if (h_num % strideh != 0 || w_num % stridew != 0)
                continue;

            int h_out = h_num / strideh;
            int w_out = w_num / stridew;

            if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out)
            {
                int gemm_row = c * sr + row;
                int gemm_col = batchN * H_out * W_out + h_out * W_out + w_out;
                temp += d_in[gemm_row * total_N + gemm_col];
            }
        }

        d_out[(batchN * C + c) * H_in * W_in + globalid] = temp;
    }

    // --- dW kernel: per-batch fused WMMA ---
    __global__ void conv2d_dW_kernel(float *dY, float *X, float *dW_batch,
                                     int N_out, int C, int H, int W, int R, int S,
                                     int H_out, int W_out, int strideh, int stridew,
                                     int padh, int padw)
    {
        int warpM = blockIdx.y * 16; // N (output channels) dimension
        int warpN = blockIdx.x * 16; // C*R*S dimension
        int batchIdx = blockIdx.z;

        int spatial = H_out * W_out; // contraction dimension
        int CRS = C * R * S;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

        wmma::fill_fragment(acc_frag, 0.0f);

        __shared__ half shA[16 * 16]; // dY tile
        __shared__ half shB[16 * 16]; // im2col(X) tile

        for (int p = 0; p < spatial; p += 16)
        {
            for (int i = 0; i < 8; ++i)
            {
                int linear_idx = threadIdx.x + i * 32;
                int row = linear_idx / 16;
                int col = linear_idx % 16;

                // A = dY_b[N, H_out*W_out]: load A[warpM+row, p+col]
                int n = warpM + row;
                int sp_idx = p + col;
                if (n < N_out && sp_idx < spatial)
                {
                    shA[linear_idx] =
                        __float2half(dY[batchIdx * N_out * spatial + n * spatial + sp_idx]);
                }
                else
                {
                    shA[linear_idx] = __float2half(0.0f);
                }

                // B = im2col(X_b)[CRS, H_out*W_out] — fused load
                int crs_idx = warpN + row;
                int sp_idx_b = p + col;

                if (crs_idx < CRS && sp_idx_b < spatial)
                {
                    int s = crs_idx % S;
                    int r = (crs_idx / S) % R;
                    int c = crs_idx / (R * S);

                    int w_out = sp_idx_b % W_out;
                    int h_out = sp_idx_b / W_out;

                    int h_in = h_out * strideh - padh + r;
                    int w_in = w_out * stridew - padw + s;

                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
                    {
                        shB[linear_idx] = __float2half(
                            X[batchIdx * C * H * W + c * H * W + h_in * W + w_in]);
                    }
                    else
                    {
                        shB[linear_idx] = __float2half(0.0f);
                    }
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

        // Store dW_b[N, CRS] for this batch element
        __shared__ float shC[16 * 16];
        wmma::store_matrix_sync(shC, acc_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (int i = 0; i < 8; ++i)
        {
            int linear_idx = threadIdx.x + i * 32;
            int row = linear_idx / 16;
            int col = linear_idx % 16;
            int gRow = warpM + row;
            int gCol = warpN + col;
            if (gRow < N_out && gCol < CRS)
            {
                dW_batch[batchIdx * N_out * CRS + gRow * CRS + gCol] = shC[linear_idx];
            }
        }
    }
    void cuda_conv2d_fwd(float *h_image, float *h_kernel, float *d_conv, int batchN,
                         int C, int H, int W, int N, int R, int S, int pad_h,
                         int pad_w, int stride_h, int stride_w)
    {
        int H_out = (H + 2 * pad_h - R) / stride_h + 1;
        int W_out = (W + 2 * pad_w - S) / stride_w + 1;

        int input_elems = batchN * C * H * W;
        int kernel_elems = N * C * R * S;
        int output_elems = batchN * N * H_out * W_out;

        // Launch convolution kernel
        int aa1 = ceil((float)H_out * W_out / 16);
        int aa2 = ceil((float)N / 16);
        dim3 tpb(32, 1);
        dim3 block(aa1, aa2, batchN);

        convulution_eff<<<block, tpb>>>(h_image, d_conv, h_kernel, N, C, H, W, R, S,
                                        pad_h, pad_w, stride_h, stride_w, H_out,
                                        W_out);

        cudaDeviceSynchronize();
    }
    // ============================================================================
    // Conv2d backward entry point
    // ============================================================================
    void cuda_conv2d_bwd(float *W, float *X, float *dY, float *dX, float *dW,
                         int batch, int C, int H, int W_in, int N, int R, int S,
                         int strideh, int stridew, int padh, int padw)
    {
        int H_out = (H + 2 * padh - R) / strideh + 1;
        int W_out = (W_in + 2 * padw - S) / stridew + 1;
        int CRS = C * R * S;

        // Step 1: dX = ConvTranspose(dY, W)
        {
            int total_N_gemm = batch * H_out * W_out;
            int M_gemm = CRS; // C * R * S
            int K_gemm = N;   // contraction dim

            float *intermediate;
            cudaMalloc(&intermediate, M_gemm * total_N_gemm * sizeof(float));

            dim3 block(32);
            dim3 grid((total_N_gemm + 15) / 16, (M_gemm + 15) / 16);

            conv2d_bwd_transmatmul<<<grid, block>>>(W, dY, intermediate, M_gemm,
                                                    total_N_gemm, K_gemm, C, R, S,
                                                    batch, H_out, W_out);
            cudaDeviceSynchronize();

            // col2im: fold [C*R*S, batch*H_out*W_out] → [batch, C, H, W_in]
            int tpb = 256;
            dim3 grid_c2i((H * W_in + tpb - 1) / tpb, C, batch);
            dim3 block_c2i(tpb);

            conv2d_bwd_col2im<<<grid_c2i, block_c2i>>>(
                dX, intermediate, C, R, S, H, W_in, H_out, W_out, stridew, strideh,
                padh, padw, total_N_gemm);
            cudaDeviceSynchronize();

            cudaFree(intermediate);
        }

        // Step 2: dW via per-batch fused WMMA + reduce
        {
            float *dW_batch;
            cudaMalloc(&dW_batch, batch * N * CRS * sizeof(float));

            dim3 block_dw(32);
            dim3 grid_dw((CRS + 15) / 16, (N + 15) / 16, batch);

            conv2d_dW_kernel<<<grid_dw, block_dw>>>(dY, X, dW_batch, N, C, H, W_in, R,
                                                    S, H_out, W_out, strideh, stridew,
                                                    padh, padw);
            cudaDeviceSynchronize();

            // Reduce across batch: reuse _weight_reduce pattern
            int dw_total = N * CRS;
            int tpb = 256;
            _weight_reduce<<<(dw_total + tpb - 1) / tpb, tpb>>>(dW, dW_batch, batch, C,
                                                                N, R, S);
            cudaDeviceSynchronize();

            cudaFree(dW_batch);
        }
    }

} // namespace seera_cuda