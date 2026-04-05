
#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
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

__global__ void convulution_eff(
    const half *input_image, half *conv, half *kernel,
    int N, int C, int H, int W, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int H_out, int W_out)
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

            int local_row = (tid / 16);  // for row divide hota hai
            int local_col = (tid % 16);  // for column mod hota hai 

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
                int kernel_idx = ni * (C * S * R) + global_kernel_; // isiliye since kernel ka row is used
                krl[tid] = kernel[kernel_idx];
            }
            else
            {
                krl[tid] = 0;
            }
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W && ky_ic < C)
            {
                int input_idx = ((batchno * C + ky_ic) * H + h_in) * W + w_in;
                iw2col[tid] = input_image[input_idx]; //- half(1); // and subsequntly im2col matrix ka column use ho raha hai. 
            }
            else
            {
                iw2col[tid] = (half)0;
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
            conv[(batchno * N + ni) * H_out * W_out + index_] = sha_conv[tid];
    }
}

float *cuda_conv2d(float *h_image_f, float *h_kernel_f,
                   int batchN, int C, int H, int W,
                   int N, int R, int S,
                   int pad_h, int pad_w,
                   int stride_h, int stride_w)
{
    int H_out = (H + 2 * pad_h - R) / stride_h + 1;
    int W_out = (W + 2 * pad_w - S) / stride_w + 1;

    int input_elems = batchN * C * H * W;
    int kernel_elems = N * C * R * S;
    int output_elems = batchN * N * H_out * W_out;

    float *d_image_f, *d_kernel_f, *d_out;
    half *d_image, *d_kernel;
    half *d_conv;

    cudaMalloc(&d_image_f, sizeof(float) * input_elems);
    cudaMalloc(&d_kernel_f, sizeof(float) * kernel_elems);

    cudaMalloc(&d_image, sizeof(half) * input_elems);
    cudaMalloc(&d_kernel, sizeof(half) * kernel_elems);

    cudaMalloc(&d_conv, sizeof(half) * output_elems);
    cudaMalloc(&d_out, sizeof(float) * output_elems);

    cudaMemset(d_conv, 0, sizeof(half) * output_elems);

    cudaMemcpy(d_image_f, h_image_f, sizeof(float) * input_elems, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_f, h_kernel_f, sizeof(float) * kernel_elems, cudaMemcpyHostToDevice);

    float2halff<<<ceil((float)input_elems / 512), 512>>>(d_image_f, d_image);
    float2halff<<<ceil((float)kernel_elems / 512), 512>>>(d_kernel_f, d_kernel);

    int aa1 = ceil((float)H_out * W_out / 16);
    int aa2 = ceil((float)N / 16);
    dim3 tpb(32, 1);
    dim3 block(aa1, aa2, batchN);

    convulution_eff<<<block, tpb>>>(
        d_image, d_conv, d_kernel,
        N, C, H, W, R, S,
        pad_h, pad_w, stride_h, stride_w,
        H_out, W_out);

    half2float<<<ceil((float)output_elems / 512), 512>>>(d_conv, d_out);
    cudaDeviceSynchronize();

    float *h_conv = new float[output_elems];

    cudaMemcpy(h_conv, d_out, sizeof(float) * output_elems, cudaMemcpyDeviceToHost);

    cudaFree(d_image_f);
    cudaFree(d_kernel_f);
    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_conv);
    cudaFree(d_out);
    return h_conv;
}

half *cuda_conv2d_gputogpu(half *h_image, half *h_kernel,
                           int batchN, int C, int H, int W,
                           int N, int R, int S,
                           int pad_h, int pad_w,
                           int stride_h, int stride_w)
{
    int H_out = (H + 2 * pad_h - R) / stride_h + 1;
    int W_out = (W + 2 * pad_w - S) / stride_w + 1;

    int input_elems = batchN * C * H * W;
    int kernel_elems = N * C * R * S;
    int output_elems = batchN * N * H_out * W_out;

    // Device pointers
    // half  *d_image, *d_kernel;
    half *d_conv;

    cudaMalloc(&d_conv, sizeof(half) * output_elems);
    cudaMemset(d_conv, 0, sizeof(half) * output_elems);

    // Launch convolution kernel
    int aa1 = ceil((float)H_out * W_out / 16);
    int aa2 = ceil((float)N / 16);
    dim3 tpb(32, 1);
    dim3 block(aa1, aa2, batchN);

    convulution_eff<<<block, tpb>>>(
        h_image, d_conv, h_kernel,
        N, C, H, W, R, S,
        pad_h, pad_w, stride_h, stride_w,
        H_out, W_out);

    cudaDeviceSynchronize();

    return d_conv;
}