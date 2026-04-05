#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda.h>
#include <time.h>
#include <math.h>
#include <cuda_fp16.h>

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

__global__ void maxPool(
    half *input_image, half *conv,
    int C, int H, int W, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int H_out, int W_out)
{
    int total_elements = H_out * W_out * C;
    int batchN = blockIdx.y;
    int index_ = blockIdx.x * blockDim.x + threadIdx.x;

    half temp = 0;
    if (index_ < total_elements)
    {
        int w_out = index_ % W_out;
        int h_out = (index_ / W_out) % H_out;
        int ni = index_ / (H_out * W_out);
        int ht = h_out * stride_h - pad_h;
        int wt = w_out * stride_w - pad_w;
        if (ht >= 0 && ht < H && wt >= 0 && wt < W && ni < C)
        {
            int input_idx = ((batchN * C + ni) * H + ht) * W + wt;
            temp = input_image[input_idx];
        }
        for (int ir = 0; ir < R; ir++)
        {
            for (int is = 0; is < S; is++)
            {
                int h_in = h_out * stride_h - pad_h + ir;
                int w_in = w_out * stride_w - pad_w + is;

                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W && ni < C)
                {
                    int input_idx = ((batchN * C + ni) * H + h_in) * W + w_in;
                    half ing = input_image[input_idx];
                    if (temp <= ing)
                        temp = ing;
                }
                else
                {
                    if (temp <= __float2half(0.0f))
                        temp =  __float2half(0.0f);
                }
            }
        }
        conv[(batchN * C + ni) * H_out * W_out + h_out * W_out + w_out] = temp;
    }
}
__global__ void maxPool_bwd(
    half *dX, half *dout, short *mask,
    int C, int H, int W, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int W_out, int H_out)
{
    int c = blockIdx.y;
    int BatchN = blockIdx.z;
    int index_ = blockIdx.x * blockDim.x + threadIdx.x;

    int rs = R * S;

    int h_out = index_ / W_out;
    int w_out = index_ % W_out;
    for (int ii = 0; ii < rs; ii++)
    {
        int s = ii % S;
        int r = ii / S;
        int h_in = h_out * stride_h - pad_h + r;
        int w_in = w_out * stride_w - pad_w + s;

        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
        {
            int index = (BatchN * C + c) * H * W + h_in * W + w_in;
            atomicAdd(&(dX[index]), dout[(BatchN * C + c) * H_out * W_out + index_] * ((half)mask[index]));
        }
    }
}



float *cuda_maxpool(float *h_image_f,
                   int batchN, int C, int H, int W,
                   int R, int S,
                   int pad_h, int pad_w,
                   int stride_h, int stride_w)
{
    int H_out = (H + 2 * pad_h - R) / stride_h + 1;
    int W_out = (W + 2 * pad_w - S) / stride_w + 1;

    int input_elems = batchN * C * H * W;
    int output_elems = batchN * C * H_out * W_out;
    int output_elems_wob = C * H_out * W_out;
    float *d_image_f,*d_out;
    half *d_image;
    half *d_conv;

    cudaMalloc(&d_image_f, sizeof(float) * input_elems);
    cudaMalloc(&d_image, sizeof(half) * input_elems);
    cudaMalloc(&d_conv, sizeof(half) * output_elems);
    cudaMalloc(&d_out, sizeof(float) * output_elems);

    cudaMemset(d_conv, 0, sizeof(half) * output_elems);
    cudaMemcpy(d_image_f, h_image_f, sizeof(float) * input_elems, cudaMemcpyHostToDevice);

    float2halff<<<ceil((float)input_elems / 512), 512>>>(d_image_f, d_image);
    int threads_per_block = 256;
    int Nblock = (output_elems_wob + threads_per_block - 1) / threads_per_block;
    dim3 blocks(Nblock, batchN);

    maxPool<<<blocks, threads_per_block>>>(
        d_image, d_conv,
        C, H, W, R, S,
        pad_h, pad_w, stride_h, stride_w,
        H_out, W_out);
    half2float<<<ceil((float)output_elems / 512), 512>>>(d_conv, d_out);

    cudaDeviceSynchronize();

    float *h_conv = new float[output_elems];
    cudaMemcpy(h_conv, d_out, sizeof(float) * output_elems, cudaMemcpyDeviceToHost);

    cudaFree(d_image_f);
    cudaFree(d_image);
    cudaFree(d_conv);
    cudaFree(d_out);

    return h_conv;
}

half *cuda_maxpool_gputogpu(half *h_image,
                   int batchN, int C, int H, int W,
                   int R, int S,
                   int pad_h, int pad_w,
                   int stride_h, int stride_w)
{
    int H_out = (H + 2 * pad_h - R) / stride_h + 1;
    int W_out = (W + 2 * pad_w - S) / stride_w + 1;

    int input_elems = batchN * C * H * W;
    int output_elems = batchN * C * H_out * W_out;
    int output_elems_wob = C * H_out * W_out;

    half *d_conv;


    cudaMalloc(&d_conv, sizeof(half) * output_elems);

    cudaMemset(d_conv, 0, sizeof(half) * output_elems);


    int threads_per_block = 256;
    int Nblock = (output_elems_wob + threads_per_block - 1) / threads_per_block;
    dim3 blocks(Nblock, batchN);

    maxPool<<<blocks, threads_per_block>>>(
        h_image, d_conv,
        C, H, W, R, S,
        pad_h, pad_w, stride_h, stride_w,
        H_out, W_out);



    cudaDeviceSynchronize();



    return d_conv;
}

half *cuda_maxpool_bwd_gputogpu(
    half *d_dout,   // gradient from upstream, already on GPU, shape [N,C,H_out,W_out]
    short *d_mask,  // mask from forward pass, already on GPU, shape [N,C,H,W]
    int batchN, int C, int H, int W,
    int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w)
{
    int H_out = (H + 2 * pad_h - R) / stride_h + 1;
    int W_out = (W + 2 * pad_w - S) / stride_w + 1;

    int input_elems        = batchN * C * H * W;
    int output_elems_wob   = H_out * W_out;   // per-batch, per-channel

    // Allocate dX (gradient w.r.t. input), same shape as input
    half *d_dX;
    cudaMalloc(&d_dX, sizeof(half) * input_elems);
    cudaMemset(d_dX,  0, sizeof(half) * input_elems);

    int threads_per_block = 256;
    int Nblock = (output_elems_wob + threads_per_block - 1) / threads_per_block;

    // grid: x = spatial blocks, y = channels, z = batch
    dim3 blocks(Nblock, C, batchN);

    maxPool_bwd<<<blocks, threads_per_block>>>(
        d_dX, d_dout, d_mask,
        C, H, W, R, S,
        pad_h, pad_w, stride_h, stride_w,
        W_out, H_out);

    cudaDeviceSynchronize();

    return d_dX;   // caller owns this allocation
}