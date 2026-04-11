#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
namespace seera_cuda
{
    __global__ void unpooling(float *inp, float *out, int N, int C, int H, int W, int sh, int sw) //-> launch with H*W(incoming)
    {
        int H_out = sh * H;
        int W_out = sw * W;

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int w_out = index % W_out;
        int h_out = index / W_out;
        int c = blockIdx.y;
        int batchN = blockIdx.z;

        if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out)
        {
            int h_in = h_out / sh;
            int w_in = w_out / sw;

            int common = batchN * C + c;
            out[common * H_out * W_out + h_out * W_out + w_out] = inp[common * H * W + h_in * W + w_in];
        }
    }

    __global__ void unpooling_bwd(float *inp, float *dx, int N, int C, int H, int W, int sh, int sw) // -> launch with incoming as per fwd call
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int w = index % W;
        int h = index / W;
        int c = blockIdx.y;
        int batchN = blockIdx.z;
        int H_out = sh * H;
        int W_out = sw * W;
        if (h >= 0 && h < H && w >= 0 && w < W)
        {
            float temp = 0.0f;
            for (int ii = 0; ii < sh; ii++)
            {
                for (int jj = 0; jj < sw; jj++)
                {
                    int h_ = h * sh + ii;
                    int w_ = w * sw + jj;

                    temp += inp[(batchN * C + c) * H_out * W_out + h_ * W_out + w_];
                }
            }
            int common = batchN * C + c;
            dx[common * H * W + index] = temp;
        }
    }

    void cuda_unpooling_fwd(float *d_inp, float *d_out,
                                 int batchN, int C, int H, int W,
                                 int sh, int sw)
    {
        int H_out = sh * H;
        int W_out = sw * W;

        int output_elems = batchN * C * H_out * W_out;
        int spatial_out = H_out * W_out;

        // assume d_out already allocated
        cudaMemset(d_out, 0, sizeof(float) * output_elems);

        int threads = 256;
        int Nblock = (spatial_out + threads - 1) / threads;
        dim3 blocks(Nblock, C, batchN);

        unpooling<<<blocks, threads>>>(d_inp, d_out,
                                       batchN, C, H, W,
                                       sh, sw);

        cudaDeviceSynchronize();
    }
    void cuda_unpooling_bwd(float *d_dout, float *d_dx,
                                     int batchN, int C, int H, int W,
                                     int sh, int sw)
    {
        int input_elems = batchN * C * H * W;
        int spatial_in = H * W;

        // assume d_dx already allocated
        cudaMemset(d_dx, 0, sizeof(float) * input_elems);

        int threads = 256;
        int Nblock = (spatial_in + threads - 1) / threads;
        dim3 blocks(Nblock, C, batchN);

        unpooling_bwd<<<blocks, threads>>>(d_dout, d_dx,
                                           batchN, C, H, W,
                                           sh, sw);

        cudaDeviceSynchronize();
    }
}