#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
namespace seera_cuda
{
    __global__ void col2im(float *d_out, float *d_in, int C, int R, int S, int H_in, int W_in, int H_out, int W_out, int stridew, int strideh, int padh, int padw)
    {
        // H_in and W_in are of original image
        // C*S*R is in row
        // H_out* W_out is in column
        int globalid = blockIdx.x * blockDim.x + threadIdx.x;
        int c = blockIdx.y;
        int batchN = blockIdx.z;
        int sr = S * R;
        int csr = C * S * R;
        int w_in = globalid % W_in;
        int h_in = globalid / W_in;

        float temp = 0.0f;
        for (int row = 0; row < sr; row++)
        {

            int s = row % S;
            int r = row / S;

            int h_num = h_in + padh - r;
            int w_num = w_in + padw - s;

            // only valid if stride divides evenly
            if (h_num % strideh != 0 || w_num % stridew != 0)
                continue;

            int h_out = h_num / strideh;
            int w_out = w_num / stridew;

            int column = h_out * W_out + w_out;

            if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out && c < C)
                temp += d_in[(batchN * csr + (c * sr + row)) * (H_out * W_out) + column];
        }

        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in)

            d_out[(batchN * C + c) * H_in * W_in + globalid] = temp;
    }

    void *cuda_col2im_gputogpu(float *d_in, float *d_out,
                               int batchN, int C,
                               int H_in, int W_in,
                               int R, int S,
                               int pad_h, int pad_w,
                               int stride_h, int stride_w)
    {
        int output_elems = batchN * C * H_in * W_in;

        int H_out = (H_in + 2 * pad_h - R) / stride_h + 1;
        int W_out = (W_in + 2 * pad_w - S) / stride_w + 1;

        int tpb = 256;

        dim3 grid((H_in * W_in + tpb - 1) / tpb, C, batchN);
        dim3 block(tpb);

        col2im<<<grid, block>>>(
            d_out, d_in,
            C, R, S,
            H_in, W_in,
            H_out, W_out,
            stride_w, stride_h,
            pad_h, pad_w);

        cudaDeviceSynchronize();
    }

}
