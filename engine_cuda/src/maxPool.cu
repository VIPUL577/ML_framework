#include <cuda.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>
namespace seera_cuda {

__global__ void maxPool(float *input_image, float *conv, int *mask, int C,
                        int H, int W, int R, int S, int pad_h, int pad_w,
                        int stride_h, int stride_w, int H_out, int W_out) {
  int total_elements = H_out * W_out * C;
  int batchN = blockIdx.y;
  int index_ = blockIdx.x * blockDim.x + threadIdx.x;

  float temp = 0.0f;
  if (index_ < total_elements) {
    int w_out = index_ % W_out;
    int h_out = (index_ / W_out) % H_out;
    int ni = index_ / (H_out * W_out);
    int ht = h_out * stride_h - pad_h;
    int wt = w_out * stride_w - pad_w;
    if (ht >= 0 && ht < H && wt >= 0 && wt < W && ni < C) {
      int input_idx = ((batchN * C + ni) * H + ht) * W + wt;
      temp = input_image[input_idx];
    }
    int rmax = 0;
    int smax = 0;
    for (int ir = 0; ir < R; ir++) {
      for (int is = 0; is < S; is++) {
        int h_in = h_out * stride_h - pad_h + ir;
        int w_in = w_out * stride_w - pad_w + is;

        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W && ni < C) {
          int input_idx = ((batchN * C + ni) * H + h_in) * W + w_in;
          float ing = input_image[input_idx];
          if (temp <= ing) {
            temp = ing;
            rmax = ir;
            smax = is;
          } else {
            mask[input_idx] = 0;
          }
        }
      }
    }
    int h_in = h_out * stride_h - pad_h + rmax;
    int w_in = w_out * stride_w - pad_w + smax;
    int input_idx = ((batchN * C + ni) * H + h_in) * W + w_in;
    mask[input_idx] = 1;
    conv[(batchN * C + ni) * H_out * W_out + h_out * W_out + w_out] = temp;
  }
}
__global__ void maxPool_bwd(float *dX, float *dout, int *mask, int C, int H,
                            int W, int R, int S, int pad_h, int pad_w,
                            int stride_h, int stride_w, int W_out, int H_out) {
  int c = blockIdx.y;
  int BatchN = blockIdx.z;
  int index_ = blockIdx.x * blockDim.x + threadIdx.x;

  int rs = R * S;

  int h_out = index_ / W_out;
  int w_out = index_ % W_out;
  for (int ii = 0; ii < rs; ii++) {
    int s = ii % S;
    int r = ii / S;
    int h_in = h_out * stride_h - pad_h + r;
    int w_in = w_out * stride_w - pad_w + s;

    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
      int index = (BatchN * C + c) * H * W + h_in * W + w_in;
      atomicAdd(&(dX[index]), dout[(BatchN * C + c) * H_out * W_out + index_] *
                                  ((float)mask[index]));
    }
  }
}

void cuda_maxpool_fwd(float *image, float *out, int *mask, int batchN, int C,
                      int H, int W, int R, int S, int pad_h, int pad_w,
                      int stride_h, int stride_w) {
  int H_out = (H + 2 * pad_h - R) / stride_h + 1;
  int W_out = (W + 2 * pad_w - S) / stride_w + 1;

  int output_elems = batchN * C * H_out * W_out;
  int output_elems_wob = C * H_out * W_out;

  // Assume 'out' is already cudaMalloc'ed by caller
  cudaMemset(out, 0.0, sizeof(float) * output_elems);
  cudaMemset(mask, 0, sizeof(int) * H * W * C * batchN);

  int threads_per_block = 256;
  int Nblock = (output_elems_wob + threads_per_block - 1) / threads_per_block;

  dim3 blocks(Nblock, batchN);

  maxPool<<<blocks, threads_per_block>>>(image, out, mask, C, H, W, R, S, pad_h,
                                         pad_w, stride_h, stride_w, H_out,
                                         W_out);

  cudaDeviceSynchronize();
}

void cuda_maxpool_bwd(float *dout, // [N,C,H_out,W_out]
                      int *mask,   // [N,C,H,W]
                      float *dX,   // [N,C,H,W] (output)
                      int batchN, int C, int H, int W, int R, int S, int pad_h,
                      int pad_w, int stride_h, int stride_w) {
  int H_out = (H + 2 * pad_h - R) / stride_h + 1;
  int W_out = (W + 2 * pad_w - S) / stride_w + 1;

  int input_elems = batchN * C * H * W;
  int output_elems_wob = H_out * W_out;

  // Assume dX already allocated
  cudaMemset(dX, 0, sizeof(float) * input_elems);

  int threads_per_block = 256;
  int Nblock = (output_elems_wob + threads_per_block - 1) / threads_per_block;

  // x = spatial, y = channels, z = batch
  dim3 blocks(Nblock, C, batchN);

  maxPool_bwd<<<blocks, threads_per_block>>>(dX, dout, mask, C, H, W, R, S,
                                             pad_h, pad_w, stride_h, stride_w,
                                             W_out, H_out);

  cudaDeviceSynchronize();
}
} // namespace seera_cuda