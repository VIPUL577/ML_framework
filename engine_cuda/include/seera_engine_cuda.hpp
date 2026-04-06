#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>

namespace seera_cuda
{
    void cuda_relu_fwd(const half *x, half *out, half *grad, int size, cudaStream_t stream = 0);

    void cuda_sigmoid_fwd(const half *x, half *out, half *grad, int size, cudaStream_t stream = 0);

    void cuda_tanh_fwd(const half *x, half *out, half *grad, int size, cudaStream_t stream = 0);

    void cuda_log_fwd(const half *x, half *out, half *grad, int size, cudaStream_t stream = 0);

    void cuda_exp_fwd(const half *x, half *out, half *grad, int size, cudaStream_t stream = 0);

    void cuda_abs_fwd(const half *x, half *out, half *grad, int size, cudaStream_t stream = 0);

    void cuda_sqrt_fwd(const half *x, half *out, half *grad, int size, cudaStream_t stream = 0);

    void cuda_pow_fwd(const half *x, float exponent, half *out, half *grad, int size, cudaStream_t stream = 0);

    void cuda_clip_fwd(const half *x, float lo, float hi, half *out, half *grad, int size, cudaStream_t stream = 0);

    void cuda_softmax_fwd(const half *x, half *out, int N, int C, cudaStream_t stream = 0);

    void cuda_softmax_vjp(const half *s, const half *dout, half *dx, int N, int C, cudaStream_t stream = 0);

    void cuda_conv2d_bwd(half *W, half *X, half *dY, half *dX, half *dW,
                         int batch, int C, int H, int W_in, int N, int R,
                         int S, int strideh, int stridew, int padh, int padw);
    void cuda_conv2d_fwd(half *h_image, half *h_kernel, half *d_conv,
                         int batchN, int C, int H, int W, int N, int R,
                         int S, int pad_h, int pad_w, int stride_h,
                         int stride_w);

    void cuda_matmul(half *hA, half *hB, half *hC, int M, int N, int K, int Nbatch);
    void cuda_matmul_bwd(half *A, half *B, half *dC, half *dA, half *dB,
                         int M, int N, int K, int Nbatch);
    void cuda_conv2DTranpose_fwd(half *hA, half *hB, half *hC, int batch,
                                 int Cin, int Hin, int Win, int Cout, int KH,
                                 int KW, int strideh, int stridew, int padh,
                                 int padw);
    void cuda_conv2DTranspose_bwd(half *W, half *X, half *dY, half *dX,
                                  half *dW, int batch, int Cin, int Hin,
                                  int Win, int Cout, int KH, int KW,
                                  int strideh, int stridew, int padh,
                                  int padw);

    void cuda_maxpool_fwd(half *image, half *out, short *mask,
                          int batchN, int C, int H, int W,
                          int R, int S,
                          int pad_h, int pad_w,
                          int stride_h, int stride_w);

    void cuda_maxpool_bwd(half *dout, short *mask, half *dX, int batchN, int C, int H, int W,
                          int R, int S, int pad_h, int pad_w, int stride_h, int stride_w);

    void cuda_unpooling_fwd(half *d_inp, half *d_out,
                            int batchN, int C, int H, int W,
                            int sh, int sw);

    void cuda_unpooling_bwd(half *d_dout, half *d_dx,
                            int batchN, int C, int H, int W,
                            int sh, int sw);
    void cuda_sum_fwd(half *A, half *out, int ndims, int dim, int *dimarr);

    void cuda_mean_fwd(half *A, half *out, int ndims, int dim, int *dimarr);

    void cuda_max_fwd(half *A, half *out, int ndims, int dim, int *dimarr);

    void cuda_min_fwd(half *A, half *out, int ndims, int dim, int *dimarr);

    void cuda_argmax_fwd(half *A, int *out, int ndims, int dim, int *dimarr);

    void cuda_argmin_fwd(half *A, int *out, int ndims, int dim, int *dimarr);

    void cuda_sum_bwd(half *dOut, half *dA, int ndims, int dim,
                      int *dimarr);

    void cuda_mean_bwd(half *dOut, half *dA, int ndims, int dim,
                       int *dimarr);

    void cuda_max_bwd(half *dOut, half *fwdInput, half *fwdOutput,
                      half *dA, int ndims, int dim, int *dimarr);

    void cuda_min_bwd(half *dOut, half *fwdInput, half *fwdOutput,
                      half *dA, int ndims, int dim, int *dimarr);

    float *to_device(float *cpu_arr, int size);
    float *to_host(float *gpu_arr, int size);
    half *to_device(half *cpu_arr, int size);
    half *to_host(half *gpu_arr, int size);

}

/*
conv2d backward -done
conv transpose fwd bwd -done
maxpool mask -done
activation functions-done
returns gradient with elemops and activations-done
concatenate
batchnorm fwd, bwd
*/
