#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>

namespace seera_cuda
{
    void cuda_relu_fwd(const float *x, float *out, float *grad, int size);

    void cuda_sigmoid_fwd(const float *x, float *out, float *grad, int size);

    void cuda_tanh_fwd(const float *x, float *out, float *grad, int size);

    void cuda_log_fwd(const float *x, float *out, float *grad, int size);

    void cuda_exp_fwd(const float *x, float *out, float *grad, int size);

    void cuda_abs_fwd(const float *x, float *out, float *grad, int size);

    void cuda_sqrt_fwd(const float *x, float *out, float *grad, int size);

    void cuda_pow_fwd(const float *x, float exponent, float *out, float *grad, int size);

    void cuda_clip_fwd(const float *x, float lo, float hi, float *out, float *grad, int size);

    void cuda_softmax_fwd(const float *x, float *out, int N, int C);

    void cuda_softmax_vjp(const float *s, const float *dout, float *dx, int N, int C);

    void cuda_conv2d_bwd(float *W, float *X, float *dY, float *dX, float *dW,
                         int batch, int C, int H, int W_in, int N, int R,
                         int S, int strideh, int stridew, int padh, int padw);
    void cuda_conv2d_fwd(float *h_image, float *h_kernel, float *d_conv,
                         int batchN, int C, int H, int W, int N, int R,
                         int S, int pad_h, int pad_w, int stride_h,
                         int stride_w);

    void cuda_matmul(float *hA, float *hB, float *hC, int M, int N, int K, int Nbatch);
    void cuda_matmul_bwd(float *A, float *B, float *dC, float *dA, float *dB,
                         int M, int N, int K, int Nbatch);
    void cuda_transpose_2d(float *in, float *out, int rows, int cols);
    void cuda_transpose_3d(float *in, float *out, int Nbatch, int M, int K);
    void cuda_conv2DTranpose_fwd(float *hA, float *hB, float *hC, int batch,
                                 int Cin, int Hin, int Win, int Cout, int KH,
                                 int KW, int strideh, int stridew, int padh,
                                 int padw);
    void cuda_conv2DTranspose_bwd(float *W, float *X, float *dY, float *dX,
                                  float *dW, int batch, int Cin, int Hin,
                                  int Win, int Cout, int KH, int KW,
                                  int strideh, int stridew, int padh,
                                  int padw);

    void cuda_maxpool_fwd(float *image, float *out, int *mask,
                          int batchN, int C, int H, int W,
                          int R, int S,
                          int pad_h, int pad_w,
                          int stride_h, int stride_w);

    void cuda_maxpool_bwd(float *dout, int *mask, float *dX, int batchN, int C, int H, int W,
                          int R, int S, int pad_h, int pad_w, int stride_h, int stride_w);

    void cuda_unpooling_fwd(float *d_inp, float *d_out,
                            int batchN, int C, int H, int W,
                            int sh, int sw);

    void cuda_unpooling_bwd(float *d_dout, float *d_dx,
                            int batchN, int C, int H, int W,
                            int sh, int sw);
    void cuda_sum_fwd(float *A, float *out, int ndims, int dim, int *dimarr);

    void cuda_mean_fwd(float *A, float *out, int ndims, int dim, int *dimarr);

    void cuda_max_fwd(float *A, float *out, int ndims, int dim, int *dimarr);

    void cuda_min_fwd(float *A, float *out, int ndims, int dim, int *dimarr);

    void cuda_argmax_fwd(float *A, int *out, int ndims, int dim, int *dimarr);

    void cuda_argmin_fwd(float *A, int *out, int ndims, int dim, int *dimarr);

    void cuda_sum_bwd(float *dOut, float *dA, int ndims, int dim,
                      int *dimarr);

    void cuda_mean_bwd(float *dOut, float *dA, int ndims, int dim,
                       int *dimarr);

    void cuda_max_bwd(float *dOut, float *fwdInput, float *fwdOutput,
                      float *dA, int ndims, int dim, int *dimarr);

    void cuda_min_bwd(float *dOut, float *fwdInput, float *fwdOutput,
                      float *dA, int ndims, int dim, int *dimarr);

    void cuda_scaler_multiply_h(float *arr, float k, int total_elements);
    void cuda_ones_h(float *arr, int total_elements);
    void cuda_zeros_h(float *arr, int total_elements);

    // float
    void cuda_scaler_multiply_f(float *arr, float k, int total_elements);
    void cuda_scaler_add_f(float *arr, float k, int total_elements);
    void cuda_scaler_power_f(float *arr, float k, int total_elements);

    void cuda_ones_f(float *arr, int total_elements);
    void cuda_zeros_f(float *arr, int total_elements);

    void cuda_elemadd_gputogpu(float *A, float *B, float *C, int size);

    void cuda_elemmult_gputogpu(float *A, float *B, float *C, int size);

    void cuda_elemsub_gputogpu(float *A, float *B, float *C, int size);

    void cuda_elemdiv_gputogpu(float *A, float *B, float *C, int size);
    // broadcasting

    void broadcast_add_4d(
        const float *A, const float *B, float *C,
        int aN, int aC, int aH, int aW,
        int bN, int bC, int bH, int bW);

    void broadcast_mul_4d(
        const float *A, const float *B, float *C,
        int aN, int aC, int aH, int aW,
        int bN, int bC, int bH, int bW);

}

/*
conv2d backward -done
conv transpose fwd bwd -done
maxpool mask -done
activation functions-done
returns gradient with elemops and activations-done
concatenate-done
batchnorm fwd, bwd
*/
