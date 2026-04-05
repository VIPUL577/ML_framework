#pragma once
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace seera {

// ── Matmul (OpenBLAS cblas_sgemm) ──────────────────────────
// C = A(M,K) @ B(K,N)
void matmul(const float* A, const float* B, float* C, int M, int K, int N);  //-> GPU supported!!

// ── Element-wise (OpenMP) ──────────────────────────────────
void add_arrays(const float* a, const float* b, float* out, int size);     //-> GPU supported!!
void mul_arrays(const float* a, const float* b, float* out, int size);//-> GPU supported!!

// ── Activations (OpenMP) — each writes out + gradient ──────
void relu_fwd(const float* x, float* out, float* grad, int size);//-> GPU supported!! - figured out just implementation required
void sigmoid_fwd(const float* x, float* out, float* grad, int size);//-> GPU supported!! - figured out just implementation required
void tanh_fwd(const float* x, float* out, float* grad, int size);//-> GPU supported!! - figured out just implementation required
void log_fwd(const float* x, float* out, float* grad, int size);//-> GPU supported!! - figured out just implementation required
void exp_fwd(const float* x, float* out, float* grad, int size);//-> GPU supported!! - figured out just implementation required
void abs_fwd(const float* x, float* out, float* grad, int size);//-> GPU supported!! - figured out just implementation required
void sqrt_fwd(const float* x, float* out, float* grad, int size);//-> GPU supported!! - figured out just implementation required
void pow_fwd(const float* x, float exponent, float* out, float* grad, int size);//-> GPU supported!! - figured out just implementation required
void clip_fwd(const float* x, float lo, float hi, float* out, float* grad, int size);//-> GPU supported!! - figured out just implementation required

// ── Softmax (per-row, OpenMP over rows) ────────────────────
void softmax_fwd(const float* x, float* out, int N, int C);
void softmax_vjp(const float* s, const float* dout, float* dx, int N, int C);

// ── Conv2D ─────────────────────────────────────────────────
void im2col_batch(const float* X, float* col,
                  int N, int C, int H, int W,
                  int KH, int KW, int stride, int pad);

void col2im_batch(const float* col, float* X,
                  int N, int C, int H, int W,
                  int KH, int KW, int stride, int pad);

void conv2d_forward(const float* X, const float* W, float* out,
                    int N, int C, int H, int Win,
                    int F, int KH, int KW, int stride, int pad);

void conv2d_backward(const float* dout, const float* X, const float* W,
                     float* dX, float* dW,
                     int N, int C, int H, int Win,
                     int F, int KH, int KW, int stride, int pad);

// ── MaxPool2D ──────────────────────────────────────────────
void maxpool2d_forward(const float* X, float* out, int32_t* mask,
                       int N, int C, int H, int W,
                       int KH, int KW, int stride, int pad);

void maxpool2d_backward(const float* dout, const int32_t* mask, float* dX,
                        int N, int C, int H, int W,
                        int OH, int OW, int KH, int KW,
                        int stride, int pad);

// ── Upsample Nearest ──────────────────────────────────────
void upsample_fwd(const float* x, float* out,
                  int N, int C, int H, int W, int sh, int sw);
void upsample_bwd(const float* dout, float* dx,
                  int N, int C, int H, int W, int sh, int sw);

// ── BatchNorm ──────────────────────────────────────────────
// 1D: input (N, C).   2D: input (N, C, H, W).
// Writes: out, x_hat, channel_mean(C), channel_std_inv(C)
// Updates running_mean/var in-place when training=true
void batchnorm_forward(const float* X, const float* gamma, const float* beta,
                       float* running_mean, float* running_var,
                       float* out, float* x_hat,
                       float* mean_out, float* std_inv_out,
                       int N, int C, int H, int W,
                       float momentum, float eps, bool training, bool is_2d);

void batchnorm_backward(const float* dout, const float* x_hat,
                        const float* std_inv, const float* gamma,
                        float* dx, float* dgamma, float* dbeta,
                        int N, int C, int H, int W,
                        int M, bool is_2d);

} // namespace seera
