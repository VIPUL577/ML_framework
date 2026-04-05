#include "seera_engine.hpp"
#include <omp.h>
#include <cstring>
#include <cfloat>

namespace seera {

// ── MaxPool2D Forward ───────────────────────────────────────
// X(N,C,H,W) → out(N,C,OH,OW), mask(N,C,OH,OW) [argmax indices]
void maxpool2d_forward(const float* X, float* out, int32_t* mask,
                       int N, int C, int H, int W,
                       int KH, int KW, int stride, int pad) {
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float mx = -FLT_MAX;
                    int mx_idx = 0;
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            int h_idx = oh * stride + kh - pad;
                            int w_idx = ow * stride + kw - pad;
                            float val = 0.0f;
                            if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W)
                                val = X[n*C*H*W + c*H*W + h_idx*W + w_idx];
                            if (val > mx) {
                                mx = val;
                                mx_idx = kh * KW + kw;
                            }
                        }
                    }
                    int out_idx = n*C*OH*OW + c*OH*OW + oh*OW + ow;
                    out[out_idx] = mx;
                    mask[out_idx] = mx_idx;
                }
            }
        }
    }
}

// ── MaxPool2D Backward ──────────────────────────────────────
void maxpool2d_backward(const float* dout, const int32_t* mask, float* dX,
                        int N, int C, int H, int W,
                        int OH, int OW, int KH, int KW,
                        int stride, int pad) {
    std::memset(dX, 0, N * C * H * W * sizeof(float));

    // Cannot parallelize with collapse since multiple windows may write to same dX location
    #pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    int out_idx = n*C*OH*OW + c*OH*OW + oh*OW + ow;
                    int mx_idx = mask[out_idx];
                    int kh = mx_idx / KW;
                    int kw = mx_idx % KW;
                    int h_idx = oh * stride + kh - pad;
                    int w_idx = ow * stride + kw - pad;
                    if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
                        // Atomic add for thread safety
                        #pragma omp atomic
                        dX[n*C*H*W + c*H*W + h_idx*W + w_idx] += dout[out_idx];
                    }
                }
            }
        }
    }
}

// ── Unpooling Nearest Forward ───────────────────────────────
void unpooling_fwd(const float* x, float* out,
                  int N, int C, int H, int W, int sh, int sw) {
    int Ho = H * sh, Wo = W * sw;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < Ho; h++) {
                for (int w = 0; w < Wo; w++) {
                    out[n*C*Ho*Wo + c*Ho*Wo + h*Wo + w] =
                        x[n*C*H*W + c*H*W + (h/sh)*W + (w/sw)];
                }
            }
        }
    }
}

// ── Unpooling Nearest Backward ──────────────────────────────
void unpooling_bwd(const float* dout, float* dx,
                  int N, int C, int H, int W, int sh, int sw) {
    int Ho = H * sh, Wo = W * sw;
    std::memset(dx, 0, N * C * H * W * sizeof(float));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    float sum = 0.0f;
                    for (int i = h*sh; i < (h+1)*sh; i++)
                        for (int j = w*sw; j < (w+1)*sw; j++)
                            sum += dout[n*C*Ho*Wo + c*Ho*Wo + i*Wo + j];
                    dx[n*C*H*W + c*H*W + h*W + w] = sum;
                }
            }
        }
    }
}

} // namespace seera
