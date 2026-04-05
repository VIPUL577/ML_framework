#include "seera_engine.hpp"
#include <omp.h>
#include <cstring>
#include <cmath>

namespace seera {

// ── BatchNorm Forward ───────────────────────────────────────
// Handles both 1D (H=W=1) and 2D modes via the is_2d flag.
// For 1D: X is (N, C),       normalize over N  per feature
// For 2D: X is (N, C, H, W), normalize over N,H,W per channel
void batchnorm_forward(const float* X, const float* gamma, const float* beta,
                       float* running_mean, float* running_var,
                       float* out, float* x_hat,
                       float* mean_out, float* std_inv_out,
                       int N, int C, int H, int W,
                       float momentum, float eps, bool training, bool is_2d) {
    // M = number of elements per channel being averaged
    int M = is_2d ? N * H * W : N;
    int spatial = is_2d ? H * W : 1;

    if (training) {
        // Compute per-channel mean
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int n = 0; n < N; n++) {
                for (int s = 0; s < spatial; s++) {
                    sum += X[n * C * spatial + c * spatial + s];
                }
            }
            mean_out[c] = sum / M;
        }

        // Compute per-channel variance
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            float mu = mean_out[c];
            for (int n = 0; n < N; n++) {
                for (int s = 0; s < spatial; s++) {
                    float diff = X[n * C * spatial + c * spatial + s] - mu;
                    sum += diff * diff;
                }
            }
            float var = sum / M;
            std_inv_out[c] = 1.0f / std::sqrt(var + eps);
            // Update running stats
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mu;
            running_var[c]  = (1.0f - momentum) * running_var[c]  + momentum * var;
        }
    } else {
        // Inference: use running stats
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < C; c++) {
            mean_out[c] = running_mean[c];
            std_inv_out[c] = 1.0f / std::sqrt(running_var[c] + eps);
        }
    }

    // Normalize + scale + shift
    #pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float mu = mean_out[c];
            float si = std_inv_out[c];
            float g  = gamma[c];
            float b  = beta[c];
            for (int s = 0; s < spatial; s++) {
                int idx = n * C * spatial + c * spatial + s;
                float xh = (X[idx] - mu) * si;
                x_hat[idx] = xh;
                out[idx] = g * xh + b;
            }
        }
    }
}

// ── BatchNorm Backward ──────────────────────────────────────
// Efficient fused formula.
// dout, x_hat: same shape as input (N*C*spatial)
// std_inv: (C,)   gamma: (C,)
// dx: same shape   dgamma, dbeta: (C,)
void batchnorm_backward(const float* dout, const float* x_hat,
                        const float* std_inv, const float* gamma,
                        float* dx, float* dgamma, float* dbeta,
                        int N, int C, int H, int W,
                        int M, bool is_2d) {
    int spatial = is_2d ? H * W : 1;

    // dgamma, dbeta: sum over batch+spatial
    std::memset(dgamma, 0, C * sizeof(float));
    std::memset(dbeta, 0, C * sizeof(float));

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < C; c++) {
        float dg = 0.0f, db = 0.0f;
        for (int n = 0; n < N; n++) {
            for (int s = 0; s < spatial; s++) {
                int idx = n * C * spatial + c * spatial + s;
                dg += dout[idx] * x_hat[idx];
                db += dout[idx];
            }
        }
        dgamma[c] = dg;
        dbeta[c] = db;
    }

    // dx: fused formula
    // dx = (1/M) * std_inv * (M * dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat))
    // where dx_hat = dout * gamma
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < C; c++) {
        float g  = gamma[c];
        float si = std_inv[c];

        // Compute sum(dx_hat) and sum(dx_hat * x_hat)
        float sum_dxh = 0.0f, sum_dxh_xh = 0.0f;
        for (int n = 0; n < N; n++) {
            for (int s = 0; s < spatial; s++) {
                int idx = n * C * spatial + c * spatial + s;
                float dxh = dout[idx] * g;
                sum_dxh += dxh;
                sum_dxh_xh += dxh * x_hat[idx];
            }
        }

        // Compute dx
        float inv_M = 1.0f / M;
        for (int n = 0; n < N; n++) {
            for (int s = 0; s < spatial; s++) {
                int idx = n * C * spatial + c * spatial + s;
                float dxh = dout[idx] * g;
                dx[idx] = inv_M * si * (M * dxh - sum_dxh - x_hat[idx] * sum_dxh_xh);
            }
        }
    }
}

} // namespace seera
