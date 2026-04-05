#include "seera_engine.hpp"
#include <omp.h>
#include <cmath>

namespace seera {

void relu_fwd(const float* x, float* out, float* grad, int size) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        out[i]  = x[i] > 0.0f ? x[i] : 0.0f;
        grad[i] = x[i] > 0.0f ? 1.0f : 0.0f;
    }
}

void sigmoid_fwd(const float* x, float* out, float* grad, int size) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        float s = 1.0f / (1.0f + std::exp(-x[i]));
        out[i]  = s;
        grad[i] = s * (1.0f - s);
    }
}

void tanh_fwd(const float* x, float* out, float* grad, int size) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        float t = std::tanh(x[i]);
        out[i]  = t;
        grad[i] = 1.0f - t * t;
    }
}

void log_fwd(const float* x, float* out, float* grad, int size) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        out[i]  = std::log(x[i]);
        grad[i] = 1.0f / x[i];
    }
}

void exp_fwd(const float* x, float* out, float* grad, int size) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        float e = std::exp(x[i]);
        out[i]  = e;
        grad[i] = e;
    }
}

void abs_fwd(const float* x, float* out, float* grad, int size) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        out[i]  = std::fabs(x[i]);
        grad[i] = x[i] > 0.0f ? 1.0f : (x[i] < 0.0f ? -1.0f : 0.0f);
    }
}

void sqrt_fwd(const float* x, float* out, float* grad, int size) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        float s = std::sqrt(x[i]);
        out[i]  = s;
        grad[i] = 0.5f / (s + 1e-12f);
    }
}

void pow_fwd(const float* x, float exponent, float* out, float* grad, int size) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        out[i]  = std::pow(x[i], exponent);
        grad[i] = exponent * std::pow(x[i], exponent - 1.0f);
    }
}

void clip_fwd(const float* x, float lo, float hi, float* out, float* grad, int size) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        out[i]  = std::clamp(x[i], lo, hi);
        grad[i] = (x[i] >= lo && x[i] <= hi) ? 1.0f : 0.0f;
    }
}

// ── Softmax forward (per-row) ───────────────────────────────
void softmax_fwd(const float* x, float* out, int N, int C) {
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const float* row = x + n * C;
        float* orow = out + n * C;
        // Max for numerical stability
        float mx = row[0];
        for (int c = 1; c < C; c++) mx = std::max(mx, row[c]);
        // Exp + sum
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            orow[c] = std::exp(row[c] - mx);
            sum += orow[c];
        }
        // Normalize
        float inv = 1.0f / sum;
        for (int c = 0; c < C; c++) orow[c] *= inv;
    }
}

// ── Softmax VJP backward ───────────────────────────────────
void softmax_vjp(const float* s, const float* dout, float* dx, int N, int C) {
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const float* sr = s + n * C;
        const float* dr = dout + n * C;
        float* dxr = dx + n * C;
        float dot = 0.0f;
        for (int c = 0; c < C; c++) dot += dr[c] * sr[c];
        for (int c = 0; c < C; c++) dxr[c] = sr[c] * (dr[c] - dot);
    }
}

} // namespace seera
