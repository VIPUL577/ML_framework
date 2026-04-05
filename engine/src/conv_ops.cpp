#include "seera_engine.hpp"
#include <cblas.h>
#include <omp.h>
#include <cstring>

namespace seera {

// ── im2col (batched, OpenMP over N) ─────────────────────────
void im2col_batch(const float* X, float* col,
                  int N, int C, int H, int W,
                  int KH, int KW, int stride, int pad) {
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;
    int col_row = C * KH * KW;
    int col_col = OH * OW;

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const float* xn = X + n * C * H * W;
        float* cn = col + n * col_row * col_col;

        for (int c = 0; c < C; c++) {
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    int row = c * KH * KW + kh * KW + kw;
                    for (int oh = 0; oh < OH; oh++) {
                        int h_idx = oh * stride + kh - pad;
                        for (int ow = 0; ow < OW; ow++) {
                            int w_idx = ow * stride + kw - pad;
                            int ci = oh * OW + ow;
                            if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W)
                                cn[row * col_col + ci] = xn[c * H * W + h_idx * W + w_idx];
                            else
                                cn[row * col_col + ci] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

// ── col2im (batched, OpenMP over N) ─────────────────────────
void col2im_batch(const float* col, float* X,
                  int N, int C, int H, int W,
                  int KH, int KW, int stride, int pad) {
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;
    int col_row = C * KH * KW;
    int col_col = OH * OW;

    std::memset(X, 0, N * C * H * W * sizeof(float));

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const float* cn = col + n * col_row * col_col;
        float* xn = X + n * C * H * W;

        for (int c = 0; c < C; c++) {
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    int row = c * KH * KW + kh * KW + kw;
                    for (int oh = 0; oh < OH; oh++) {
                        int h_idx = oh * stride + kh - pad;
                        for (int ow = 0; ow < OW; ow++) {
                            int w_idx = ow * stride + kw - pad;
                            int ci = oh * OW + ow;
                            if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W)
                                xn[c * H * W + h_idx * W + w_idx] += cn[row * col_col + ci];
                        }
                    }
                }
            }
        }
    }
}

// ── Conv2D Forward ──────────────────────────────────────────
// X(N,C,H,Win), W(F,C,KH,KW) → out(N,F,OH,OW)
void conv2d_forward(const float* X, const float* W, float* out,
                    int N, int C, int H, int Win,
                    int F, int KH, int KW, int stride, int pad) {
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (Win + 2 * pad - KW) / stride + 1;
    int col_row = C * KH * KW;
    int col_col = OH * OW;

    // Allocate im2col buffer
    float* col = new float[N * col_row * col_col];
    im2col_batch(X, col, N, C, H, Win, KH, KW, stride, pad);

    // For each sample: out[n] = W_flat(F, col_row) @ col[n](col_row, col_col) → (F, col_col)
    #pragma omp parallel for schedule(static) // -> for batch in parfallel
    for (int n = 0; n < N; n++) {
        const float* cn = col + n * col_row * col_col;
        float* on = out + n * F * OH * OW;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    F, col_col, col_row,
                    1.0f, W, col_row,       // W_flat(F, C*KH*KW)
                          cn, col_col,      // col(C*KH*KW, OH*OW)
                    0.0f, on, col_col);     // out(F, OH*OW)
    }

    delete[] col;
}

// ── Conv2D Backward ─────────────────────────────────────────
// dout(N,F,OH,OW), X(N,C,H,Win), W(F,C,KH,KW)
// → dX(N,C,H,Win), dW(F,C,KH,KW)
void conv2d_backward(const float* dout, const float* X, const float* W,
                     float* dX, float* dW,
                     int N, int C, int H, int Win,
                     int F, int KH, int KW, int stride, int pad) {
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (Win + 2 * pad - KW) / stride + 1;
    int col_row = C * KH * KW;
    int col_col = OH * OW;

    float* col = new float[N * col_row * col_col];
    im2col_batch(X, col, N, C, H, Win, KH, KW, stride, pad);

    // dW = sum_n( dout_n(F, OH*OW) @ col_n.T(OH*OW, C*KH*KW) )
    std::memset(dW, 0, F * C * KH * KW * sizeof(float));

    // Per-sample dW contributions (accumulate with mutex or per-thread buffers)
    float* dW_local = new float[N * F * col_row];
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const float* dn = dout + n * F * col_col;
        const float* cn = col  + n * col_row * col_col;
        float* dwn = dW_local + n * F * col_row;
        // dwn = dout_n @ col_n.T  → (F, col_row)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    F, col_row, col_col,
                    1.0f, dn, col_col,
                          cn, col_col,
                    0.0f, dwn, col_row);
    }
    // Sum dW contributions
    for (int n = 0; n < N; n++) {
        const float* dwn = dW_local + n * F * col_row;
        for (int i = 0; i < F * col_row; i++) dW[i] += dwn[i];
    }
    delete[] dW_local;

    // dX via col2im
    float* dX_col = new float[N * col_row * col_col];
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const float* dn = dout + n * F * col_col;
        float* dcn = dX_col + n * col_row * col_col;
        // dcn = W.T(col_row, F) @ dout_n(F, col_col) → (col_row, col_col)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    col_row, col_col, F,
                    1.0f, W, col_row,       // W(F, col_row) transposed
                          dn, col_col,
                    0.0f, dcn, col_col);
    }
    col2im_batch(dX_col, dX, N, C, H, Win, KH, KW, stride, pad);

    delete[] col;
    delete[] dX_col;
}

// ── ConvTranspose2D Forward ─────────────────────────────────
// X(N,Cin,H,Win), W(Cin,Cout,KH,KW) → out(N,Cout,Hout,Wout)
// Hout = (H-1)*stride - 2*pad + KH
// Wout = (Win-1)*stride - 2*pad + KW
//
// For each sample n:
//   X_flat = X[n].reshape(Cin, H*Win)
//   W_flat = W.reshape(Cin, Cout*KH*KW)
//   col = W_flat.T @ X_flat   → (Cout*KH*KW, H*Win)
//   out[n] = col2im(col, ...)
void conv_transpose2d_forward(const float* X, const float* W, float* out,
                              int N, int Cin, int H, int Win,
                              int Cout, int KH, int KW,
                              int stride, int pad) {
    int Hout = (H - 1) * stride - 2 * pad + KH;
    int Wout = (Win - 1) * stride - 2 * pad + KW;
    int col_row = Cout * KH * KW;     // rows of col matrix
    int spatial_in = H * Win;          // columns of col matrix (input spatial)

    // Allocate col buffer for all samples
    float* col = new float[N * col_row * spatial_in];

    // GEMM: col[n] = W_flat.T(col_row, Cin) @ X_flat[n](Cin, spatial_in)
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const float* xn = X + n * Cin * spatial_in;
        float* cn = col + n * col_row * spatial_in;
        // W is (Cin, Cout*KH*KW) in memory = W_flat
        // We need W_flat.T @ X_flat = (Cout*KH*KW, Cin) @ (Cin, spatial_in)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    col_row, spatial_in, Cin,
                    1.0f, W, col_row,          // W(Cin, col_row) transposed
                          xn, spatial_in,       // X_flat(Cin, spatial_in)
                    0.0f, cn, spatial_in);      // col(col_row, spatial_in)
    }

    // col2im: scatter col into output (N, Cout, Hout, Wout)
    // NOTE: col2im expects col as (Cout*KH*KW, OH*OW) where the output
    // of col2im has shape (N, Cout, Hout, Wout). Here the "image" being
    // reconstructed is the output, and the "patches" come from the input spatial locs.
    // We need to use col2im with the OUTPUT dimensions as the "image" dims.
    // The col matrix has shape (Cout*KH*KW, H*Win) where H*Win are the
    // number of spatial positions in the input (which correspond to the
    // "patch locations" in col2im with stride and pad).
    col2im_batch(col, out, N, Cout, Hout, Wout, KH, KW, stride, pad);

    delete[] col;
}

// ── ConvTranspose2D Backward ────────────────────────────────
// dout(N,Cout,Hout,Wout), X(N,Cin,H,Win), W(Cin,Cout,KH,KW)
// → dX(N,Cin,H,Win), dW(Cin,Cout,KH,KW)
void conv_transpose2d_backward(const float* dout, const float* X, const float* W,
                               float* dX, float* dW,
                               int N, int Cin, int H, int Win,
                               int Cout, int KH, int KW,
                               int stride, int pad) {
    int Hout = (H - 1) * stride - 2 * pad + KH;
    int Wout = (Win - 1) * stride - 2 * pad + KW;
    int col_row = Cout * KH * KW;
    int spatial_in = H * Win;

    // im2col on dout: extract patches from dout
    // This gives us col_dout(N, Cout*KH*KW, H*Win)
    // where H*Win = number of output patches using (Hout, Wout) as image with KH,KW,stride,pad
    float* col_dout = new float[N * col_row * spatial_in];
    im2col_batch(dout, col_dout, N, Cout, Hout, Wout, KH, KW, stride, pad);

    // ─── dW = sum_n( X_flat[n] @ col_dout[n].T ) ───
    // X_flat[n]: (Cin, spatial_in),  col_dout[n]: (col_row, spatial_in)
    // dW_n = X_flat @ col_dout.T → (Cin, col_row) = (Cin, Cout*KH*KW)
    std::memset(dW, 0, Cin * col_row * sizeof(float));
    float* dW_local = new float[N * Cin * col_row];

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const float* xn = X + n * Cin * spatial_in;
        const float* cn = col_dout + n * col_row * spatial_in;
        float* dwn = dW_local + n * Cin * col_row;
        // dwn = X_flat(Cin, spatial_in) @ col_dout.T(spatial_in, col_row)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    Cin, col_row, spatial_in,
                    1.0f, xn, spatial_in,
                          cn, spatial_in,
                    0.0f, dwn, col_row);
    }
    // Sum dW contributions across batch
    for (int n = 0; n < N; n++) {
        const float* dwn = dW_local + n * Cin * col_row;
        for (int i = 0; i < Cin * col_row; i++) dW[i] += dwn[i];
    }
    delete[] dW_local;

    // ─── dX: dX_flat[n] = W_flat @ col_dout[n] ───
    // W_flat: (Cin, col_row),  col_dout[n]: (col_row, spatial_in)
    // dX_flat = (Cin, spatial_in)
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const float* cn = col_dout + n * col_row * spatial_in;
        float* dxn = dX + n * Cin * spatial_in;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Cin, spatial_in, col_row,
                    1.0f, W, col_row,
                          cn, spatial_in,
                    0.0f, dxn, spatial_in);
    }

    delete[] col_dout;
}

} // namespace seera
