#include "seera_engine.hpp"
#include <cblas.h>
#include <omp.h>

namespace seera {

// ── Matmul via OpenBLAS ─────────────────────────────────────
// C(M,N) = A(M,K) @ B(K,N)
void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, A, K,   // alpha=1, lda=K
                      B, N,   // ldb=N
                0.0f, C, N);  // beta=0, ldc=N
}

// ── Element-wise add (flat, OpenMP) ─────────────────────────
void add_arrays(const float* a, const float* b, float* out, int size) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        out[i] = a[i] + b[i];
    }
}

// ── Element-wise mul (flat, OpenMP) ─────────────────────────
void mul_arrays(const float* a, const float* b, float* out, int size) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        out[i] = a[i] * b[i];
    }
}

} // namespace seera
