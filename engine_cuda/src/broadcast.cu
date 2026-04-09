#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 256

/* ─────────────────────────────────────────────────────────────
   Kernel
   ───────────────────────────────────────────────────────────── */
namespace seera_cuda{
__global__ void broadcast_kernel_4d(
    const float*  A,
    const float*  B,
    float*  C,

    int N, int Cc, int H, int W,

    int aN, int aC, int aH, int aW,
    int bN, int bC, int bH, int bW,

    long long total,
    int op
)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    // Decode tid → (n, c, h, w)
    int w = tid % W;
    int h = (tid / W) % H;
    int c = (tid / (W * H)) % Cc;
    int n = tid / (W * H * Cc);

    // Broadcast mapping for A
    int an = (aN == 1) ? 0 : n;
    int ac = (aC == 1) ? 0 : c;
    int ah = (aH == 1) ? 0 : h;
    int aw = (aW == 1) ? 0 : w;

    // Broadcast mapping for B
    int bn = (bN == 1) ? 0 : n;
    int bc = (bC == 1) ? 0 : c;
    int bh = (bH == 1) ? 0 : h;
    int bw = (bW == 1) ? 0 : w;

    // Flatten index for A
    long long a_idx =
        ((long long)an * aC * aH * aW) +
        ((long long)ac * aH * aW) +
        ((long long)ah * aW) +
        aw;

    // Flatten index for B
    long long b_idx =
        ((long long)bn * bC * bH * bW) +
        ((long long)bc * bH * bW) +
        ((long long)bh * bW) +
        bw;

    float av = A[a_idx];
    float bv = B[b_idx];

    C[tid] = (op == 0) ? (av + bv) : (av * bv);
}

/* ─────────────────────────────────────────────────────────────
   Host: Compute output shape
   ───────────────────────────────────────────────────────────── */

static bool compute_out_shape_4d(
    int aN, int aC, int aH, int aW,
    int bN, int bC, int bH, int bW,
    int &oN, int &oC, int &oH, int &oW)
{
    auto resolve = [](int a, int b) -> int {
        if (a == b) return a;
        if (a == 1) return b;
        if (b == 1) return a;
        return -1;
    };

    oN = resolve(aN, bN);
    oC = resolve(aC, bC);
    oH = resolve(aH, bH);
    oW = resolve(aW, bW);

    return (oN != -1 && oC != -1 && oH != -1 && oW != -1);
}

/* ─────────────────────────────────────────────────────────────
   Main API
   ───────────────────────────────────────────────────────────── */

void broadcast_op_4d(
    const float* A, 
    const float* B, 
    float* C,
    int aN, int aC, int aH, int aW,
    int bN, int bC, int bH, int bW,
    int op                    // 0 = add, 1 = multiply
)
{
    // Compute output shape
    int oN, oC, oH, oW;
    if (!compute_out_shape_4d(
            aN, aC, aH, aW,
            bN, bC, bH, bW,
            oN, oC, oH, oW))
    {
        fprintf(stderr, "Broadcast shape mismatch!\n");
        return;
    }

    long long total = (long long)oN * oC * oH * oW;
    if (total == 0) return;

    // Launch kernel
    int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    broadcast_kernel_4d<<<grid, BLOCK_SIZE>>>(
        A, B, C,
        oN, oC, oH, oW,
        aN, aC, aH, aW,
        bN, bC, bH, bW,
        total,
        op
    );

    cudaDeviceSynchronize(); // remove later for performance
}

/* ─────────────────────────────────────────────────────────────
   Convenience wrappers
   ───────────────────────────────────────────────────────────── */
/* ─────────────────────────────────────────────────────────────
   Compute total output size (number of elements)
   Returns -1 if shapes are incompatible for broadcasting.
   ───────────────────────────────────────────────────────────── */
void broadcast_add_4d(
    const float* A, const float* B, float* C,
    int aN, int aC, int aH, int aW,
    int bN, int bC, int bH, int bW)
{
    broadcast_op_4d(A, B, C, aN, aC, aH, aW, bN, bC, bH, bW, 0);
}

void broadcast_mul_4d(
    const float* A, const float* B, float* C,
    int aN, int aC, int aH, int aW,
    int bN, int bC, int bH, int bW)
{
    broadcast_op_4d(A, B, C, aN, aC, aH, aW, bN, bC, bH, bW, 1);
}}