#include "seera_engine_cuda.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>

#define THREADS_PER_BLOCK 256

namespace seera_cuda
{
__global__ void cuda_relu_fwd(const half* x, half* out, half* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = __half2float(x[i]);
        out[i]  = __float2half(val > 0.0f ? val : 0.0f);
        grad[i] = __float2half(val > 0.0f ? 1.0f : 0.0f);
    }
}

__global__ void cuda_sigmoid_fwd(const half* x, half* out, half* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = __half2float(x[i]);
        float s = 1.0f / (1.0f + expf(-val));
        out[i]  = __float2half(s);
        grad[i] = __float2half(s * (1.0f - s));
    }
}

__global__ void cuda_tanh_fwd(const half* x, half* out, half* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = __half2float(x[i]);
        float t = tanhf(val);
        out[i]  = __float2half(t);
        grad[i] = __float2half(1.0f - t * t);
    }
}

__global__ void cuda_log_fwd(const half* x, half* out, half* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = __half2float(x[i]);
        out[i]  = __float2half(logf(val));
        grad[i] = __float2half(1.0f / val);
    }
}

__global__ void cuda_exp_fwd(const half* x, half* out, half* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = __half2float(x[i]);
        float e = expf(val);
        out[i]  = __float2half(e);
        grad[i] = __float2half(e);
    }
}

__global__ void cuda_abs_fwd(const half* x, half* out, half* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = __half2float(x[i]);
        out[i]  = __float2half(fabsf(val));
        grad[i] = __float2half(val > 0.0f ? 1.0f : (val < 0.0f ? -1.0f : 0.0f));
    }
}

__global__ void cuda_sqrt_fwd(const half* x, half* out, half* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = __half2float(x[i]);
        float s = sqrtf(val);
        out[i]  = __float2half(s);
        grad[i] = __float2half(0.5f / (s + 1e-12f));
    }
}

__global__ void cuda_pow_fwd(const half* x, float exponent, half* out, half* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = __half2float(x[i]);
        out[i]  = __float2half(powf(val, exponent));
        grad[i] = __float2half(exponent * powf(val, exponent - 1.0f));
    }
}

__global__ void cuda_clip_fwd(const half* x, float lo, float hi, half* out, half* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = __half2float(x[i]);
        out[i]  = __float2half(fminf(fmaxf(val, lo), hi));
        grad[i] = __float2half((val >= lo && val <= hi) ? 1.0f : 0.0f);
    }
}



__global__ void cuda_softmax_fwd(const half* x, half* out, int N, int C) {
    extern __shared__ float smem[]; // Size: blockDim.x
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    const half* x_row = x + row * C;
    half* out_row = out + row * C;

    float thread_max = -CUDART_INF_F;
    for (int i = tid; i < C; i += blockDim.x) {
        thread_max = fmaxf(thread_max, __half2float(x_row[i]));
    }
    smem[tid] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    float row_max = smem[0];

    float thread_sum = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        thread_sum += expf(__half2float(x_row[i]) - row_max);
    }
    smem[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float row_sum = smem[0];
    float inv_sum = 1.0f / row_sum;

    for (int i = tid; i < C; i += blockDim.x) {
        float val = expf(__half2float(x_row[i]) - row_max) * inv_sum;
        out_row[i] = __float2half(val);
    }
}

__global__ void cuda_softmax_vjp(const half* s, const half* dout, half* dx, int N, int C) {
    extern __shared__ float smem[]; // Size: blockDim.x
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    const half* s_row = s + row * C;
    const half* d_row = dout + row * C;
    half* dx_row = dx + row * C;
    float thread_dot = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        thread_dot += __half2float(s_row[i]) * __half2float(d_row[i]);
    }
    smem[tid] = thread_dot;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float row_dot = smem[0];

    for (int i = tid; i < C; i += blockDim.x) {
        float s_val = __half2float(s_row[i]);
        float d_val = __half2float(d_row[i]);
        float dx_val = s_val * (d_val - row_dot);
        dx_row[i] = __float2half(dx_val);
    }
}


inline int get_blocks(int size) {
    return (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

void cuda_relu_fwd(const half* x, half* out, half* grad, int size, cudaStream_t stream = 0) {
    relu_fwd<<<get_blocks(size), THREADS_PER_BLOCK, 0, stream>>>(x, out, grad, size);
}

void cuda_sigmoid_fwd(const half* x, half* out, half* grad, int size, cudaStream_t stream = 0) {
    sigmoid_fwd<<<get_blocks(size), THREADS_PER_BLOCK, 0, stream>>>(x, out, grad, size);
}

void cuda_tanh_fwd(const half* x, half* out, half* grad, int size, cudaStream_t stream = 0) {
    tanh_fwd<<<get_blocks(size), THREADS_PER_BLOCK, 0, stream>>>(x, out, grad, size);
}

void cuda_log_fwd(const half* x, half* out, half* grad, int size, cudaStream_t stream = 0) {
    log_fwd<<<get_blocks(size), THREADS_PER_BLOCK, 0, stream>>>(x, out, grad, size);
}

void cuda_exp_fwd(const half* x, half* out, half* grad, int size, cudaStream_t stream = 0) {
    exp_fwd<<<get_blocks(size), THREADS_PER_BLOCK, 0, stream>>>(x, out, grad, size);
}

void cuda_abs_fwd(const half* x, half* out, half* grad, int size, cudaStream_t stream = 0) {
    abs_fwd<<<get_blocks(size), THREADS_PER_BLOCK, 0, stream>>>(x, out, grad, size);
}

void cuda_sqrt_fwd(const half* x, half* out, half* grad, int size, cudaStream_t stream = 0) {
    sqrt_fwd<<<get_blocks(size), THREADS_PER_BLOCK, 0, stream>>>(x, out, grad, size);
}

void cuda_pow_fwd(const half* x, float exponent, half* out, half* grad, int size, cudaStream_t stream = 0) {
    pow_fwd<<<get_blocks(size), THREADS_PER_BLOCK, 0, stream>>>(x, exponent, out, grad, size);
}

void cuda_clip_fwd(const half* x, float lo, float hi, half* out, half* grad, int size, cudaStream_t stream = 0) {
    clip_fwd<<<get_blocks(size), THREADS_PER_BLOCK, 0, stream>>>(x, lo, hi, out, grad, size);
}


void cuda_softmax_fwd(const half* x, half* out, int N, int C, cudaStream_t stream = 0) {
    int shared_mem_bytes = THREADS_PER_BLOCK * sizeof(float);
    softmax_fwd<<<N, THREADS_PER_BLOCK, shared_mem_bytes, stream>>>(x, out, N, C);
}

void cuda_softmax_vjp(const half* s, const half* dout, half* dx, int N, int C, cudaStream_t stream = 0) {
    int shared_mem_bytes = THREADS_PER_BLOCK * sizeof(float);
    softmax_vjp<<<N, THREADS_PER_BLOCK, shared_mem_bytes, stream>>>(s, dout, dx, N, C);
}}