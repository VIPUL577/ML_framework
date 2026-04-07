#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>

#define THREADS_PER_BLOCK 256

namespace seera_cuda
{
__global__ void _cuda_relu_fwd(const float* x, float* out, float* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = x[i];
        out[i]  = val > 0.0f ? val : 0.0f;
        grad[i] = val > 0.0f ? 1.0f : 0.0f;
    }
}

__global__ void _cuda_sigmoid_fwd(const float* x, float* out, float* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = x[i];
        float s = 1.0f / (1.0f + expf(-val));
        out[i]  = s;
        grad[i] = s * (1.0f - s);
    }
}

__global__ void _cuda_tanh_fwd(const float* x, float* out, float* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = x[i];
        float t = tanhf(val);
        out[i]  = t;
        grad[i] = 1.0f - t * t;
    }
}

__global__ void _cuda_log_fwd(const float* x, float* out, float* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = x[i];
        out[i]  = logf(val);
        grad[i] = 1.0f / val;
    }
}

__global__ void _cuda_exp_fwd(const float* x, float* out, float* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = x[i];
        float e = expf(val);
        out[i]  = e;
        grad[i] = e;
    }
}

__global__ void _cuda_abs_fwd(const float* x, float* out, float* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = x[i];
        out[i]  = fabsf(val);
        grad[i] = val > 0.0f ? 1.0f : (val < 0.0f ? -1.0f : 0.0f);
    }
}

__global__ void _cuda_sqrt_fwd(const float* x, float* out, float* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = x[i];
        float s = sqrtf(val);
        out[i]  = s;
        grad[i] = 0.5f / (s + 1e-12f);
    }
}

__global__ void _cuda_pow_fwd(const float* x, float exponent, float* out, float* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = x[i];
        out[i]  = powf(val, exponent);
        grad[i] = exponent * powf(val, exponent - 1.0f);
    }
}

__global__ void _cuda_clip_fwd(const float* x, float lo, float hi, float* out, float* grad, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float val = x[i];
        out[i]  = fminf(fmaxf(val, lo), hi);
        grad[i] = (val >= lo && val <= hi) ? 1.0f : 0.0f;
    }
}



__global__ void _cuda_softmax_fwd(const float* x, float* out, int N, int C) {
    extern __shared__ float smem[]; // Size: blockDim.x
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    const float* x_row = x + row * C;
    float* out_row = out + row * C;

    float thread_max = -CUDART_INF_F;
    for (int i = tid; i < C; i += blockDim.x) {
        thread_max = fmaxf(thread_max, x_row[i]);
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
        thread_sum += expf(x_row[i] - row_max);
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
        float val = expf(x_row[i] - row_max) * inv_sum;
        out_row[i] = val;
    }
}

__global__ void _cuda_softmax_vjp(const float* s, const float* dout, float* dx, int N, int C) {
    extern __shared__ float smem[]; // Size: blockDim.x
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    const float* s_row = s + row * C;
    const float* d_row = dout + row * C;
    float* dx_row = dx + row * C;
    float thread_dot = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        thread_dot += s_row[i] * d_row[i];
    }
    smem[tid] = thread_dot;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float row_dot = smem[0];

    for (int i = tid; i < C; i += blockDim.x) {
        float s_val = s_row[i];
        float d_val = d_row[i];
        float dx_val = s_val * (d_val - row_dot);
        dx_row[i] = dx_val;
    }
}


inline int get_blocks(int size) {
    return (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

void cuda_relu_fwd(const float* x, float* out, float* grad, int size) {
    _cuda_relu_fwd<<<get_blocks(size), THREADS_PER_BLOCK>>>(x, out, grad, size);
}

void cuda_sigmoid_fwd(const float* x, float* out, float* grad, int size) {
    _cuda_sigmoid_fwd<<<get_blocks(size), THREADS_PER_BLOCK>>>(x, out, grad, size);
}

void cuda_tanh_fwd(const float* x, float* out, float* grad, int size) {
    _cuda_tanh_fwd<<<get_blocks(size), THREADS_PER_BLOCK>>>(x, out, grad, size);
}

void cuda_log_fwd(const float* x, float* out, float* grad, int size) {
    _cuda_log_fwd<<<get_blocks(size), THREADS_PER_BLOCK>>>(x, out, grad, size);
}

void cuda_exp_fwd(const float* x, float* out, float* grad, int size) {
    _cuda_exp_fwd<<<get_blocks(size), THREADS_PER_BLOCK>>>(x, out, grad, size);
}

void cuda_abs_fwd(const float* x, float* out, float* grad, int size) {
    _cuda_abs_fwd<<<get_blocks(size), THREADS_PER_BLOCK>>>(x, out, grad, size);
}

void cuda_sqrt_fwd(const float* x, float* out, float* grad, int size) {
    _cuda_sqrt_fwd<<<get_blocks(size), THREADS_PER_BLOCK>>>(x, out, grad, size);
}

void cuda_pow_fwd(const float* x, float exponent, float* out, float* grad, int size) {
    _cuda_pow_fwd<<<get_blocks(size), THREADS_PER_BLOCK>>>(x, exponent, out, grad, size);
}

void cuda_clip_fwd(const float* x, float lo, float hi, float* out, float* grad, int size) {
    _cuda_clip_fwd<<<get_blocks(size), THREADS_PER_BLOCK>>>(x, lo, hi, out, grad, size);
}


void cuda_softmax_fwd(const float* x, float* out, int N, int C) {
    int shared_mem_bytes = THREADS_PER_BLOCK * sizeof(float);
    _cuda_softmax_fwd<<<N, THREADS_PER_BLOCK, shared_mem_bytes>>>(x, out, N, C);
}

void cuda_softmax_vjp(const float* s, const float* dout, float* dx, int N, int C) {
    int shared_mem_bytes = THREADS_PER_BLOCK * sizeof(float);
    _cuda_softmax_vjp<<<N, THREADS_PER_BLOCK, shared_mem_bytes>>>(s, dout, dx, N, C);
}}