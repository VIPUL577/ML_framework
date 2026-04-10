<p align="center">
  <h1 align="center">⚡ Seera</h1>
  <p align="center">
    <strong>A GPU-Accelerated Deep Learning Framework Built from Scratch</strong>
  </p>
  <p align="center">
    Custom autograd engine · Hand-written CUDA kernels with Tensor Core WMMA · NumPy + C++ + CUDA tri-backend
  </p>
</p>

---

Seera is a from-scratch deep learning framework written in Python, C++ and CUDA. It features a complete autograd engine, a high-level Keras-style API for building and training neural networks, and a set of hand-written CUDA kernels that leverage **NVIDIA Tensor Cores (WMMA)** for matrix multiplication, convolution, and transposed convolution. The framework supports seamless CPU and GPU execution — tensors, layers, optimizers, and the entire backward pass all operate directly on device memory without host round-trips.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Building the C++ Engine](#building-the-c-engine)
  - [Building the CUDA Engine](#building-the-cuda-engine)
- [Quick Start](#quick-start)
  - [CPU Training](#cpu-training)
  - [GPU Training](#gpu-training)
- [API Reference](#api-reference)
  - [Tensor](#tensor)
  - [Layers](#layers)
  - [Sequential Model](#sequential-model)
  - [Loss Functions](#loss-functions)
  - [Optimizers](#optimizers)
- [Autograd Engine](#autograd-engine)
- [GPU Tensor: cuTen](#gpu-tensor-cuten)
- [CUDA Kernels — Deep Dive](#cuda-kernels--deep-dive)
  - [GEMM — Tensor Core Matrix Multiplication](#1-gemm--tensor-core-matrix-multiplication-gemmcu)
  - [Convolution — Fused im2col + WMMA](#2-convolution--fused-im2col--wmma-convolutioncu)
  - [Transposed Convolution / Upsampling](#3-transposed-convolution--upsampling-upsamplingcu)
  - [Activation Kernels](#4-activation-kernels-activationscu)
  - [Reduction Kernels](#5-reduction-kernels-reductionkernelscu)
  - [Element-wise Operations](#6-element-wise-operations-elemopscu)
  - [MaxPool2D](#7-maxpool2d-maxpoolcu)
  - [Nearest-Neighbor Unpooling](#8-nearest-neighbor-unpooling-unpoolingcu)
  - [Broadcasting](#9-broadcasting-broadcastcu)
  - [Col2Im](#10-col2im-col2imcu)
  - [Tensor Essentials](#11-tensor-essentials-cuten_essentailscu)
  - [Pybind11 Bindings](#12-pybind11-bindings-cuda_bindingscpp)
- [C++ CPU Engine](#c-cpu-engine)
- [Benchmarking](#benchmarking)
- [Model Save & Load](#model-save--load)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         User API  (Seera.py)                         │
│   Input │ Dense │ Conv2D │ ConvTranspose2D │ Flatten │ MaxPool2D ...  │
│   Sequential │ Loss │ SGD │ Adam                                     │
├──────────────────────────────────────────────────────────────────────┤
│                  Tensor + Autograd  (Seera_init.py)                   │
│   Tensor class with operator overloading, computation graph,         │
│   forward ops (conv2d, matmul, softmax, reductions ...)              │
├──────────────────────────────────────────────────────────────────────┤
│              Autograd Engine  (Seera_Engine.py)                       │
│   Topological sort → backward_step()  (CPU + GPU codepaths)          │
├─────────────────────────┬────────────────────────────────────────────┤
│   CPU Backend           │              GPU Backend                    │
│   ┌─────────────────┐   │   ┌─────────────────────────────────────┐  │
│   │ seera_cpp (.so)  │   │   │  seera_cuda (.so)                   │  │
│   │ C++17 + OpenBLAS │   │   │  CUDA + WMMA Tensor Cores           │  │
│   │ + OpenMP         │   │   │  ~3,000 lines of hand-written .cu   │  │
│   │ 6 source files   │   │   │  12 source files                    │  │
│   └─────────────────┘   │   ├─────────────────────────────────────┤  │
│                          │   │  cuTen  (cuTen.py)                   │  │
│   NumPy fallback         │   │  GPU tensor class wrapping raw       │  │
│   (always available)     │   │  device pointers, automatic memory   │  │
│                          │   │  management via __del__              │  │
│                          │   └─────────────────────────────────────┘  │
└─────────────────────────┴────────────────────────────────────────────┘
```

**Key design decisions:**
- **Three execution tiers**: Every operation checks `_is_gpu(value)` first, then `_USE_CPP`, then falls back to NumPy.
- **No host-device data transfer during training**: Weights, gradients, optimizer state, and intermediate tensors all live on-device. Only the input data is transferred once per batch.
- **cuTen wraps raw `cudaMalloc` pointers**: This avoids the overhead of PyCUDA or CuPy and gives full control over memory lifetime.

---

## Features

| Feature | CPU (NumPy) | CPU (C++) | GPU (CUDA) |
|---|:---:|:---:|:---:|
| Dense (fully-connected) layers | ✅ | ✅ OpenBLAS GEMM | ✅ WMMA Tensor Core |
| Conv2D (forward + backward) | ✅ | ✅ im2col + BLAS | ✅ Fused im2col + WMMA |
| ConvTranspose2D | ✅ | ✅ | ✅ WMMA |
| MaxPool2D (forward + backward) | ✅ | ✅ | ✅ |
| Nearest-Neighbor Unpooling | ✅ | ✅ | ✅ |
| Batch Normalization (1D / 2D) | ✅ | ✅ | Forward only |
| Activations (ReLU, Sigmoid, Tanh) | ✅ | ✅ | ✅ |
| Softmax + VJP backward | ✅ | ✅ | ✅ Shared-memory reduction |
| log, exp, abs, sqrt, pow, clip | ✅ | ✅ | ✅ |
| Reductions (sum, mean, max, min) | ✅ | — | ✅ N-dimensional |
| Argmax / Argmin | — | — | ✅ |
| Broadcasting (add, mul) | ✅ | — | ✅ 4D kernel |
| Concatenation (1D, 2D) | ✅ | — | ✅ |
| Transpose (.T) | ✅ | — | ✅ 2D/3D kernels |
| Flatten | ✅ | — | ✅ |
| Autograd (full backward pass) | ✅ | ✅ | ✅ |
| SGD optimizer (w/ momentum) | ✅ | — | ✅ GPU-native |
| Adam optimizer | ✅ | — | ✅ GPU-native |
| Model save/load (pickle) | ✅ | ✅ | ✅ |

---

## Project Structure

```
ML_framework/
├── Seera.py                    # High-level API: Layers, Sequential, Loss, Optimizers
├── Seera_Engine.py             # Autograd engine (backward pass, CPU + GPU)
├── Seera_init.py               # Tensor class, operator overloading, forward ops
├── cuTen.py                    # GPU tensor class wrapping raw CUDA pointers
│
├── engine/                     # C++ CPU acceleration engine
│   ├── include/
│   │   └── seera_engine.hpp    # C++ function declarations
│   └── src/
│       ├── tensor_ops.cpp      # Matmul (OpenBLAS), element-wise ops
│       ├── activation_ops.cpp  # ReLU, Sigmoid, Tanh, Log, Exp, Sqrt, etc.
│       ├── conv_ops.cpp        # Conv2D, ConvTranspose2D (im2col + BLAS)
│       ├── pool_ops.cpp        # MaxPool2D, Unpooling
│       ├── batchnorm_ops.cpp   # BatchNorm forward + backward
│       └── bindings.cpp        # pybind11 bindings → seera_cpp.so
│
├── engine_cuda/                # CUDA GPU acceleration engine
│   ├── include/
│   │   └── seera_engine_cuda.hpp   # CUDA kernel declarations
│   └── src/
│       ├── GEMM.cu             # Tensor Core WMMA matmul + backward + transpose
│       ├── convolution.cu      # Conv2D forward + backward (fused WMMA)
│       ├── upsampling.cu       # ConvTranspose2D fwd + bwd (WMMA + col2im)
│       ├── activations.cu      # All activation fwd kernels + softmax VJP
│       ├── reductionKernels.cu # Sum, Mean, Max, Min, Argmax, Argmin + backward
│       ├── elemops.cu          # Element-wise add, sub, mul, div
│       ├── maxPool.cu          # MaxPool2D forward + backward (with mask)
│       ├── unpooling.cu        # Nearest-neighbor upsample + backward
│       ├── broadcast.cu        # 4D broadcasting (add, mul) kernel
│       ├── col2im.cu           # Col-to-image reconstruction
│       ├── cuTen_essentails.cu # Scalar ops, fill, power, zeros, ones
│       └── cuda_bindings.cpp   # pybind11 bindings → seera_cuda.so
│
├── build_engine.py             # Build script for C++ engine (g++)
├── build_engine_cuda.py        # Build script for CUDA engine (nvcc)
├── benchmark.py                # PyTorch MNIST benchmark for comparison
│
├── Testing/                    # Test scripts and diagnostic tools
│   ├── test_1.py ... test_15.py
│   ├── batch_example.py
│   ├── diagnose_*.py
│   └── mnist_seera_vs_pytorch.py
│
├── demo.ipynb                  # Jupyter notebook demos
├── demo3.ipynb
└── testing.ipynb
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **NumPy**
- **pybind11** (`pip install pybind11`)
- **GCC / G++ 11+** with C++17 support
- **OpenBLAS** (`apt install libopenblas-dev`)
- **CUDA Toolkit 12.0+** with an NVIDIA GPU supporting `sm_89` (RTX 4090, RTX 4080, etc.)
- **matplotlib** (for training curve visualization)

### Building the C++ Engine

```bash
# Compiles engine/src/*.cpp → seera_cpp.cpython-*.so
python build_engine.py
```

This uses `g++` with `-O3`, `-fopenmp`, and links against OpenBLAS. Produces a shared library that is imported as `seera_cpp` in Python.

### Building the CUDA Engine

```bash
# Compiles engine_cuda/src/*.cu → seera_cuda.cpython-*.so
python build_engine_cuda.py
```

This performs a two-step build:

1. **nvcc** compiles each `.cu` file to an object file (`-arch=sm_89`, `-O3`, `-std=c++17`)
2. **nvcc** links all `.o` files into a shared library linked against `libcudart`

> **Note**: Change `-arch=sm_89` in `build_engine_cuda.py` to match your GPU's compute capability (e.g. `sm_75` for Turing, `sm_80` for Ampere, `sm_86` for GA10x).

---

## Quick Start

### CPU Training

```python
from Seera import *
from Seera_Engine import Tensor, np

# Define model
model = Sequential([
    Input((784,)),
    Dense(784, 128, activation="relu"),
    Dense(128, 64,  activation="relu"),
    Dense(64,  10,  activation="softmax"),
])

# Loss and optimizer
loss_fn = Loss().categorical_cross_entropy
optimizer = Adam(model, lr=0.001)

# Train
history = model.fit(X_train, y_train_onehot, optimizer, loss_fn,
                    Epochs=10, batch_size=32, Loss_interval=1)
```

### GPU Training

```python
from Seera import *
from Seera_Engine import Tensor, np

# Simply pass device="cuda" — everything moves to GPU
model = Sequential([
    Input((784,)),
    Dense(784, 128, activation="relu"),
    Dense(128, 64,  activation="relu"),
    Dense(64,  10,  activation="softmax"),
], device="cuda")

loss_fn = Loss().categorical_cross_entropy
optimizer = Adam(model, lr=0.001)

# Training runs entirely on GPU — no code changes needed
history = model.fit(X_train, y_train_onehot, optimizer, loss_fn,
                    Epochs=10, batch_size=64, Loss_interval=1)
```

### CNN Example

```python
model = Sequential([
    Input((1, 28, 28)),
    Conv2D(32, 1, (3, 3), activation="relu", zero_padding=1),
    MaxPool2D(pool_size=(2, 2), stride=2),
    Conv2D(64, 32, (3, 3), activation="relu", zero_padding=1),
    MaxPool2D(pool_size=(2, 2), stride=2),
    Flatten(),
    Dense(64 * 7 * 7, 128, activation="relu"),
    Dense(128, 10, activation="softmax"),
], device="cuda")
```

---

## API Reference

### Tensor

The `Tensor` class (defined in `Seera_init.py`) is the core data structure. It wraps either a NumPy array (CPU) or a `cuten` object (GPU) and maintains a computation graph for automatic differentiation.

```python
# Creation
x = Tensor(np.array([1, 2, 3]), is_leaf=True)
x = Tensor(data, is_leaf=True, device="cuda")   # GPU tensor
x = Tensor.zeros((3, 4))
x = Tensor.randn(3, 4)
x = Tensor.eye(4)

# Operations (all tracked in the computation graph)
y = x.matmul(w)           # Matrix multiplication
y = x.relu()              # Activations: relu, sigmoid, tanh, softmax
y = x.conv2d(kernel, stride=1, padding=0)
y = x.maxpool2d((2, 2), stride=2)
y = x.sum(axis=0)         # Reductions: sum, mean, max, min
y = x.flatten()
y = x.T()                 # Transpose
y = x.log(); y = x.exp()
y = x ** 2                # Power
y = x.clip(0.0, 1.0)
```

### Layers

| Layer | Description | Parameters |
|---|---|---|
| `Input(shape)` | Input placeholder | shape: per-sample shape |
| `Dense(in, out, activation)` | Fully-connected | kernel_initializer, bias_initializer |
| `Conv2D(out_ch, in_ch, kernel_size, activation)` | 2D convolution (NCHW) | stride, zero_padding, initializer |
| `ConvTranspose2D(out_ch, in_ch, kernel_size, activation)` | Transposed convolution | stride, zero_padding |
| `MaxPool2D(pool_size, stride, padding)` | Max pooling | — |
| `Flatten()` | Batch-preserving flatten | — |
| `Unpool2D_Nearest(size)` | Nearest-neighbor upsample | scale factor (h, w) |
| `Concatenate()` | Channel concatenation | 2 layers max |
| `BatchNorm1d(num_features)` | Batch normalization for Dense | momentum, eps |
| `BatchNorm2d(num_channels)` | Batch normalization for Conv | momentum, eps |

**Supported activations**: `"relu"`, `"sigmoid"`, `"softmax"`, `"tanh"`

**Supported initializers**: `"zeros"`, `"ones"`, `"random_normal"`, `"random_uniform"`, `"he_normal"`, `"he_uniform"`, `"glorot_normal"`, `"glorot_uniform"`, `"lecun_normal"`, `"lecun_uniform"`

### Sequential Model

```python
model = Sequential(layers, device="cpu")  # or device="cuda"
model.summary()                           # Print architecture
output = model.forward(X)                 # Forward pass
output = model.predict(X)                 # Eval mode (disables BN training stats)
history = model.fit(X, y, optimizer, loss, Epochs=100, batch_size=32)
model.save("model.pkl")
model = Sequential.load("model.pkl")
```

### Loss Functions

```python
loss = Loss()
l = loss.mse(y_pred, y)                              # Mean Squared Error
l = loss.mae(y_pred, y)                               # Mean Absolute Error
l = loss.binary_cross_entropy(y_pred, y)              # Binary Cross-Entropy
l = loss.categorical_cross_entropy(y_pred, y)         # Categorical Cross-Entropy
```

### Optimizers

```python
# SGD with optional momentum
optimizer = SGD(model, lr=0.01, momentum=0.9)

# Adam
optimizer = Adam(model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
```

Both optimizers operate **entirely on-device** when `model.device == "cuda"` — no host-device transfers during parameter updates.

---

## Autograd Engine

The autograd engine (`Seera_Engine.py`) implements reverse-mode automatic differentiation using a dynamic computation graph:

1. **Graph Construction**: Every `Tensor` operation records its children and local gradients in `node.child_grad`. Special context flags (`.matm`, `.iconv2d`, `.isoftmax`, etc.) mark operations that require custom backward logic.

2. **Topological Sort**: `buildgraph()` performs a DFS from the loss tensor to produce a reverse topological ordering.

3. **Backward Pass**: `backward_step()` dispatches to the appropriate kernel based on context flags:
   - **Matmul backward**: `dA = dC @ B^T`, `dB = A^T @ dC` (uses WMMA GEMM on GPU)
   - **Conv2D backward**: WMMA-accelerated `dX` via transposed convolution + fused WMMA `dW` with batch reduction
   - **ConvTranspose2D backward**: Standard conv2d of `dY` with `W` for `dX` + fused WMMA `dW`
   - **Softmax backward**: VJP formula `dx = s * (dout - dot(s, dout))` via shared-memory reduction kernel
   - **Reduction backward**: Dimension-aware broadcasting/scatter (sum, mean, max, min) 
   - **Element-wise backward**: Chain rule with automatic broadcast gradient reduction

4. **Dual GPU/CPU paths**: Every backward operation checks `_is_gpu()` and dispatches to the appropriate CUDA kernel or NumPy/C++ fallback. GPU gradient accumulation uses `cuten.__add__` (never in-place `+=` on device pointers).

---

## GPU Tensor: cuTen

`cuTen` (`cuTen.py`) is the GPU tensor class that wraps raw `cudaMalloc` device pointers. It is designed as a GPU-native analog to NumPy:

- **Automatic memory management**: `__del__` calls `cuda_free()` on garbage collection. Pointer ownership is tracked via `_owns_memory` to prevent double-frees when reshaping (which shares memory).
- **Zero-copy reshape**: `reshape()` creates a new `cuten` that shares the same device pointer but with different shape metadata.
- **Full operator overloading**: `+`, `-`, `*`, `/`, `**` with both tensor-tensor and tensor-scalar support.
- **Broadcasting**: 4D broadcasting for mismatched shapes via the `broadcast_add_4d` / `broadcast_mul_4d` CUDA kernels.
- **20+ operations**: relu, sigmoid, tanh, log, exp, abs, sqrt, clip, softmax, matmul, conv2d, maxpool2d, unpool, conv2d_transpose, concatenate, flatten, sum, mean, max, min, transpose.

```python
from cuTen import cuten
import numpy as np

# Create a GPU tensor
x = cuten(np.random.randn(64, 784).astype(np.float32))

# All operations run on GPU — no host round-trips
y = x.matmul(w)        # WMMA Tensor Core matmul
y = y.relu()            # CUDA activation kernel
y = y.softmax()         # Shared-memory softmax
s = y.sum(dim=1)        # Reduction kernel

# Transfer back to host when needed
result = y.to_host_f32()  # → numpy array
```

---

## CUDA Kernels — Deep Dive

The `engine_cuda/` directory contains **~3,000 lines of hand-written CUDA code** across 12 source files. All kernels live in the `seera_cuda` namespace.

### 1. GEMM — Tensor Core Matrix Multiplication (`GEMM.cu`)

**348 lines** | The core computational kernel powering Dense layer forward/backward.

#### Forward: `cuda_matmul`

Computes batched matrix multiplication `C[b,M,N] = A[b,M,K] @ B[K,N]` using **NVIDIA Tensor Cores** via the WMMA (Warp Matrix Multiply-Accumulate) API.

**How it works:**

```
Grid: ((N+15)/16, (M+15)/16, Nbatch)  — one block per 16×16 output tile per batch
Block: 32 threads (one warp)
```

1. Each warp computes a **16×16 output tile** of C.
2. The K dimension is tiled in steps of 16. For each tile:
   - **32 threads cooperatively load** a 16×16 tile of A and a 16×16 tile of B into shared memory, converting `float32 → half` on the fly.
   - A is batched (`A[batchno*M + row, col]`), B is shared across batches.
   - `wmma::load_matrix_sync` loads from shared memory into WMMA fragments.
   - `wmma::mma_sync` performs the 16×16×16 FP16 multiply-accumulate into an FP32 accumulator.
3. After all K-tiles, the FP32 accumulator is stored to a shared-memory staging buffer, then written back to global memory with **bounds checking** to handle non-multiple-of-16 dimensions.

**Bounds handling**: The kernel zero-pads tiles where `global_row >= M` or `global_col >= N`, so arbitrary matrix dimensions work correctly.

#### Backward: `cuda_matmul_bwd`

Computes gradients `dA` and `dB` from the upstream gradient `dC`:
- **`dA[b,M,K] = dC[b,M,N] @ B^T[N,K]`** — single batched WMMA call (B^T is shared across batches)
- **`dB[K,N] = Σ_b A_b^T[K,M] @ dC_b[M,N]`** — per-batch WMMA calls with element-wise accumulation

Internally allocates temporary GPU buffers for transposed matrices (`B_T`, `A_T`), performs the transpositions via dedicated `transpose_2d` / `transpose_3d_batch` kernels, then reuses the same `matmul_wmma_bound` kernel for the actual multiplications.

#### Transpose Kernels

- **`transpose_2d`**: One thread per element. `out[c*rows + r] = in[r*cols + c]`.
- **`transpose_3d_batch`**: Same logic but with `blockIdx.z` selecting the batch slice. Used for batched `A^T`.

---

### 2. Convolution — Fused im2col + WMMA (`convolution.cu`)

**455 lines** | Conv2D forward and backward, both using Tensor Core acceleration via **fused im2col** — the im2col transformation is performed on-the-fly during WMMA tile loading instead of materializing a separate matrix.

#### Forward: `convulution_eff`

Formulates convolution as a matrix multiplication:
```
Output[b,n,h,w] = Σ_{c,r,s} Kernel[n, c*R*S + r*S+s] × im2col(X)[c*R*S + r*S+s, h_out*W_out + w_out]
```

**Grid layout**: `((H_out*W_out+15)/16, (N_filters+15)/16, Batch)` — one block covers a 16×16 tile of the `[spatial × filters]` output.

**Key optimization**: Instead of a separate im2col pass (which doubles memory), each thread computes the input image coordinates `(h_in, w_in)` on-the-fly from the kernel offsets `(r, s)` and the output position `(h_out, w_out)`:

```c
int h_in = iy_h_out * stride_h - pad_h + ky_ir;
int w_in = iy_w_out * stride_w - pad_w + ky_is;
```

This fused approach:
- **Saves ~2× GPU memory** (no explicit im2col buffer)
- **Reduces memory bandwidth** (data read directly from input image)
- Handles padding via bounds checks (out-of-bounds → zero)

#### Backward — dX: `conv2d_bwd_transmatmul` + `conv2d_bwd_col2im`

`dX` is computed as **ConvTranspose(dY, W)**, implemented in two stages:
1. **WMMA transposed matmul**: Multiplies the transposed weight matrix by the upstream gradient. Uses the same fused im2col loading pattern but with transposed weight indexing: `W[A_cin * (Cout*R*S) + A_cout*(R*S) + A_r*S + A_s]`
2. **col2im**: A separate kernel reconstructs the spatial dimensions from the column buffer, respecting stride and padding.

#### Backward — dW: `conv2d_dW_kernel`

Per-batch fused WMMA kernel computing: `dW_b[N, C*R*S] = dY_b[N, spatial] @ im2col(X_b)[C*R*S, spatial]^T`

- Uses **`wmma::col_major`** for the B fragment to implicitly transpose the im2col matrix
- Batch results are reduced via `_weight_reduce` (each thread sums one weight element across all batches)

---

### 3. Transposed Convolution / Upsampling (`upsampling.cu`)

**403 lines** | ConvTranspose2D (learnable upsampling) forward and backward passes.

#### Forward: `cuda_conv2DTranpose_fwd`

Transposed convolution `X(B,Cin,H,W) @ W(Cin,Cout,KH,KW) → (B,Cout,Hout,Wout)` where:
```
Hout = (H-1)*stride_h - 2*pad_h + KH
Wout = (W-1)*stride_w - 2*pad_w + KW
```

**Two-phase approach:**
1. **`conv2dTransmatmul` (WMMA)**: Produces an intermediate `[Cout*KH*KW, Batch*Hin*Win]` matrix. Weight matrix A is loaded with transposed indexing, input X is loaded with spatial-batch interleaving.
2. **`col2im`**: Folds the column matrix back into spatial dimensions `(B, Cout, Hout, Wout)`, handling stride and padding.

#### Backward: `cuda_conv2DTranspose_bwd`

- **dX**: Standard conv2d of `dY(B,Cout,Hout,Wout)` with `W(Cin,Cout,KH,KW)` → `dX(B,Cin,Hin,Win)`. Uses `convulution_eff_bwd` kernel.
- **dW**: Per-batch fused WMMA kernel (`conv2dTrans_dW_kernel`). The im2col over `dY` is fused into the shared memory load: each thread computes the `dY` output position corresponding to input position `(h_in, w_in)` and kernel offset `(kh, kw)`, then fetches the value if in bounds. Batch reduction via `dW_batch_reduce`.

---

### 4. Activation Kernels (`activations.cu`)

**209 lines** | All activation functions produce both the **forward output** and the **local gradient** in a single kernel launch — no separate backward pass needed.

| Kernel | Forward | Gradient |
|---|---|---|
| `_cuda_relu_fwd` | `max(x, 0)` | `x > 0 ? 1 : 0` |
| `_cuda_sigmoid_fwd` | `1 / (1 + exp(-x))` | `s * (1 - s)` |
| `_cuda_tanh_fwd` | `tanh(x)` | `1 - t²` |
| `_cuda_log_fwd` | `log(x)` | `1 / x` |
| `_cuda_exp_fwd` | `exp(x)` | `exp(x)` |
| `_cuda_abs_fwd` | `|x|` | `sign(x)` |
| `_cuda_sqrt_fwd` | `√x` | `0.5 / (√x + ε)` |
| `_cuda_pow_fwd` | `x^n` | `n * x^(n-1)` |
| `_cuda_clip_fwd` | `clamp(x, lo, hi)` | `1 if lo ≤ x ≤ hi else 0` |

All use grid-stride loops: `for (int i = tid; i < size; i += blockDim.x * gridDim.x)`.

#### Softmax Forward: `_cuda_softmax_fwd`

One block per row (sample), with shared-memory parallel reductions:
1. **Find row max** (for numerical stability): parallel max reduction via `smem[tid]`
2. **Compute exp-sum**: each thread handles multiple columns via striding, then shared-memory sum reduction
3. **Normalize**: `out[i] = exp(x[i] - max) / sum`

#### Softmax VJP Backward: `_cuda_softmax_vjp`

Computes `dx = s * (dout - dot(s, dout))` per-row:
1. Parallel dot product `dot(s, dout)` via shared-memory reduction
2. Element-wise: `dx[i] = s[i] * (dout[i] - row_dot)`

---

### 5. Reduction Kernels (`reductionKernels.cu`)

**381 lines** | Generic N-dimensional reduction along any axis. Supports both forward and backward for sum, mean, max, min, argmax, argmin.

#### Forward Architecture

All reductions use a single unified approach based on **stride/limit decomposition**:

```
For an array of shape (d0, d1, ..., d_{n-1}) reducing dim k:
  stride = product of dims after k (d_{k+1} * ... * d_{n-1})
  limit  = stride * d_k
  totalthreads = product of all dims except d_k
```

Each thread handles one output element. It iterates through `limit/stride` elements (the size of the reduced dimension), accumulating the result:

```c
int inner = tid % stride;
int outer = tid / stride;
int base  = outer * limit + inner;
for (int i = 0; i < limit; i += stride)
    temp += arr[base + i];     // sum/mean
    // or fmaxf(temp, arr[...])  // max
    // or fminf(temp, arr[...])  // min
```

For **mean**, the divisor `dimarr[dim]` is passed in and applied: `output[tid] = temp / divisor`.

#### Backward Kernels

- **Sum backward** (`Reductionsum_bwd`): Broadcasts the upstream gradient to all positions along the reduced dimension: `dA[base+i] = dOut[tid]` for all `i`.
- **Mean backward**: Same as sum backward but divides by `dimarr[dim]`.
- **Max/Min backward** (`Reductionmax_bwd`, `Reductionmin_bwd`): Sparse gradient — only the position matching `fwdOutput[tid]` receives the gradient. All others get zero. Uses a `found` flag to select only the first match.

---

### 6. Element-wise Operations (`elemops.cu`)

**124 lines** | Simple per-element kernels for same-shape tensor operations.

- **`elemadd`**: `C[i] = A[i] + B[i]`
- **`elemsub`**: `C[i] = A[i] - B[i]`
- **`elemmult`**: `C[i] = A[i] * B[i]`
- **`elemdiv`**: `C[i] = A[i] / B[i]`

Each uses 256 threads per block with `(size + 255) / 256` blocks. These are the fast path for same-shape operands; broadcasting uses the separate 4D kernel.

---

### 7. MaxPool2D (`maxPool.cu`)

**157 lines** | Forward pool with mask recording + backward scatter.

#### Forward: `maxPool`

One thread per output element `(c, h_out, w_out)`. Iterates over the `R×S` kernel window, tracks the maximum value and its position `(rmax, smax)`, then:
- Writes the max value to `conv[output_idx]`
- Sets `mask[input_idx_of_max] = 1` (binary mask on input spatial positions)

#### Backward: `maxPool_bwd`

Uses `atomicAdd` to scatter the upstream gradient to the max positions:
```c
atomicAdd(&dX[index], dout[...] * (float)mask[index]);
```

Grid uses `(spatial, C, Batch)` dimensions for full parallelism.

---

### 8. Nearest-Neighbor Unpooling (`unpooling.cu`)

**98 lines** | Upsample by integer scale factors `(sh, sw)`.

#### Forward: `unpooling`

Each thread writes one output pixel by looking up the corresponding input pixel via integer division:
```c
int h_in = h_out / sh;
int w_in = w_out / sw;
out[...] = inp[...];
```

#### Backward: `unpooling_bwd`

Each thread handles one input pixel and sums the `sh × sw` gradient contributions from the upsampled region:
```c
for (ii = 0; ii < sh; ii++)
    for (jj = 0; jj < sw; jj++)
        temp += dout[h*sh+ii, w*sw+jj];
dx[...] = temp;
```

---

### 9. Broadcasting (`broadcast.cu`)

**154 lines** | NumPy-style 4D broadcast for add and multiply.

#### Kernel: `broadcast_kernel_4d`

Each thread computes one output element at position `(n, c, h, w)`. The broadcast mapping zeros out indices where the corresponding dimension is 1:

```c
int an = (aN == 1) ? 0 : n;   // If A has size 1 in dim N, always read index 0
int bn = (bN == 1) ? 0 : n;   // Same for B
// ... same for C, H, W dims
C[tid] = (op == 0) ? (A[a_idx] + B[b_idx]) : (A[a_idx] * B[b_idx]);
```

The host-side function `compute_out_shape_4d` resolves the output shape following NumPy rules, returning -1 for incompatible shapes.

---

### 10. Col2Im (`col2im.cu`)

**77 lines** | Column-to-image reconstruction used in convolution backward passes.

Each thread reconstructs one pixel `(batchN, c, h_in, w_in)` by iterating over all `R×S` kernel positions that could have contributed to this pixel, accumulating from the column matrix. Handles stride/padding via modulo checks.

---

### 11. Tensor Essentials (`cuTen_essentails.cu`)

**149 lines** | Utility kernels for tensor initialization and scalar operations:

- `cuda_scaler_multiply_{h,f}`: In-place `arr[i] *= k`
- `cuda_scaler_add_f`: In-place `arr[i] += k`
- `cuda_scaler_power_f`: In-place `arr[i] = pow(arr[i], k)`
- `cuda_ones_{h,f}`: Fill with 1.0
- `cuda_zeros_{h,f}`: Fill with 0.0

---

### 12. Pybind11 Bindings (`cuda_bindings.cpp`)

**~800 lines** | Bridges the CUDA kernels to Python via pybind11. Handles:
- **Host-to-device transfers**: `to_device_f32(np.ndarray) → int (device pointer)`
- **Device-to-host transfers**: `to_host_f32(ptr, shape) → np.ndarray`
- **Memory management**: `cuda_malloc_f32`, `cuda_free`, `cuda_memset`, `cuda_memcopy_devicetodevice`
- **Kernel wrappers**: All CUDA kernel functions are exposed with pointer-based signatures
- **Device pointers as Python ints**: GPU pointers are cast to `uintptr_t` and passed as Python integers

---

## C++ CPU Engine

The `engine/` directory contains a **C++17 CPU backend** (~940 lines) used when a CUDA GPU is unavailable:

| File | Operations | Acceleration |
|---|---|---|
| `tensor_ops.cpp` | Matmul, element-wise add/mul | OpenBLAS `cblas_sgemm` |
| `activation_ops.cpp` | ReLU, Sigmoid, Tanh, Log, Exp, Abs, Sqrt, Pow, Clip | OpenMP |
| `conv_ops.cpp` | Conv2D fwd/bwd, ConvTranspose2D fwd/bwd, im2col, col2im | OpenBLAS + OpenMP |
| `pool_ops.cpp` | MaxPool2D fwd/bwd, Unpooling fwd/bwd | OpenMP |
| `batchnorm_ops.cpp` | BatchNorm forward + backward (1D and 2D) | OpenMP |
| `bindings.cpp` | pybind11 bindings → `seera_cpp.so` | — |

The framework auto-detects available backends at import time:
```python
try:
    import seera_cuda   # GPU available
except ImportError:
    try:
        import seera_cpp  # CPU C++ available
    except ImportError:
        # Pure NumPy fallback
```

---

## Benchmarking

`benchmark.py` provides a PyTorch reference implementation for MNIST classification. Use it to compare Seera's convergence and speed:

```bash
python benchmark.py   # PyTorch reference (CUDA)
```

The `Testing/mnist_seera_vs_pytorch.py` script performs a head-to-head gradient comparison between Seera (CPU), Seera (GPU), and PyTorch (GPU) to validate numerical parity.

---

## Model Save & Load

Models are serialized using Python's `pickle` module:

```python
# Save — automatically transfers GPU tensors to host
model.save("my_model.pkl")

# Load — reconstructs all layers and reconnects the graph
model = Sequential.load("my_model.pkl")
```

The serialization captures:
- Layer types and configuration (shapes, activations, stride, padding)
- Weight and bias values (converted to NumPy if on GPU)
- BatchNorm running statistics

---

<p align="center">
  <sub>Built with ❤️, CUDA, and many late nights.</sub>
</p>
