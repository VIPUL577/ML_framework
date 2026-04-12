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



## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         User API  (Seera.py)                         │
│   Input │ Dense │ Conv2D │ ConvTranspose2D │ Flatten │ MaxPool2D ... │
│   Sequential │ Loss │ SGD │ Adam                                     │
├──────────────────────────────────────────────────────────────────────┤
│                  Tensor + Autograd  (Seera_init.py)                  │
│   Tensor class with operator overloading, computation graph,         │
│   forward ops (conv2d, matmul, softmax, reductions ...)              │
├──────────────────────────────────────────────────────────────────────┤
│              Autograd Engine  (Seera_Engine.py)                      │
│   Topological sort → backward_step()  (CPU + GPU codepaths)          │
├─────────────────────────┬────────────────────────────────────────────┤
│   CPU Backend           │              GPU Backend                   │
│   ┌─────────────────┐   │   ┌─────────────────────────────────────┐  │
│   │ seera_cpp (.so)  │  │   │  seera_cuda (.so)                   │  │
│   │ C++17 + OpenBLAS │  │   │  CUDA + WMMA Tensor Cores           │  │
│   │ + OpenMP         │  │   │  ~3,000 lines of hand-written .cu   │  │
│   │ 6 source files   │  │   │  12 source files                    │  │
│   └─────────────────┘   │   ├─────────────────────────────────────┤  │
│                         │   │  cuTen  (cuTen.py)                  │  │
│   NumPy fallback        │   │  GPU tensor class wrapping raw      │  │
│   (always available)    │   │  device pointers, automatic memory  │  │
│                         │   │  management via __del__             │  │
│                         │   └─────────────────────────────────────┘  │
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
- **CUDA Toolkit 12.0+** with an NVIDIA GPU supporting `sm_89` (All RTX series. Mine was 4050!!)
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
  <sub>Built with ❤️ for CUDA and programming</sub>
</p>
