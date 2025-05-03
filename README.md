# Seera: A Minimal Deep Learning Framework

Seera is a minimal, educational deep learning framework built from scratch in Python and NumPy. It features a custom autograd engine, a flexible tensor class, and a modular neural network API inspired by modern ML libraries. Seera is designed for learning, experimentation, and understanding the inner workings of deep learning frameworks.

---

## Features

- **Custom Tensor Class:** Supports element-wise operations, broadcasting, and gradients.
- **Autograd Engine:** Handles automatic differentiation for both standard and neural network-specific operations.
- **Layer API:** Modular layers (Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, Concatenate, etc.) for building neural networks.
- **Model API:** Sequential model building, training loop, and support for saving/loading models.
- **Optimizers:** Includes SGD and Adam optimizers.
- **Loss Functions:** MSE, MAE, Binary Cross-Entropy, and Categorical Cross-Entropy.
- **Numba Acceleration:** Core tensor ops use Numba for speed.
- **Educational:** Minimal dependencies, readable code, and clear separation of components.

---

## Limitations

> **Current Limitations of Seera:**
>
> - **Batch Size:** The current version only supports batch size 1 for training and inference. While a batched version exists, it is computationally heavy and not practical for normal CPUs.
> - **Python Only:** The entire codebase is implemented in Python, which may limit performance compared to frameworks with C/C++ backends.
> - **Chain Product Update:** The method used for updating the chain product (backpropagation through chained matrix products) is computationally heavy and may be slow for deep or wide networks.

---

## Installation

```bash
git clone https://github.com/yourusername/seera.git
cd seera
# (Optional) Install dependencies
pip install numpy numba matplotlib tensorflow
```

---

## Quick Start

```python
from Seera import Sequential, Input, Dense, Loss, SGD
from Seera_Engine import Tensor

# Build a simple model
model = Sequential([
    Input((3, 1)),  # Input shape (features, batch)
    Dense(3, 2, activation="relu"),
    Dense(2, 1, activation="sigmoid")
])

# Dummy data
X_train = Tensor.random((10, 3, 1))
y_train = Tensor.random((10, 1, 1))

# Compile and train
loss_fn = Loss().mse
optimizer = SGD(model, lr=0.01)
model.fit(X_train.value, y_train.value, optimizer, loss_fn, Epochs=10)
```

---

## Core Components

| Component    | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `Tensor`     | Numpy-like object with autograd support.                                    |
| `autograd4nn`| Custom backward engine for neural networks.                                 |
| `Layer`      | Base class for all layers (Dense, Conv2D, etc.).                            |
| `Sequential` | Model container for stacking layers.                                        |
| `Loss`       | Collection of loss functions.                                               |
| `SGD`, `Adam`| Optimizers for parameter updates.                                           |

---

## Example: Custom Model

```python
from Seera import Sequential, Input, Dense, Flatten, Conv2D, MaxPool2D

model = Sequential([
    Input((1, 28, 28)),
    Conv2D(8, 1, (3, 3), activation="relu"),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(1352, 10, activation="softmax")
])
model.summary()
```

---

## About

Seera is developed as a learning project to demystify how deep learning frameworks work under the hood. It is not intended for production use, but as a reference for students, educators, and anyone interested in the building blocks of neural networks.

**Key Principles:**
- Minimalism and clarity
- Explicit autograd and computation graph construction
- Layer and model abstractions similar to Keras/PyTorch
- Easy to extend for new operations and layers

---

## Contributing

Contributions, bug reports, and suggestions are welcome! Please open an issue or submit a pull request.
