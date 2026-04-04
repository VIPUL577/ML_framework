"""
Seera Framework — Batch Training Example
=========================================
Demonstrates:
  1. ANN batch training with BatchNorm1d
  2. CNN batch training with BatchNorm2d
  3. Gradient correctness check via finite differences
"""

import numpy as np
import sys
import os

# Ensure the framework can be imported
sys.path.insert(0, os.path.dirname(__file__))

from Seera_init import tensor as Tensor
from Seera_Engine import autograd4nn
from Seera import (
    Sequential, Input, Dense, Conv2D, Flatten, MaxPool2D,
    BatchNorm1d, BatchNorm2d, Loss, Adam, SGD,
)


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# 1. Gradient Check (finite differences)
# ─────────────────────────────────────────────────────────────
def gradient_check():
    separator("Gradient Check — Dense Layer (batch=4)")
    np.random.seed(42)

    N, D_in, D_out = 4, 3, 2
    X_np = np.random.randn(N, D_in).astype(np.float32)
    y_np = np.random.randn(N, D_out).astype(np.float32)
    W_np = np.random.randn(D_in, D_out).astype(np.float32) * 0.1
    b_np = np.random.randn(1, D_out).astype(np.float32) * 0.1

    # --- Analytical gradient via autograd ---
    X = Tensor(X_np, is_leaf=True)
    W = Tensor(W_np.copy(), is_leaf=True)
    b = Tensor(b_np.copy(), is_leaf=True)

    z = X.matmul(W) + b
    a = z.relu()
    loss = ((a - Tensor(y_np)) ** 2).mean()

    autograd4nn(loss)
    analytic_dW = W.node.cp.copy()
    analytic_db = b.node.cp.copy()

    # --- Numerical gradient via finite differences ---
    eps = 1e-4
    num_dW = np.zeros_like(W_np)
    for i in range(W_np.shape[0]):
        for j in range(W_np.shape[1]):
            W_plus = W_np.copy(); W_plus[i, j] += eps
            W_minus = W_np.copy(); W_minus[i, j] -= eps

            z_p = X_np @ W_plus + b_np
            a_p = np.where(z_p > 0, z_p, 0)
            loss_p = np.mean((a_p - y_np) ** 2)

            z_m = X_np @ W_minus + b_np
            a_m = np.where(z_m > 0, z_m, 0)
            loss_m = np.mean((a_m - y_np) ** 2)

            num_dW[i, j] = (loss_p - loss_m) / (2 * eps)

    num_db = np.zeros_like(b_np)
    for j in range(b_np.shape[1]):
        b_plus = b_np.copy(); b_plus[0, j] += eps
        b_minus = b_np.copy(); b_minus[0, j] -= eps

        z_p = X_np @ W_np + b_plus
        a_p = np.where(z_p > 0, z_p, 0)
        loss_p = np.mean((a_p - y_np) ** 2)

        z_m = X_np @ W_np + b_minus
        a_m = np.where(z_m > 0, z_m, 0)
        loss_m = np.mean((a_m - y_np) ** 2)

        num_db[0, j] = (loss_p - loss_m) / (2 * eps)

    dW_err = np.max(np.abs(analytic_dW - num_dW))
    db_err = np.max(np.abs(analytic_db - num_db))

    print(f"  dW max abs error: {dW_err:.2e}  {'✓ PASS' if dW_err < 1e-3 else '✗ FAIL'}")
    print(f"  db max abs error: {db_err:.2e}  {'✓ PASS' if db_err < 1e-3 else '✗ FAIL'}")
    return dW_err < 1e-3 and db_err < 1e-3


# ─────────────────────────────────────────────────────────────
# 2. ANN Batch Training with BatchNorm
# ─────────────────────────────────────────────────────────────
def ann_batch_training():
    separator("ANN Batch Training (XOR, batch_size=4)")
    np.random.seed(0)

    # XOR dataset (replicated for meaningful batches)
    X_train = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    y_train = np.array([[0],[1],[1],[0]], dtype=np.float32)

    # Repeat to get more samples
    X_train = np.tile(X_train, (25, 1))  # 100 samples
    y_train = np.tile(y_train, (25, 1))

    model = Sequential([
        Input((2,)),
        Dense(2, 8, activation="relu"),
        BatchNorm1d(8),
        Dense(8, 4, activation="relu"),
        Dense(4, 1, activation="sigmoid"),
    ])
    model.summary()

    loss_fn = Loss().mse
    optimizer = Adam(model, lr=0.01)

    history = model.fit(
        X_train, y_train, optimizer, loss_fn,
        Epochs=100, batch_size=4, Loss_interval=20,
    )

    # Test predictions
    X_test = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32), is_leaf=True)
    pred = model.predict(X_test)
    print("\n  XOR Predictions:")
    for i in range(4):
        expected = [0, 1, 1, 0][i]
        actual = pred.value[i, 0]
        print(f"    {X_test.value[i]} → {actual:.4f}  (expected {expected})")

    return history


# ─────────────────────────────────────────────────────────────
# 3. CNN Batch Training with BatchNorm2d
# ─────────────────────────────────────────────────────────────
def cnn_batch_training():
    separator("CNN Batch Training (synthetic 8x8, batch_size=4)")
    np.random.seed(1)

    # Synthetic dataset: classify 8x8 images
    # Class 0: random noise, Class 1: checkerboard-ish pattern
    num_samples = 32
    X_data = []
    y_data = []
    for i in range(num_samples):
        if i % 2 == 0:
            img = np.random.randn(1, 8, 8).astype(np.float32) * 0.3
            y_data.append([1, 0])
        else:
            img = np.zeros((1, 8, 8), dtype=np.float32)
            img[0, ::2, ::2] = 1.0
            img[0, 1::2, 1::2] = 1.0
            img += np.random.randn(1, 8, 8).astype(np.float32) * 0.1
            y_data.append([0, 1])
        X_data.append(img)

    X_train = np.array(X_data)  # (32, 1, 8, 8)
    y_train = np.array(y_data, dtype=np.float32)  # (32, 2)

    model = Sequential([
        Input((1, 8, 8)),
        Conv2D(4, 1, (3, 3), activation="relu"),
        BatchNorm2d(4),
        MaxPool2D(pool_size=(2, 2), stride=2),
        Flatten(),
        Dense(4 * 3 * 3, 2, activation="softmax"),
    ])
    model.summary()

    loss_fn = Loss().categorical_cross_entropy
    optimizer = Adam(model, lr=0.005)

    history = model.fit(
        X_train, y_train, optimizer, loss_fn,
        Epochs=50, batch_size=4, Loss_interval=10,
    )

    return history


# ─────────────────────────────────────────────────────────────
# 4. Shape Consistency Test
# ─────────────────────────────────────────────────────────────
def shape_test():
    separator("Shape Consistency Test")
    np.random.seed(42)

    N = 4
    # Test Dense shapes
    x = Tensor(np.random.randn(N, 5).astype(np.float32), is_leaf=True)
    w = Tensor(np.random.randn(5, 3).astype(np.float32), is_leaf=True)
    b = Tensor(np.random.randn(1, 3).astype(np.float32), is_leaf=True)

    z = x.matmul(w) + b
    assert z.shape == (N, 3), f"Expected (4,3), got {z.shape}"
    print(f"  Dense forward:   x{x.shape} @ w{w.shape} + b{b.shape} → z{z.shape}  ✓")

    a = z.relu()
    assert a.shape == (N, 3), f"Expected (4,3), got {a.shape}"
    print(f"  ReLU:            {a.shape}  ✓")

    s = a.softmax(axis=-1)
    assert s.shape == (N, 3), f"Expected (4,3), got {s.shape}"
    print(f"  Softmax:         {s.shape}  ✓")

    # Test Conv2D shapes
    img = Tensor(np.random.randn(N, 1, 8, 8).astype(np.float32), is_leaf=True)
    filters = Tensor(np.random.randn(4, 1, 3, 3).astype(np.float32), is_leaf=True)
    conv_out = img.conv2d(filters, stride=1, padding=0)
    assert conv_out.shape == (N, 4, 6, 6), f"Expected (4,4,6,6), got {conv_out.shape}"
    print(f"  Conv2D:          img{img.shape} * W{filters.shape} → {conv_out.shape}  ✓")

    # MaxPool
    pool_out = conv_out.maxpool2d(kernelsize=(2, 2), stride=2)
    # (6-2)/2+1 = 3
    assert pool_out.shape == (N, 4, 3, 3), f"Expected (4,4,3,3), got {pool_out.shape}"
    print(f"  MaxPool2D(2,2):  {conv_out.shape} → {pool_out.shape}  ✓")

    # Flatten
    flat = pool_out.flatten()
    assert flat.shape == (N, 4 * 3 * 3), f"Expected (4,36), got {flat.shape}"
    print(f"  Flatten:         {pool_out.shape} → {flat.shape}  ✓")

    print("\n  All shape tests passed ✓")


# ─────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Seera Framework — Batch Support Demo")
    print("=" * 60)

    # 1. Shape tests
    shape_test()

    # 2. Gradient check
    grad_ok = gradient_check()

    # 3. ANN training
    ann_history = ann_batch_training()

    # 4. CNN training
    cnn_history = cnn_batch_training()

    separator("Summary")
    print(f"  Gradient check:  {'PASSED' if grad_ok else 'FAILED'}")
    print(f"  ANN final loss:  {ann_history[-1]:.6f}")
    print(f"  CNN final loss:  {cnn_history[-1]:.6f}")
    print(f"\n  All demos completed successfully! ✓")
