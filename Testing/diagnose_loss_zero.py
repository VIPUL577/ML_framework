#!/usr/bin/env python3
"""
Diagnose: Loss drops to 0 after epoch 0.
We'll do 3 forward passes on the same batch and print:
  - weights before/after each step
  - gradients after backward
  - loss value
  - softmax output (check for NaN/Inf/all-same)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cuTen import cuten
from Seera_init import tensor as Tensor, _where, _is_gpu
from Seera_Engine import autograd4nn
from Seera import Dense, Input, Sequential, Flatten, Loss, Adam

np.random.seed(42)

# Small data (like MNIST but tiny)
N = 4
X_np = np.random.randn(N, 1, 28, 28).astype(np.float32) * 0.01  # small values like normalized MNIST
labels = np.array([3, 7, 1, 0])
Y_np = np.zeros((N, 10), dtype=np.float32)
Y_np[np.arange(N), labels] = 1.0

model = Sequential([
    Input((1, 28, 28)),
    Flatten(),
    Dense(28*28, 128, activation="relu"),
    Dense(128, 64, activation="relu"),
    Dense(64, 10, activation="softmax"),
], device="cuda")

loss_fn = Loss()
optimizer = Adam(model, lr=0.001)

def to_np(val):
    if isinstance(val, cuten):
        return val.to_host_f32()
    if hasattr(val, 'value'):
        v = val.value
        if isinstance(v, cuten):
            return v.to_host_f32()
        return np.array(v)
    return np.array(val)

for step in range(5):
    print(f"\n{'='*60}")
    print(f"  STEP {step}")
    print(f"{'='*60}")
    
    X_t = Tensor(X_np.copy(), is_leaf=True, device="cuda")
    Y_t = Tensor(Y_np.copy(), device="cuda")
    
    # Forward
    pred = model.forward(X_t)
    pred_np = to_np(pred)
    print(f"  pred shape: {pred_np.shape}")
    print(f"  pred[0] (softmax output): {pred_np[0]}")
    print(f"  pred sum per sample: {pred_np.sum(axis=1)}")
    print(f"  pred max: {pred_np.max()}, min: {pred_np.min()}")
    print(f"  any NaN: {np.any(np.isnan(pred_np))}, any Inf: {np.any(np.isinf(pred_np))}")
    
    # Loss
    loss = loss_fn.categorical_cross_entropy(pred, Y_t)
    loss_val = float(to_np(loss).ravel()[0])
    print(f"  loss value: {loss_val}")
    print(f"  loss any NaN: {np.any(np.isnan(to_np(loss)))}")
    
    # Backward
    model.zero_grad()
    autograd4nn(loss)
    
    # Check gradients
    for i, layer in enumerate(model.model):
        if hasattr(layer, 'weights') and isinstance(layer.weights, Tensor):
            w_grad = to_np(layer.weights.node.cp)
            b_grad = to_np(layer.bais.node.cp)
            w_val = to_np(layer.weights.value)
            print(f"  Layer {i}: |W|_max={np.max(np.abs(w_val)):.6f}, "
                  f"|dW|_max={np.max(np.abs(w_grad)):.6f}, "
                  f"|dB|_max={np.max(np.abs(b_grad)):.6f}, "
                  f"dW_any_nan={np.any(np.isnan(w_grad))}, "
                  f"W_any_nan={np.any(np.isnan(w_val))}")
    
    # Optimizer step
    optimizer.step()
    
    # Check weights after step
    for i, layer in enumerate(model.model):
        if hasattr(layer, 'weights') and isinstance(layer.weights, Tensor):
            w_val = to_np(layer.weights.value)
            print(f"  Layer {i} after step: |W|_max={np.max(np.abs(w_val)):.6f}, "
                  f"W_any_nan={np.any(np.isnan(w_val))}, "
                  f"W_any_inf={np.any(np.isinf(w_val))}")

print("\nDone.")
