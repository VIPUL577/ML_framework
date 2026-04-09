"""
Diagnostic script to trace why loss is always zero.
Tests each component individually and prints intermediate values.
"""
import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import numpy as np
from Seera_init import tensor as Tensor
from Seera_Engine import autograd4nn
from Seera import Loss

np.random.seed(42)

print("=" * 60)
print("DIAGNOSTIC: Tracing zero-loss issue")
print("=" * 60)

# ── Test 1: Simple scalar operations ──
print("\n--- Test 1: Basic tensor ops ---")
a = Tensor(np.array([[1.0, 2.0, 3.0]]), is_leaf=True)
b = Tensor(np.array([[4.0, 5.0, 6.0]]), is_leaf=True)
c = a * b
print(f"a * b = {c.value}")
d = c.sum()
print(f"sum(a*b) = {d.value}")
autograd4nn(d)
print(f"grad a = {a.node.cp}")
print(f"grad b = {b.node.cp}")
print(f"Expected grad a = [4, 5, 6], grad b = [1, 2, 3]")

# ── Test 2: Matmul gradient ──
print("\n--- Test 2: Matmul gradient ---")
np.random.seed(42)
x = Tensor(np.random.randn(2, 3).astype(np.float32), is_leaf=True)
W = Tensor(np.random.randn(3, 2).astype(np.float32), is_leaf=True)
y = x.matmul(W)
loss = y.sum()
print(f"x shape: {x.shape}, W shape: {W.shape}")
print(f"y = x @ W: {y.value}")
print(f"loss = sum(y): {loss.value}")
autograd4nn(loss)
print(f"grad x:\n{x.node.cp}")
print(f"grad W:\n{W.node.cp}")
# Expected: grad_x = ones(2,2) @ W.T, grad_W = x.T @ ones(2,2)
expected_grad_x = np.ones((2, 2)) @ W.value.T
expected_grad_W = x.value.T @ np.ones((2, 2))
print(f"Expected grad x:\n{expected_grad_x}")
print(f"Expected grad W:\n{expected_grad_W}")

# ── Test 3: Log gradient ──
print("\n--- Test 3: Log gradient ---")
x = Tensor(np.array([[0.5, 0.8, 0.2]]), is_leaf=True)
y = x.log()
loss = y.sum()
print(f"log({x.value}) = {y.value}")
print(f"Expected: {np.log(x.value)}")
autograd4nn(loss)
print(f"grad x = {x.node.cp}")
print(f"Expected grad = {1.0 / x.value}")

# ── Test 4: Softmax + log ──
print("\n--- Test 4: Softmax + log ---")
logits = Tensor(np.array([[1.0, 2.0, 3.0]]), is_leaf=True)
s = logits.softmax()
print(f"softmax({logits.value}) = {s.value}")
print(f"Expected: {np.exp([1,2,3]) / np.sum(np.exp([1,2,3]))}")

# ── Test 5: CCE loss ──
print("\n--- Test 5: Categorical Cross Entropy ---")
np.random.seed(42)
logits = Tensor(np.array([[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]]), is_leaf=True)
s = logits.softmax()
print(f"softmax output:\n{s.value}")

y_true = Tensor(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))

loss_fn = Loss()
# Manually trace the loss computation
epsilon = 1e-15
print(f"\ny_pred (softmax): {s.value}")
print(f"y_true: {y_true.value}")

# Step by step
step1 = s + epsilon
print(f"\nstep1 (y_pred + eps): {step1.value}")

step2 = step1.log()
print(f"step2 (log(y_pred + eps)): {step2.value}")

step3 = -y_true  # This calls y_true * (-1)
print(f"step3 (-y_true): {step3.value}")

step4 = step3 * step2
print(f"step4 (-y * log(y_pred+eps)): {step4.value}")

step5 = step4.sum(axis=-1)
print(f"step5 (sum along classes): {step5.value}")

step6 = step5.mean()
print(f"step6 (mean over batch): {step6.value}")

print(f"\nExpected loss:")
s_np = s.value
expected = -np.sum(y_true.value * np.log(s_np + epsilon), axis=-1)
print(f"  per-sample: {expected}")
print(f"  mean: {np.mean(expected)}")

# Now test backward
print("\n--- Backward through CCE ---")
autograd4nn(step6)
print(f"grad logits:\n{logits.node.cp}")

# PyTorch reference
try:
    import torch
    import torch.nn.functional as F
    
    logits_t = torch.tensor([[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]], requires_grad=True)
    s_t = F.softmax(logits_t, dim=-1)
    y_true_t = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    loss_t = F.cross_entropy(logits_t, y_true_t.argmax(dim=-1))
    loss_t.backward()
    print(f"\nPyTorch reference:")
    print(f"  loss: {loss_t.item()}")
    print(f"  grad logits:\n  {logits_t.grad}")
    
    # Also manual CCE in torch to match our computation
    logits_t2 = torch.tensor([[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]], requires_grad=True)
    s_t2 = F.softmax(logits_t2, dim=-1)
    loss_manual = (-y_true_t * torch.log(s_t2 + 1e-15)).sum(dim=-1).mean()
    loss_manual.backward()
    print(f"\nPyTorch manual CCE:")
    print(f"  loss: {loss_manual.item()}")
    print(f"  grad logits:\n  {logits_t2.grad}")
except ImportError:
    print("  (PyTorch not available for comparison)")

# ── Test 6: Full Dense forward-backward ──
print("\n\n--- Test 6: Dense layer forward-backward ---")
np.random.seed(42)
W_init = np.random.randn(3, 2).astype(np.float32) * 0.1
b_init = np.zeros((1, 2), dtype=np.float32)
x_data = np.array([[0.5, 1.0, -0.5]], dtype=np.float32)
y_data = np.array([[1.0, 0.0]], dtype=np.float32)

# Seera
x = Tensor(x_data, is_leaf=True)
W = Tensor(W_init.copy(), is_leaf=True)
b = Tensor(b_init.copy(), is_leaf=True)

z = x.matmul(W) + b
print(f"z (before activation): {z.value}")

out = z.sigmoid()
print(f"sigmoid(z): {out.value}")

loss = ((out - y_data) ** 2).mean()
print(f"MSE loss: {loss.value}")

autograd4nn(loss)
print(f"grad W:\n{W.node.cp}")
print(f"grad b:\n{b.node.cp}")

try:
    import torch
    x_t = torch.tensor(x_data, requires_grad=True)
    W_t = torch.tensor(W_init.copy(), requires_grad=True)
    b_t = torch.tensor(b_init.copy(), requires_grad=True)
    z_t = x_t @ W_t + b_t
    out_t = torch.sigmoid(z_t)
    loss_t = ((out_t - torch.tensor(y_data)) ** 2).mean()
    loss_t.backward()
    print(f"\nPyTorch reference:")
    print(f"  loss: {loss_t.item()}")
    print(f"  grad W:\n  {W_t.grad}")
    print(f"  grad b:\n  {b_t.grad}")
    print(f"  grad x:\n  {x_t.grad}")
    
    print(f"\nDiff W grad: {np.max(np.abs(W.node.cp - W_t.grad.numpy()))}")
    print(f"Diff b grad: {np.max(np.abs(b.node.cp - b_t.grad.numpy()))}")
except ImportError:
    print("  (PyTorch not available)")

# ── Test 7: Full training loop simulation ──
print("\n\n--- Test 7: Mini training simulation ---")
from Seera import Input, Dense, Sequential, Loss, SGD

np.random.seed(42)
x_data = np.random.randn(4, 3).astype(np.float32)
y_data = np.zeros((4, 2), dtype=np.float32)
y_data[0, 0] = 1.0
y_data[1, 1] = 1.0
y_data[2, 0] = 1.0
y_data[3, 1] = 1.0

model = Sequential([
    Input((3,)),
    Dense(3, 2, activation="softmax"),
],"cuda")

loss_fn = Loss()
optimizer = SGD(model, lr=0.01)

for epoch in range(5):
    X_batch = Tensor(x_data, is_leaf=True)
    y_batch = Tensor(y_data)
    
    ypred = model.forward(X_batch)
    print(f"\nEpoch {epoch}: ypred = {ypred.value}")
    
    loss = loss_fn.categorical_cross_entropy(ypred, y_batch)
    print(f"  loss (per-sample): {loss.value}")
    
    if loss.value.ndim > 0 and loss.value.size > 1:
        loss = loss.mean()
    print(f"  loss (scalar): {loss.value}")
    
    model.zero_grad()
    autograd4nn(loss)
    
    W, b = model.model[1].get_weights()
    print(f"  W grad:\n  {W.node.cp}")
    print(f"  b grad:\n  {b.node.cp}")
    
    optimizer.step()
    W2, b2 = model.model[1].get_weights()
    print(f"  W after update:\n  {W2.value}")

print("\nDiagnostic complete.")
