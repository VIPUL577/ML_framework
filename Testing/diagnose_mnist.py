"""
Diagnose MNIST loss=0 issue.
Trace exact forward pass through Flatten → Dense pipeline.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from cuTen import cuten
from Seera_init import tensor as Tensor, _is_gpu
from Seera_Engine import autograd4nn
from Seera import Input, Flatten, Dense, Sequential, Loss, Adam

np.random.seed(42)

# Fake "MNIST-like" data: (batch, 1, 28, 28)
batch = 4
X_np = np.random.randn(batch, 1, 28, 28).astype(np.float32) * 0.01  # pixel-like
y_np = np.zeros((batch, 10), dtype=np.float32)
for i in range(batch):
    y_np[i, i % 10] = 1.0

print("=" * 60)
print("MNIST-like pipeline diagnostic")
print("=" * 60)

# ── Step 1: Check input transfer ──
X_tensor = Tensor(X_np, is_leaf=True, device="cuda")
X_host = X_tensor.value.to_host_f32()
print(f"\n1. Input transfer:")
print(f"   Shape: {X_tensor.value.shape}")
print(f"   Max diff from numpy: {np.max(np.abs(X_host - X_np)):.10f}")
print(f"   X[:1,:1,:3,:3] = {X_host[0, 0, :3, :3]}")

# ── Step 2: Flatten ──
flat = Tensor.flatten(X_tensor)
flat_host = flat.value.to_host_f32()
flat_exp = X_np.reshape(batch, -1)
print(f"\n2. Flatten:")
print(f"   Shape: {flat.value.shape} (expected {flat_exp.shape})")
print(f"   Max diff: {np.max(np.abs(flat_host - flat_exp)):.10f}")
print(f"   flat[:2,:5] = {flat_host[:2, :5]}")
print(f"   exp[:2,:5]  = {flat_exp[:2, :5]}")

# ── Step 3: Manual Dense forward ──
W1 = np.random.randn(784, 128).astype(np.float32) * np.sqrt(6.0 / (784 + 128)) * 0.1
b1 = np.zeros((1, 128), dtype=np.float32)

W1_gpu = cuten(W1)
b1_gpu = cuten(b1)

# matmul
mm = flat.value.matmul(W1_gpu)
mm_host = mm.to_host_f32()
mm_exp = flat_exp @ W1
print(f"\n3. Matmul (flat @ W1):")
print(f"   Shape: {mm.shape}")
print(f"   Max diff: {np.max(np.abs(mm_host - mm_exp)):.6f}")
print(f"   mm max value: {np.max(np.abs(mm_host)):.6f}")

# ── Step 4: Full model forward ──
print(f"\n4. Full model forward:")
model = Sequential([
    Input((1, 28, 28)),
    Flatten(),
    Dense(784, 128, activation="relu"),
    Dense(128, 64, activation="relu"),
    Dense(64, 10, activation="softmax"),
], device="cuda")

X_batch = Tensor(X_np.copy(), is_leaf=True, device="cuda")
y_batch = Tensor(y_np.copy(), device="cuda")

pred = model.forward(X_batch)
pred_np = pred.value.to_host_f32()
print(f"   Pred shape: {pred.value.shape}")
print(f"   Pred[:2] = {pred_np[:2]}")
print(f"   Pred row sums: {pred_np.sum(axis=-1)}")

# ── Step 5: CCE loss ──
loss = Loss().categorical_cross_entropy(pred, y_batch)
loss_val = float(loss.value.to_host_f32().ravel()[0])
print(f"\n5. CCE loss: {loss_val:.6f}")
print(f"   Expected ~log(10) = {np.log(10):.6f}")

# ── Step 6: Backward + check gradients ──
model.zero_grad()
autograd4nn(loss)

dW1 = model.model[2].weights.node.cp
dW1_np = dW1.to_host_f32() if isinstance(dW1, cuten) else np.asarray(dW1)
print(f"\n6. Gradients:")
print(f"   dW1 shape: {dW1_np.shape}")
print(f"   dW1 max: {np.max(np.abs(dW1_np)):.8f}")
print(f"   dW1 has NaN: {np.any(np.isnan(dW1_np))}")

# ── Step 7: Single training step ──
print(f"\n7. Single training step:")
optimizer = Adam(model, lr=0.001)
optimizer.step()

# Second forward pass
X_batch2 = Tensor(X_np.copy(), is_leaf=True, device="cuda")
y_batch2 = Tensor(y_np.copy(), device="cuda")
pred2 = model.forward(X_batch2)
loss2 = Loss().categorical_cross_entropy(pred2, y_batch2)
loss2_val = float(loss2.value.to_host_f32().ravel()[0])
print(f"   Loss before: {loss_val:.6f}")
print(f"   Loss after:  {loss2_val:.6f}")
print(f"   Pred2[:2] = {pred2.value.to_host_f32()[:2]}")

# ── Step 8: Multi-epoch training ──
print(f"\n8. Multi-epoch training (5 epochs):")
np.random.seed(42)
model2 = Sequential([
    Input((1, 28, 28)),
    Flatten(),
    Dense(784, 128, activation="relu"),
    Dense(128, 64, activation="relu"),
    Dense(64, 10, activation="softmax"),
], device="cuda")
optimizer2 = Adam(model2, lr=0.001)
loss_fn = Loss()

for epoch in range(5):
    X_batch = Tensor(X_np.copy(), is_leaf=True, device="cuda")
    y_batch = Tensor(y_np.copy(), device="cuda")
    pred = model2.forward(X_batch)
    loss = loss_fn.categorical_cross_entropy(pred, y_batch)
    loss_val = float(loss.value.to_host_f32().ravel()[0])
    print(f"   Epoch {epoch}: loss={loss_val:.6f}, pred[:1]={pred.value.to_host_f32()[:1]}")
    model2.zero_grad()
    autograd4nn(loss)
    optimizer2.step()

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
