"""Debug script to isolate GPU crash."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from Seera_init import tensor as Tensor
from cuTen import cuten
from Seera import (
    Input, Dense, Flatten,
    Sequential, Loss, Adam,
)

# ── Simple Dense-only model ──
print("Building model...")
model = Sequential([
    Input((4,)),
    Dense(4, 3, activation="relu"),
    Dense(3, 2, activation="softmax"),
], device="cuda")
model.summary()

# Check weights are on GPU
for layer in model.model:
    if hasattr(layer, 'weights') and hasattr(layer.weights, 'value'):
        val = layer.weights.value
        print(f"  {layer}: weights type={type(val).__name__}, shape={val.shape if hasattr(val,'shape') else 'N/A'}")

# ── Tiny data ──
X = np.random.randn(4, 4).astype(np.float32)
y = np.array([[1,0],[0,1],[1,0],[0,1]], dtype=np.float32)

print("\nForward pass...")
X_t = Tensor(X, is_leaf=True, device="cuda")
y_t = Tensor(y, device="cuda")
ypred = model.forward(X_t)
print(f"  ypred shape: {ypred.value.shape}")
print(f"  ypred host: {ypred.value.to_host_f32()}")

print("\nComputing loss...")
loss_fn = Loss()
# manually: y * log(ypred + eps), sum, mean
epsilon = 1e-15
logp = (ypred + epsilon).log()
print(f"  log(ypred+eps) shape: {logp.value.shape}")
neg_y_logp = -y_t * logp
print(f"  -y*log shape: {neg_y_logp.value.shape}")
s = neg_y_logp.sum(axis=-1)
print(f"  sum(axis=-1) shape: {s.value.shape}")
m = s.mean()
print(f"  mean() shape: {m.value.shape}")
print(f"  loss value: {m.value.to_host_f32()}")

print("\nDone — no crash!")
