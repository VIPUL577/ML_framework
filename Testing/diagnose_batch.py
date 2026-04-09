"""
Reproduce the model.fit loss=0 bug:
EXACT same setup as deep_debug Test G but with step-by-step tracing.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from cuTen import cuten
from Seera_init import tensor as Tensor, _is_gpu
from Seera_Engine import autograd4nn
from Seera import Input, Dense, Sequential, Loss, SGD

np.random.seed(42)
x_data = np.random.randn(32, 3).astype(np.float32)
y_data = np.zeros((32, 2), dtype=np.float32)
for i in range(32):
    y_data[i, i % 2] = 1.0

# ── Test 1: model.fit with batch_size=1 ──
print("=" * 60)
print("Test 1: batch_size=1")
print("=" * 60)
np.random.seed(99)
model1 = Sequential([Input((3,)), Dense(3, 2, activation="softmax")], "cuda")
opt1 = SGD(model1, lr=0.05)
h1 = model1.fit(x_data, y_data, Optimizer=opt1,
                Loss=Loss().categorical_cross_entropy,
                Epochs=3, batch_size=1, Loss_interval=1)
print(f"  history: {h1}")

# ── Test 2: model.fit with batch_size=8 ──
print("\n" + "=" * 60)
print("Test 2: batch_size=8")
print("=" * 60)
np.random.seed(99)
model2 = Sequential([Input((3,)), Dense(3, 2, activation="softmax")], "cuda")
opt2 = SGD(model2, lr=0.05)
h2 = model2.fit(x_data, y_data, Optimizer=opt2,
                Loss=Loss().categorical_cross_entropy,
                Epochs=3, batch_size=8, Loss_interval=1)
print(f"  history: {h2}")

# ── Test 3: model.fit with batch_size=32 (full batch) ──
print("\n" + "=" * 60)
print("Test 3: batch_size=32 (full batch)")
print("=" * 60)
np.random.seed(99)
model3 = Sequential([Input((3,)), Dense(3, 2, activation="softmax")], "cuda")
opt3 = SGD(model3, lr=0.05)
h3 = model3.fit(x_data, y_data, Optimizer=opt3,
                Loss=Loss().categorical_cross_entropy,
                Epochs=3, batch_size=32, Loss_interval=1)
print(f"  history: {h3}")

# ── Test 4: Manual training loop with batch_size=8 ──
print("\n" + "=" * 60)
print("Test 4: MANUAL loop batch_size=8 (bypass model.fit)")
print("=" * 60)
np.random.seed(99)
model4 = Sequential([Input((3,)), Dense(3, 2, activation="softmax")], "cuda")
opt4 = SGD(model4, lr=0.05)
loss_fn = Loss()

for epoch in range(3):
    epoch_loss = 0.0
    n_batches = 0
    perm = np.random.permutation(32)
    X_shuf = x_data[perm]
    y_shuf = y_data[perm]
    
    for start in range(0, 32, 8):
        end = min(start + 8, 32)
        X_batch = Tensor(X_shuf[start:end], is_leaf=True, device="cuda")
        y_batch = Tensor(y_shuf[start:end], device="cuda")
        
        pred = model4.forward(X_batch)
        
        # Check intermediate values
        pred_np = pred.value.to_host_f32()
        
        loss = loss_fn.categorical_cross_entropy(pred, y_batch)
        loss_val_raw = loss.value.to_host_f32()
        
        if start == 0 and epoch == 0:
            print(f"  First batch pred[:2] = {pred_np[:2]}")
            print(f"  First batch pred sum = {pred_np.sum(axis=-1)[:4]}")
            print(f"  First batch loss raw = {loss_val_raw}")
            print(f"  First batch loss shape = {loss.value.shape}")
        
        # Extract scalar
        if len(loss.value.shape) > 0 and loss.value.size > 1:
            loss = loss.mean()
        batch_loss = float(loss.value.to_host_f32().ravel()[0])
        epoch_loss += batch_loss
        n_batches += 1
        
        model4.zero_grad()
        autograd4nn(loss)
        opt4.step()
    
    avg = epoch_loss / n_batches
    print(f"  Epoch {epoch+1}: loss = {avg:.6f}")

# ── Test 5: Manual loop batch_size=8 with EXPLICIT CCE ──
print("\n" + "=" * 60)
print("Test 5: Manual loop batch_size=8 with EXPLICIT CCE steps")
print("=" * 60)
np.random.seed(99)
model5 = Sequential([Input((3,)), Dense(3, 2, activation="softmax")], "cuda")
opt5 = SGD(model5, lr=0.05)

for epoch in range(3):
    epoch_loss = 0.0
    n_batches = 0
    perm = np.random.permutation(32)
    X_shuf = x_data[perm]
    y_shuf = y_data[perm]
    
    for start in range(0, 32, 8):
        end = min(start + 8, 32)
        X_batch = Tensor(X_shuf[start:end], is_leaf=True, device="cuda")
        y_batch = Tensor(y_shuf[start:end], device="cuda")
        
        pred = model5.forward(X_batch)
        
        # Manual CCE steps with tracing
        eps = 1e-15
        step1 = pred + eps                # pred + epsilon
        step2 = step1.log()               # log(pred + eps)
        step3 = -y_batch * step2          # -y * log(pred + eps) — NOT y_batch
        step4 = step3.sum(axis=-1)        # sum over classes
        step5 = step4.mean()              # mean over samples
        
        if start == 0 and epoch == 0:
            print(f"  pred[:2] = {pred.value.to_host_f32()[:2]}")
            print(f"  step1 (pred+eps)[:2] = {step1.value.to_host_f32()[:2]}")
            print(f"  step2 (log)[:2] = {step2.value.to_host_f32()[:2]}")
            print(f"  step3 (-y*log)[:2] = {step3.value.to_host_f32()[:2]}")
            print(f"  step4 (sum axis=-1) = {step4.value.to_host_f32()}")
            print(f"  step5 (mean) = {step5.value.to_host_f32()}")
        
        batch_loss = float(step5.value.to_host_f32().ravel()[0])
        epoch_loss += batch_loss
        n_batches += 1
        
        model5.zero_grad()
        autograd4nn(step5)
        opt5.step()
    
    avg = epoch_loss / n_batches
    print(f"  Epoch {epoch+1}: loss = {avg:.6f}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
