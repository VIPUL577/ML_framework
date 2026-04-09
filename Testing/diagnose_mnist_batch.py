"""
Diagnose when MNIST loss goes to 0 during training.
Tracks loss per batch to find the exact failure point.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from tensorflow.keras.datasets import mnist

import gc
from Seera_init import tensor as Tensor
from Seera_Engine import autograd4nn
from Seera import Input, Flatten, Dense, Sequential, Loss, Adam,Conv2D, MaxPool2D

np.random.seed(42)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train_ = np.expand_dims(X_train,axis=1)/255
y_train_ = np.zeros((60000,10))
for i in range (0,60000):
    y_train_[i,:] = np.eye(1,10,y_train[i])
X_test_ = np.expand_dims(X_test,axis=1)/255

n_samples = 60000  # subset
batch_size = 16

model = Sequential([
    Input((1, 28, 28)),
    # Conv2D(8, 1, (3, 3), activation="relu", stride=1, zero_padding=1),
    # MaxPool2D(pool_size=(2, 2), stride=2),
    # Conv2D(16, 8, (3, 3), activation="relu", stride=1, zero_padding=1),
    # MaxPool2D(pool_size=(2, 2), stride=2),
    Flatten(),
    Dense(16*7*7, 32, activation="relu"),
    Dense(32, 16, activation="relu"),
    
    Dense(16, 10, activation="softmax"),
], device="cuda")

loss_fn = Loss()
optimizer = Adam(model, lr=1e-5)

n_batches = n_samples // batch_size
print(f"Samples: {n_samples}, Batch size: {batch_size}, Batches/epoch: {n_batches}")
print("=" * 70)

first_zero_batch = None
for epoch in range(2):
    epoch_loss = 0.0
    for b in range(n_batches):
        start = b * batch_size
        end = start + batch_size
        X_batch = Tensor(X_train_[start:end], is_leaf=True, device="cuda")
        y_batch = Tensor(y_train_[start:end], device="cuda")
        
        pred = model.forward(X_batch)
        loss = loss_fn.categorical_cross_entropy(pred, y_batch)
        loss_val = float(loss.value.to_host_f32().ravel()[0])
        epoch_loss += loss_val
        
        # Print every 10 batches, or if loss is suspicious
        if b < 5 or b % 10 == 0 or loss_val < 0.01:
            print(f"  Epoch {epoch} Batch {b:4d}: loss={loss_val:.6f}")
            if loss_val < 0.01 and first_zero_batch is None:
                first_zero_batch = (epoch, b)
                print(f"  >>> LOSS COLLAPSED at epoch {epoch}, batch {b} <<<")
        
        model.zero_grad()
        autograd4nn(loss)
        optimizer.step()
        
        # Force garbage collection periodically to free Python-side references
        # (won't free GPU memory since there's no __del__, but cleans up numpy)
        if b % 50 == 0:
            gc.collect()
    
    avg = epoch_loss / n_batches
    print(f"\n  Epoch {epoch} avg loss: {avg:.6f}\n")

if first_zero_batch:
    print(f"\nLoss first collapsed at: epoch {first_zero_batch[0]}, batch {first_zero_batch[1]}")
else:
    print("\nLoss never collapsed! ✓")
