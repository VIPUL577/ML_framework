"""
MNIST classification on GPU using Seera framework.
Same architecture as the CPU test but runs on CUDA via cuten tensors.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from Seera_init import tensor as Tensor
from cuTen import cuten
from Seera import (
    Input, Conv2D, MaxPool2D, Flatten, Dense,
    Sequential, Loss, Adam,
)
from Seera_Engine import autograd4nn

# ─────────────────────────────────────────────────────────────
# 1. Load MNIST data
# ─────────────────────────────────────────────────────────────
def load_mnist(num_train=500, num_test=100):
    """Load a small MNIST subset via sklearn."""
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist["data"].astype(np.float32), mnist["target"].astype(np.int32)

    # Normalize to [0, 1] and reshape to (N, 1, 28, 28)
    X = X / 255.0
    X = X.reshape(-1, 1, 28, 28)

    # One-hot encode labels (10 classes)
    num_classes = 10
    y_onehot = np.zeros((len(y), num_classes), dtype=np.float32)
    for i, label in enumerate(y):
        y_onehot[i, label] = 1.0

    X_train, y_train = X[:num_train], y_onehot[:num_train]
    X_test, y_test = X[60000:60000 + num_test], y_onehot[60000:60000 + num_test]
    y_test_labels = y[60000:60000 + num_test]

    return X_train, y_train, X_test, y_test, y_test_labels


def main():
    print("=" * 60)
    print("  MNIST GPU Test — Seera Framework (CUDA)")
    print("=" * 60)

    X_train, y_train, X_test, y_test, y_test_labels = load_mnist(
        num_train=20000, num_test=6000
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # ─── Build Model (device="cuda") ─────────────────────────
    model = Sequential([
        Input((1, 28, 28)),
        Conv2D(8, 1, (3, 3), activation="relu", stride=1, zero_padding=1),
        MaxPool2D(pool_size=(2, 2), stride=2),
        Conv2D(16, 8, (3, 3), activation="relu", stride=1, zero_padding=1),
        MaxPool2D(pool_size=(2, 2), stride=2),
        Flatten(),
        Dense(16 * 7 * 7, 64, activation="relu"),
        Dense(64, 10, activation="softmax"),
    ], device="cuda")
    model.summary()

    loss_fn = Loss()
    optimizer = Adam(model, lr=1e-4)
        # ─── Train ───────────────────────────────────────────────
    batch_size = 16
    epochs = 2  
    loss_track = 0.0
    print(type(model))
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, 20000, batch_size):
            x_batch, y_batch = X_train[i:i+batch_size], y_train[i:i+batch_size]
            X_batch = Tensor(x_batch, is_leaf=True, device="cuda")
            Y_batch = Tensor(y_batch, device="cuda")
            
            
            pred = model.forward(X_batch)
            loss = loss_fn.categorical_cross_entropy(pred, Y_batch)
            if(float(loss.value.to_host_f32().ravel()[0])==0):
                
                print(X_batch.value.to_host_f32().sum())
                print(Y_batch.value.to_host_f32())
            loss_val = float(loss.value.to_host_f32().ravel()[0])
            epoch_loss += loss_val
            n_batches += 1
            # print(Y_batch.sum())
            
            del(X_batch)
            del(Y_batch)

            model.zero_grad()
            autograd4nn(loss)
            optimizer.step()
        
        
        avg_loss = epoch_loss / n_batches
        print(f"EPOCH {epoch+1}/{epochs}: Loss: {avg_loss:.10f}")

    # ─── Evaluate ────────────────────────────────────────────
    correct = 0
    for i in range(len(X_test)):
        x = Tensor(X_test[i:i+1], is_leaf=True, device="cuda")
        pred = model.predict(x)
        # bring to host for argmax
        pred_np = pred.value.to_host_f32()
        pred_label = np.argmax(pred_np)
        if pred_label == y_test_labels[i]:
            correct += 1

    accuracy = correct / len(X_test) * 100
    print(f"\nTest Accuracy: {accuracy:.1f}% ({correct}/{len(X_test)})")
    print("GPU test complete ✓")


if __name__ == "__main__":
    main()
