"""
MNIST classification on CPU using Seera framework.
Downloads MNIST via sklearn, trains a small CNN for a few epochs.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from Seera_init import tensor as Tensor
from Seera import (
    Input, Conv2D, MaxPool2D, Flatten, Dense,
    Sequential, Loss, Adam,
)

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
    print("  MNIST CPU Test — Seera Framework")
    print("=" * 60)

    X_train, y_train, X_test, y_test, y_test_labels = load_mnist(
        num_train=200, num_test=50
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # ─── Build Model ─────────────────────────────────────────
    model = Sequential([
        Input((1, 28, 28)),
        Conv2D(8, 1, (3, 3), activation="relu", stride=1, zero_padding=1),
        MaxPool2D(pool_size=(2, 2), stride=2),
        Conv2D(16, 8, (3, 3), activation="relu", stride=1, zero_padding=1),
        MaxPool2D(pool_size=(2, 2), stride=2),
        Flatten(),
        Dense(16 * 7 * 7, 64, activation="relu"),
        Dense(64, 10, activation="softmax"),
    ])
    model.summary()

    # ─── Train ───────────────────────────────────────────────
    loss_fn = Loss()
    optimizer = Adam(model, lr=0.001)

    history = model.fit(
        X_train, y_train,
        Optimizer=optimizer,
        Loss=loss_fn.categorical_cross_entropy,
        Epochs=5,
        batch_size=16,
        Loss_interval=1,
    )

    # ─── Evaluate ────────────────────────────────────────────
    correct = 0
    for i in range(len(X_test)):
        x = Tensor(X_test[i:i+1], is_leaf=True)
        pred = model.predict(x)
        pred_label = np.argmax(pred.value)
        if pred_label == y_test_labels[i]:
            correct += 1

    accuracy = correct / len(X_test) * 100
    print(f"\nTest Accuracy: {accuracy:.1f}% ({correct}/{len(X_test)})")
    print("CPU test complete ✓")


if __name__ == "__main__":
    main()
