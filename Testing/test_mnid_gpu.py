from tensorflow.keras.datasets import mnist
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Seera_init import tensor as Tensor
import matplotlib.pyplot as plt
from Seera import (
    Input, Conv2D, MaxPool2D, Flatten, Dense,
    Sequential, Loss, Adam,
)
from Seera_Engine import autograd4nn

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train_ = np.expand_dims(X_train,axis=1)/255
y_train_ = np.zeros((60000,10))
for i in range (0,60000):
    y_train_[i,:] = np.eye(1,10,y_train[i])
# y_train_ = y_train_[:,:,np.newaxis]
X_test_ = np.expand_dims(X_test,axis=1)/255
print("=" * 60)
print("  MNIST GPU Test — Seera Framework (cuda)")
print("=" * 60)

print(f"Train: {X_train.shape}, Test: {y_test.shape}")

# ─── Build Model (device="cuda") ─────────────────────────
model = Sequential([
    Input((1, 28, 28)),
    Conv2D(8, 1, (3, 3), activation="relu", stride=1, zero_padding=1),
    MaxPool2D(pool_size=(2, 2), stride=2),
    Conv2D(16, 8, (3, 3), activation="relu", stride=1, zero_padding=1),
    MaxPool2D(pool_size=(2, 2), stride=2),
    Flatten(),
    Dense(16*7*7, 128, activation="relu"),
    Dense(128, 16, activation="relu"),
    
    Dense(16, 10, activation="softmax"),
], device="cuda")
model.summary()
# model = Sequential([
#     Input((1,28,28,)),
#     Flatten(),
#     Dense(784, 256, activation="relu"),
#     Dense(256, 128, activation="relu"),
#     Dense(128, 10, activation="softmax"),
# ], "cuda")

loss_fn = Loss()
optimizer = Adam(model, lr=1e-4)
# ─── Train ───────────────────────────────────────────────
# loss_fn = Loss()
# optimizer = Adam(model, lr=0.001)


idx = np.random.permutation(len(X_train_))
X, y = X_train_[idx], y_train_[idx]  # shuffle (assuming labels y)

batch_size = 16

loss_track = 0.0
for epoch in range(5):
    epoch_loss = 0.0
    n_batches = 0
    for i in range(0, 60000, batch_size):
        X_batch, y_batch = X[i:i+batch_size], y[i:i+batch_size]
        X_batch = Tensor(X_batch, is_leaf=True, device="cuda")
        y_batch = Tensor(y_batch, device="cuda")
        pred = model.forward(X_batch)
        loss = loss_fn.categorical_cross_entropy(pred, y_batch)
        # print(loss)
        loss_val = float(loss.value.to_host_f32().ravel()[0])
        epoch_loss += loss_val
        n_batches += 1

        model.zero_grad()
        autograd4nn(loss)
        optimizer.step()
    # exit()
        
    avg_loss = epoch_loss / n_batches
    print(f"EPOCH {epoch+1}/30: Loss: {avg_loss:.6f}")
# history = model.fit(
#     X_train_, y_train_,
#     Optimizer=optimizer,
#     Loss=loss_fn.categorical_cross_entropy,
#     Epochs=5,
#     batch_size=16,
#     Loss_interval=1,
# )

# ─── Evaluate ────────────────────────────────────────────
correct = 0
for i in range(len(X_test)):
    x = Tensor(X_test_[i:i+1], is_leaf=True, device="cuda")
    pred = model.predict(x)
    # bring to host for argmax
    pred_np = pred.value.to_host_f32()
    pred_label = np.argmax(pred_np)
    if pred_label == y_test[i]:
        correct += 1

accuracy = correct / len(X_test) * 100
print(f"\nTest Accuracy: {accuracy:.1f}% ({correct}/{len(X_test)})")
print("GPU test complete ✓")