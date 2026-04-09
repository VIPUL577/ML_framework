import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import numpy as np

# ========================
# Device (CUDA)
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========================
# Load MNIST from TensorFlow
# ========================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize + reshape
x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32) / 255.0

# Flatten (60000, 28, 28) → (60000, 784)
x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

# Convert to PyTorch tensors
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train, dtype=torch.long)

x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test, dtype=torch.long)

# Move to GPU ONCE
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test  = x_test.to(device)
y_test  = y_test.to(device)

# ========================
# Model Definition
# ========================
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)  # logits (no softmax here)

model = ANN().to(device)

# ========================
# Loss + Optimizer
# ========================
criterion = nn.CrossEntropyLoss()  # includes softmax internally
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ========================
# Training
# ========================
batch_size = 256
epochs = 5

N = x_train.shape[0]

for epoch in range(epochs):
    perm = torch.randperm(N, device=device)

    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]

        xb = x_train[idx]
        yb = y_train[idx]

        # Forward
        outputs = model(xb)
        loss = criterion(outputs, yb)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ========================
# Evaluation
# ========================
with torch.no_grad():
    outputs = model(x_test)
    preds = torch.argmax(outputs, dim=1)
    acc = (preds == y_test).float().mean()

print("Test Accuracy:", acc.item())