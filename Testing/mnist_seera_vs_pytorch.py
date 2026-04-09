"""
MNIST: Seera CPU vs Seera GPU vs PyTorch — 3-way gradient comparison
====================================================================
Runs ONE batch through all three backends with identical weights.
Diagnoses exactly where GPU gradients diverge.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from Seera_init import tensor as Tensor, _is_gpu
from Seera_Engine import autograd4nn
from Seera import Input, Flatten, Dense, Sequential, Loss, SGD
from cuTen import cuten

# ── Helpers ──────────────────────────────────────────────────
PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"

def to_np(x):
    if isinstance(x, cuten):   return x.to_host_f32()
    if isinstance(x, Tensor):
        v = x.value
        return v.to_host_f32() if isinstance(v, cuten) else np.array(v)
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)

def compare(name, a, b, atol=1e-3):
    a_np, b_np = to_np(a).ravel(), to_np(b).ravel()
    ad = float(np.max(np.abs(a_np - b_np)))
    print(f"  {PASS if ad < atol else FAIL}  {name:50s}  Δ={ad:.6e}")
    return ad < atol

def stats(name, x):
    a = to_np(x)
    print(f"  │  {name:25s}  min={a.min():.6e}  max={a.max():.6e}  "
          f"absmax={np.max(np.abs(a)):.6e}  nan={np.any(np.isnan(a))}")

def section(title):
    print(f"\n{'═'*72}\n  {title}\n{'═'*72}")


# ═══════════════════════════════════════════════════════════════
#  1. Data
# ═══════════════════════════════════════════════════════════════
section("1. Loading MNIST")
mnist = datasets.MNIST(root='./data', train=True, download=True,
                       transform=transforms.ToTensor())

BATCH = 32
LR = 0.01

X_np = mnist.data[:BATCH].numpy().astype(np.float32).reshape(BATCH, 1, 28, 28) / 255.0
y_idx = mnist.targets[:BATCH].numpy()
y_oh  = np.zeros((BATCH, 10), dtype=np.float32)
y_oh[np.arange(BATCH), y_idx] = 1.0
print(f"  batch={BATCH}  X={X_np.shape}  y={y_oh.shape}")


# ═══════════════════════════════════════════════════════════════
#  2. Shared weights
# ═══════════════════════════════════════════════════════════════
section("2. Shared initial weights")
np.random.seed(42)
W1 = (np.random.randn(784, 128) * np.sqrt(2.0/784)).astype(np.float32)
b1 = np.zeros((1, 128), dtype=np.float32)
W2 = (np.random.randn(128, 10) * np.sqrt(2.0/128)).astype(np.float32)
b2 = np.zeros((1, 10), dtype=np.float32)
print(f"  W1 {W1.shape}  W2 {W2.shape}")


# ═══════════════════════════════════════════════════════════════
#  3. Build three models
# ═══════════════════════════════════════════════════════════════
section("3. Building models")

# ── Seera CPU ──
cpu_model = Sequential([
    Input((1,28,28)), Flatten(),
    Dense(784, 128, activation="relu"),
    Dense(128, 10, activation="softmax"),
], device="cpu")
cpu_model.model[2].set_weights(W1.copy(), b1.copy())
cpu_model.model[3].set_weights(W2.copy(), b2.copy())
print("  Seera CPU ✓")

# ── Seera GPU ──
gpu_model = Sequential([
    Input((1,28,28)), Flatten(),
    Dense(784, 128, activation="relu"),
    Dense(128, 10, activation="softmax"),
], device="cuda")
gpu_model.model[2].set_weights(
    Tensor(W1.copy(), is_leaf=True, device="cuda"),
    Tensor(b1.copy(), is_leaf=True, device="cuda"),
)
gpu_model.model[3].set_weights(
    Tensor(W2.copy(), is_leaf=True, device="cuda"),
    Tensor(b2.copy(), is_leaf=True, device="cuda"),
)
print("  Seera GPU ✓")

# ── PyTorch GPU ──
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(self.flat(x))))

pt_model = Net().cuda()
with torch.no_grad():
    pt_model.fc1.weight.copy_(torch.from_numpy(W1.T.copy()))
    pt_model.fc1.bias.copy_(torch.from_numpy(b1.ravel().copy()))
    pt_model.fc2.weight.copy_(torch.from_numpy(W2.T.copy()))
    pt_model.fc2.bias.copy_(torch.from_numpy(b2.ravel().copy()))
print("  PyTorch GPU ✓")


# ═══════════════════════════════════════════════════════════════
#  4. Verify initial weights match
# ═══════════════════════════════════════════════════════════════
section("4. Weight parity check")
compare("W1: CPU vs GPU",     cpu_model.model[2].weights.value, gpu_model.model[2].weights.value)
compare("W2: CPU vs GPU",     cpu_model.model[3].weights.value, gpu_model.model[3].weights.value)
compare("W1: CPU vs PyTorch", cpu_model.model[2].weights.value, pt_model.fc1.weight.data.cpu().T)
compare("W2: CPU vs PyTorch", cpu_model.model[3].weights.value, pt_model.fc2.weight.data.cpu().T)


# ═══════════════════════════════════════════════════════════════
#  5. Snapshot GPU W2 before forward
# ═══════════════════════════════════════════════════════════════
gpu_W2_snapshot_pre_fwd = to_np(gpu_model.model[3].weights.value).copy()


# ═══════════════════════════════════════════════════════════════
#  6A. SEERA CPU: forward + backward
# ═══════════════════════════════════════════════════════════════
section("5A. Seera CPU forward + backward")

X_cpu = Tensor(X_np.copy(), is_leaf=True)
y_cpu = Tensor(y_oh.copy())
pred_cpu = cpu_model.forward(X_cpu)

# Manual CCE (same as PyTorch: -y*log(p+eps) summed over classes, mean over batch)
eps_cpu = Tensor(np.full(pred_cpu.shape, 1e-15, dtype=np.float32))
loss_cpu = ((y_cpu * (pred_cpu + eps_cpu).log()) * (-1)).sum(axis=-1).mean()
loss_cpu_val = float(loss_cpu.value)

cpu_model.zero_grad()
autograd4nn(loss_cpu)

cpu_gW1 = cpu_model.model[2].weights.node.cp.copy()
cpu_gb1 = cpu_model.model[2].bais.node.cp.copy()
cpu_gW2 = cpu_model.model[3].weights.node.cp.copy()
cpu_gb2 = cpu_model.model[3].bais.node.cp.copy()

print(f"  Loss = {loss_cpu_val:.6f}")
stats("∇W1", cpu_gW1)
stats("∇b1", cpu_gb1)
stats("∇W2", cpu_gW2)
stats("∇b2", cpu_gb2)


# ═══════════════════════════════════════════════════════════════
#  6B. SEERA GPU: forward + backward
# ═══════════════════════════════════════════════════════════════
section("5B. Seera GPU forward + backward")

# Check W2 hasn't changed yet
gpu_W2_snapshot_pre_fwd2 = to_np(gpu_model.model[3].weights.value).copy()
compare("W2 unchanged before GPU fwd", gpu_W2_snapshot_pre_fwd, gpu_W2_snapshot_pre_fwd2)

X_gpu = Tensor(X_np.copy(), is_leaf=True, device="cuda")
y_gpu = Tensor(y_oh.copy(), device="cuda")
pred_gpu = gpu_model.forward(X_gpu)

# Check W2 right after forward (before backward)
gpu_W2_after_fwd = to_np(gpu_model.model[3].weights.value).copy()
compare("W2 unchanged after GPU fwd", gpu_W2_snapshot_pre_fwd, gpu_W2_after_fwd)

eps_gpu = Tensor(np.full(pred_gpu.value.shape, 1e-15, dtype=np.float32), device="cuda")
loss_gpu = ((y_gpu * (pred_gpu + eps_gpu).log()) * (-1)).sum(axis=-1).mean()
loss_gpu_val = float(to_np(loss_gpu).ravel()[0])

gpu_model.zero_grad()

# Check W2 right after zero_grad (before backward)
gpu_W2_after_zero = to_np(gpu_model.model[3].weights.value).copy()
compare("W2 unchanged after zero_grad", gpu_W2_snapshot_pre_fwd, gpu_W2_after_zero)

autograd4nn(loss_gpu)

# Check W2 right after backward
gpu_W2_after_bwd = to_np(gpu_model.model[3].weights.value).copy()
compare("W2 unchanged after backward", gpu_W2_snapshot_pre_fwd, gpu_W2_after_bwd)

gpu_gW1 = to_np(gpu_model.model[2].weights.node.cp).copy()
gpu_gb1 = to_np(gpu_model.model[2].bais.node.cp).copy()
gpu_gW2 = to_np(gpu_model.model[3].weights.node.cp).copy()
gpu_gb2 = to_np(gpu_model.model[3].bais.node.cp).copy()

print(f"  Loss = {loss_gpu_val:.6f}")
stats("∇W1", gpu_gW1)
stats("∇b1", gpu_gb1)
stats("∇W2", gpu_gW2)
stats("∇b2", gpu_gb2)


# ═══════════════════════════════════════════════════════════════
#  6C. PYTORCH: forward + backward
# ═══════════════════════════════════════════════════════════════
section("5C. PyTorch forward + backward")

X_pt = torch.from_numpy(X_np.copy()).cuda()
y_oh_pt = torch.from_numpy(y_oh.copy()).cuda()
pt_optim = torch.optim.SGD(pt_model.parameters(), lr=LR)
pt_optim.zero_grad()

logits_pt = pt_model(X_pt)
sm_pt = F.softmax(logits_pt, dim=-1)
loss_pt = (-y_oh_pt * torch.log(sm_pt + 1e-15)).sum(dim=-1).mean()
loss_pt.backward()
loss_pt_val = loss_pt.item()

pt_gW1 = pt_model.fc1.weight.grad.cpu().numpy().T.copy()
pt_gb1 = pt_model.fc1.bias.grad.cpu().numpy().reshape(1,-1).copy()
pt_gW2 = pt_model.fc2.weight.grad.cpu().numpy().T.copy()
pt_gb2 = pt_model.fc2.bias.grad.cpu().numpy().reshape(1,-1).copy()

print(f"  Loss = {loss_pt_val:.6f}")
stats("∇W1", pt_gW1)
stats("∇b1", pt_gb1)
stats("∇W2", pt_gW2)
stats("∇b2", pt_gb2)


# ═══════════════════════════════════════════════════════════════
#  7. 3-Way Gradient Comparison
# ═══════════════════════════════════════════════════════════════
section("6. 3-Way Gradient Comparison")

print(f"\n  Loss:  CPU={loss_cpu_val:.6f}  GPU={loss_gpu_val:.6f}  "
      f"PyTorch={loss_pt_val:.6f}")
print()

compare("Softmax: CPU vs GPU",      pred_cpu, pred_gpu)
compare("Softmax: CPU vs PyTorch",   pred_cpu, sm_pt)
print()

compare("∇W1: CPU vs GPU",          cpu_gW1, gpu_gW1)
compare("∇W1: CPU vs PyTorch",      cpu_gW1, pt_gW1)
compare("∇W1: GPU vs PyTorch",      gpu_gW1, pt_gW1)
print()

compare("∇b1: CPU vs GPU",          cpu_gb1, gpu_gb1)
compare("∇b1: CPU vs PyTorch",      cpu_gb1, pt_gb1)
print()

compare("∇W2: CPU vs GPU",          cpu_gW2, gpu_gW2)
compare("∇W2: CPU vs PyTorch",      cpu_gW2, pt_gW2)
compare("∇W2: GPU vs PyTorch",      gpu_gW2, pt_gW2)
print()

compare("∇b2: CPU vs GPU",          cpu_gb2, gpu_gb2)
compare("∇b2: CPU vs PyTorch",      cpu_gb2, pt_gb2)


# ═══════════════════════════════════════════════════════════════
#  8. W2 memory corruption check
# ═══════════════════════════════════════════════════════════════
section("7. GPU W2 Memory Integrity")

w2_delta = gpu_W2_after_bwd - gpu_W2_snapshot_pre_fwd
print(f"  W2 changed during fwd+bwd?  max|Δ| = {np.max(np.abs(w2_delta)):.6e}")
if np.max(np.abs(w2_delta)) > 1e-6:
    print(f"  ⚠ GPU W2 was CORRUPTED during forward/backward!")
    print(f"  ⚠ This is likely an out-of-bounds write in a CUDA kernel.")
    stats("ΔW2 (corruption)", w2_delta)
else:
    print(f"  ✓ GPU W2 memory is intact.")


print(f"\n{'═'*72}")
print("  DONE")
print(f"{'═'*72}")
