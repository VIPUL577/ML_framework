"""
Seera GPU Deep Debug Suite
==========================
Comprehensive gradient-level comparison of Seera (GPU) vs PyTorch.
Each test isolates a specific component and prints detailed diagnostics.

Run:  python test_gpu_deep_debug.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import sys, traceback, gc

# ── Seera imports ──
from Seera_init import tensor as Tensor, _is_gpu

from Seera_Engine import autograd4nn
from Seera import (
    Input, Dense, Conv2D, MaxPool2D, Flatten,
    Sequential, Loss, SGD, Adam,
)
from cuTen import cuten

# ── PyTorch ──
try:
    import torch
    import torch.nn as tnn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠ PyTorch not found — tests will run in self-check mode only")

# ── Helpers ──
total_pass = 0
total_fail = 0

def to_np(val):
    if isinstance(val, cuten):
        return val.to_host_f32()
    return np.asarray(val, dtype=np.float32)

def max_diff(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return float(np.max(np.abs(a - b)))

def check(name, cond, detail=""):
    global total_pass, total_fail
    if cond:
        total_pass += 1
        print(f"  ✅ {name}  {detail}")
    else:
        total_fail += 1
        print(f"  ❌ {name}  {detail}")


# ══════════════════════════════════════════════════════════════
#  TEST A: Isolated matmul backward (GPU vs PyTorch)
# ══════════════════════════════════════════════════════════════
def test_a_matmul_backward():
    print("\n" + "="*60)
    print("TEST A: Matmul backward — gradient comparison")
    print("="*60)
    np.random.seed(42)
    A_np = np.random.randn(16, 64).astype(np.float32) * 0.1
    B_np = np.random.randn(64, 32).astype(np.float32) * 0.1

    # Seera
    A = Tensor(A_np.copy(), is_leaf=True, device="cuda")
    B = Tensor(B_np.copy(), is_leaf=True, device="cuda")
    C = A.matmul(B)
    loss = C.sum()
    autograd4nn(loss)
    dA = to_np(A.node.cp)
    dB = to_np(B.node.cp)

    if HAS_TORCH:
        A_t = torch.tensor(A_np.copy(), requires_grad=True)
        B_t = torch.tensor(B_np.copy(), requires_grad=True)
        C_t = A_t @ B_t
        C_t.sum().backward()

        d_dA = max_diff(dA, A_t.grad.numpy())
        d_dB = max_diff(dB, B_t.grad.numpy())
        print(f"  dA max_diff={d_dA:.6f}, dB max_diff={d_dB:.6f}")
        check("matmul dA GPU==PT", d_dA < 0.05, f"diff={d_dA:.6f}")
        check("matmul dB GPU==PT", d_dB < 0.05, f"diff={d_dB:.6f}")
    else:
        check("matmul dA non-zero", np.any(np.abs(dA) > 1e-6))
        check("matmul dB non-zero", np.any(np.abs(dB) > 1e-6))


# ══════════════════════════════════════════════════════════════
#  TEST B: Isolated softmax + CCE backward
# ══════════════════════════════════════════════════════════════
def test_b_softmax_cce():
    print("\n" + "="*60)
    print("TEST B: Softmax + CCE backward — gradient comparison")
    print("="*60)
    np.random.seed(42)
    logits_np = np.random.randn(8, 5).astype(np.float32) * 0.5
    labels_np = np.zeros((8, 5), dtype=np.float32)
    for i in range(8):
        labels_np[i, i % 5] = 1.0

    # Seera
    z = Tensor(logits_np.copy(), is_leaf=True, device="cuda")
    s = z.softmax()
    loss = Loss().categorical_cross_entropy(s, Tensor(labels_np.copy(), device="cuda"))
    autograd4nn(loss)
    grad_z = to_np(z.node.cp)
    loss_val = float(to_np(loss.value).ravel()[0])
    print(f"  Seera loss = {loss_val:.6f}")

    if HAS_TORCH:
        z_t = torch.tensor(logits_np.copy(), requires_grad=True)
        s_t = torch.softmax(z_t, dim=-1)
        eps = 1e-15
        per_sample = (-torch.tensor(labels_np) * torch.log(s_t + eps)).sum(dim=-1)
        loss_t = per_sample.mean()
        loss_t.backward()
        print(f"  PyTorch loss = {loss_t.item():.6f}")

        d_loss = abs(loss_val - loss_t.item())
        d_grad = max_diff(grad_z, z_t.grad.numpy())
        print(f"  loss diff={d_loss:.6f}, grad max_diff={d_grad:.6f}")
        check("softmax+CCE loss GPU==PT", d_loss < 1e-3, f"diff={d_loss:.6f}")
        check("softmax+CCE grad GPU==PT", d_grad < 1e-2, f"diff={d_grad:.6f}")
    else:
        check("loss > 0", loss_val > 0.01, f"loss={loss_val:.6f}")
        check("grad non-zero", np.any(np.abs(grad_z) > 1e-6))


# ══════════════════════════════════════════════════════════════
#  TEST C: Isolated BCE backward
# ══════════════════════════════════════════════════════════════
def test_c_bce():
    print("\n" + "="*60)
    print("TEST C: BCE loss backward — gradient comparison")
    print("="*60)
    np.random.seed(42)
    pred_np = np.random.rand(8, 1).astype(np.float32) * 0.8 + 0.1
    target_np = np.array([[1],[0],[1],[0],[1],[0],[1],[0]], dtype=np.float32)

    # Seera
    p = Tensor(pred_np.copy(), is_leaf=True, device="cuda")
    t = Tensor(target_np.copy(), device="cuda")
    loss = Loss().binary_cross_entropy(p, t)
    autograd4nn(loss)
    loss_val = float(to_np(loss.value).ravel()[0])
    grad_p = to_np(p.node.cp)
    print(f"  Seera loss = {loss_val:.6f}")
    print(f"  Seera grad_p[:4] = {grad_p[:4].ravel()}")

    if HAS_TORCH:
        p_t = torch.tensor(pred_np.copy(), requires_grad=True)
        t_t = torch.tensor(target_np.copy())
        eps = 1e-15
        loss_t = (-t_t * torch.log(p_t + eps) - (1 - t_t) * torch.log(1 - p_t + eps)).mean()
        loss_t.backward()
        print(f"  PyTorch loss = {loss_t.item():.6f}")
        print(f"  PyTorch grad[:4] = {p_t.grad.numpy()[:4].ravel()}")

        d_loss = abs(loss_val - loss_t.item())
        d_grad = max_diff(grad_p, p_t.grad.numpy())
        print(f"  loss diff={d_loss:.6f}, grad max_diff={d_grad:.6f}")
        check("BCE loss GPU==PT", d_loss < 1e-3, f"diff={d_loss:.6f}")
        check("BCE grad GPU==PT", d_grad < 1e-2, f"diff={d_grad:.6f}")
    else:
        check("loss > 0", loss_val > 0.01, f"loss={loss_val:.6f}")
        check("grad non-zero", np.any(np.abs(grad_p) > 1e-6))


# ══════════════════════════════════════════════════════════════
#  TEST D: Multi-layer Dense forward+backward (GPU vs PyTorch)
# ══════════════════════════════════════════════════════════════
def test_d_multilayer_dense():
    print("\n" + "="*60)
    print("TEST D: Multi-layer Dense (3-layer) gradient comparison")
    print("="*60)
    np.random.seed(42)
    x_np = np.random.randn(16, 32).astype(np.float32) * 0.1
    y_np = np.zeros((16, 5), dtype=np.float32)
    for i in range(16):
        y_np[i, i % 5] = 1.0

    # Shared weights
    W1_np = np.random.randn(32, 64).astype(np.float32) * 0.05
    b1_np = np.zeros((1, 64), dtype=np.float32)
    W2_np = np.random.randn(64, 32).astype(np.float32) * 0.05
    b2_np = np.zeros((1, 32), dtype=np.float32)
    W3_np = np.random.randn(32, 5).astype(np.float32) * 0.05
    b3_np = np.zeros((1, 5), dtype=np.float32)

    # ── Seera ──
    model = Sequential([
        Input((32,)),
        Dense(32, 64, activation="relu"),
        Dense(64, 32, activation="relu"),
        Dense(32, 5, activation="softmax"),
    ], "cuda")

    model.model[1].set_weights(
        Tensor(W1_np.copy(), is_leaf=True, device="cuda"),
        Tensor(b1_np.copy(), is_leaf=True, device="cuda"),
    )
    model.model[2].set_weights(
        Tensor(W2_np.copy(), is_leaf=True, device="cuda"),
        Tensor(b2_np.copy(), is_leaf=True, device="cuda"),
    )
    model.model[3].set_weights(
        Tensor(W3_np.copy(), is_leaf=True, device="cuda"),
        Tensor(b3_np.copy(), is_leaf=True, device="cuda"),
    )

    X_batch = Tensor(x_np.copy(), is_leaf=True, device="cuda")
    y_batch = Tensor(y_np.copy(), device="cuda")
    pred = model.forward(X_batch)
    loss_fn = Loss()
    loss = loss_fn.categorical_cross_entropy(pred, y_batch)
    loss_val = float(to_np(loss.value).ravel()[0])
    print(f"  Seera loss = {loss_val:.6f}")

    model.zero_grad()
    autograd4nn(loss)

    # Extract gradients
    dW1 = to_np(model.model[1].weights.node.cp)
    db1 = to_np(model.model[1].bais.node.cp)
    dW2 = to_np(model.model[2].weights.node.cp)
    db2 = to_np(model.model[2].bais.node.cp)
    dW3 = to_np(model.model[3].weights.node.cp)
    db3 = to_np(model.model[3].bais.node.cp)

    print(f"  dW1 shape={dW1.shape} max={np.max(np.abs(dW1)):.6f}")
    print(f"  db1 shape={db1.shape} max={np.max(np.abs(db1)):.6f}")
    print(f"  dW2 shape={dW2.shape} max={np.max(np.abs(dW2)):.6f}")
    print(f"  dW3 shape={dW3.shape} max={np.max(np.abs(dW3)):.6f}")

    check("loss > 0", loss_val > 0.01, f"loss={loss_val:.6f}")
    check("dW1 non-zero", np.max(np.abs(dW1)) > 1e-8, f"max={np.max(np.abs(dW1)):.8f}")
    check("dW2 non-zero", np.max(np.abs(dW2)) > 1e-8, f"max={np.max(np.abs(dW2)):.8f}")
    check("dW3 non-zero", np.max(np.abs(dW3)) > 1e-8, f"max={np.max(np.abs(dW3)):.8f}")
    check("db1 non-zero", np.max(np.abs(db1)) > 1e-8, f"max={np.max(np.abs(db1)):.8f}")
    check("db3 non-zero", np.max(np.abs(db3)) > 1e-8, f"max={np.max(np.abs(db3)):.8f}")
    check("no NaN in dW1", not np.any(np.isnan(dW1)))
    check("no NaN in dW3", not np.any(np.isnan(dW3)))

    if HAS_TORCH:
        # ── PyTorch reference ──
        class RefModel(tnn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = tnn.Linear(32, 64, bias=True)
                self.l2 = tnn.Linear(64, 32, bias=True)
                self.l3 = tnn.Linear(32, 5, bias=True)

            def forward(self, x):
                x = torch.relu(self.l1(x))
                x = torch.relu(self.l2(x))
                x = torch.softmax(self.l3(x), dim=-1)
                return x

        ref = RefModel()
        with torch.no_grad():
            ref.l1.weight.copy_(torch.tensor(W1_np.T))
            ref.l1.bias.copy_(torch.tensor(b1_np.ravel()))
            ref.l2.weight.copy_(torch.tensor(W2_np.T))
            ref.l2.bias.copy_(torch.tensor(b2_np.ravel()))
            ref.l3.weight.copy_(torch.tensor(W3_np.T))
            ref.l3.bias.copy_(torch.tensor(b3_np.ravel()))

        x_t = torch.tensor(x_np.copy())
        y_t = torch.tensor(y_np.copy())
        pred_t = ref(x_t)
        eps = 1e-15
        per_sample = (-y_t * torch.log(pred_t + eps)).sum(dim=-1)
        loss_t = per_sample.mean()
        loss_t.backward()
        pt_loss = loss_t.item()

        # PyTorch uses (out, in) weight convention, Seera uses (in, out)
        dW1_pt = ref.l1.weight.grad.numpy().T  # (32, 64)
        db1_pt = ref.l1.bias.grad.numpy().reshape(1, -1)
        dW3_pt = ref.l3.weight.grad.numpy().T  # (32, 5)
        db3_pt = ref.l3.bias.grad.numpy().reshape(1, -1)

        d_loss = abs(loss_val - pt_loss)
        d_dW1 = max_diff(dW1, dW1_pt)
        d_db1 = max_diff(db1, db1_pt)
        d_dW3 = max_diff(dW3, dW3_pt)
        d_db3 = max_diff(db3, db3_pt)

        print(f"\n  PyTorch loss = {pt_loss:.6f}")
        print(f"  loss diff = {d_loss:.6f}")
        print(f"  dW1 diff = {d_dW1:.6f}")
        print(f"  db1 diff = {d_db1:.6f}")
        print(f"  dW3 diff = {d_dW3:.6f}")
        print(f"  db3 diff = {d_db3:.6f}")
        check("multi-layer loss GPU==PT", d_loss < 0.01, f"diff={d_loss:.6f}")
        check("multi-layer dW1 GPU==PT", d_dW1 < 0.05, f"diff={d_dW1:.6f}")
        check("multi-layer dW3 GPU==PT", d_dW3 < 0.05, f"diff={d_dW3:.6f}")
        check("multi-layer db3 GPU==PT", d_db3 < 0.05, f"diff={d_db3:.6f}")


# ══════════════════════════════════════════════════════════════
#  TEST E: Adam optimizer single-step verification
# ══════════════════════════════════════════════════════════════
def test_e_adam_single_step():
    print("\n" + "="*60)
    print("TEST E: Adam optimizer — single step weight update")
    print("="*60)
    np.random.seed(42)
    x_np = np.random.randn(8, 4).astype(np.float32) * 0.1
    y_np = np.zeros((8, 3), dtype=np.float32)
    for i in range(8):
        y_np[i, i % 3] = 1.0

    W_np = np.random.randn(4, 3).astype(np.float32) * 0.1
    b_np = np.zeros((1, 3), dtype=np.float32)

    # ── Seera ──
    model = Sequential([
        Input((4,)),
        Dense(4, 3, activation="softmax"),
    ], "cuda")
    model.model[1].set_weights(
        Tensor(W_np.copy(), is_leaf=True, device="cuda"),
        Tensor(b_np.copy(), is_leaf=True, device="cuda"),
    )

    optimizer = Adam(model, lr=0.01)
    X_batch = Tensor(x_np.copy(), is_leaf=True, device="cuda")
    y_batch = Tensor(y_np.copy(), device="cuda")
    pred = model.forward(X_batch)
    loss = Loss().categorical_cross_entropy(pred, y_batch)
    loss_val = float(to_np(loss.value).ravel()[0])

    model.zero_grad()
    autograd4nn(loss)
    optimizer.step()

    W_after = to_np(model.model[1].weights.value)
    print(f"  loss = {loss_val:.6f}")
    print(f"  W before max = {np.max(np.abs(W_np)):.6f}")
    print(f"  W after max  = {np.max(np.abs(W_after)):.6f}")
    print(f"  W change max = {np.max(np.abs(W_after - W_np)):.6f}")

    check("Adam: loss > 0", loss_val > 0.01, f"loss={loss_val:.6f}")
    check("Adam: weights changed", np.max(np.abs(W_after - W_np)) > 1e-6,
          f"max_change={np.max(np.abs(W_after - W_np)):.8f}")
    check("Adam: no NaN in weights", not np.any(np.isnan(W_after)))

    if HAS_TORCH:
        # ── PyTorch reference ──
        l = tnn.Linear(4, 3, bias=True)
        with torch.no_grad():
            l.weight.copy_(torch.tensor(W_np.T))
            l.bias.copy_(torch.tensor(b_np.ravel()))

        opt_t = torch.optim.Adam(l.parameters(), lr=0.01)
        x_t = torch.tensor(x_np.copy())
        y_t = torch.tensor(y_np.copy())
        pred_t = torch.softmax(l(x_t), dim=-1)
        eps = 1e-15
        loss_t = (-y_t * torch.log(pred_t + eps)).sum(dim=-1).mean()
        opt_t.zero_grad()
        loss_t.backward()
        opt_t.step()

        W_pt = l.weight.detach().numpy().T
        d_W = max_diff(W_after, W_pt)
        print(f"\n  PT W after max = {np.max(np.abs(W_pt)):.6f}")
        print(f"  W diff = {d_W:.6f}")
        check("Adam step GPU==PT", d_W < 0.05, f"diff={d_W:.6f}")


# ══════════════════════════════════════════════════════════════
#  TEST F: Multi-layer training — loss convergence (Adam)
# ══════════════════════════════════════════════════════════════
def test_f_adam_convergence():
    print("\n" + "="*60)
    print("TEST F: Multi-layer Adam convergence (400→256→128→3)")
    print("="*60)
    np.random.seed(42)
    x_data = np.random.randn(16, 400).astype(np.float32) * 0.1
    y_data = np.zeros((16, 3), dtype=np.float32)
    for i in range(16):
        y_data[i, i % 3] = 1.0

    model = Sequential([
        Input((400,)),
        Dense(400, 256, activation="relu"),
        Dense(256, 128, activation="relu"),
        Dense(128, 3, activation="softmax"),
    ], "cuda")

    loss_fn = Loss()
    optimizer = Adam(model, lr=0.001)  # Lower LR for stability
    losses = []

    for epoch in range(50):
        X_batch = Tensor(x_data, is_leaf=True, device="cuda")
        y_batch = Tensor(y_data, device="cuda")
        pred = model.forward(X_batch)
        loss = loss_fn.categorical_cross_entropy(pred, y_batch)
        loss_val = float(to_np(loss.value).ravel()[0])
        losses.append(loss_val)

        model.zero_grad()
        autograd4nn(loss)
        optimizer.step()

    print(f"  Epoch  1: {losses[0]:.6f}")
    print(f"  Epoch 25: {losses[24]:.6f}")
    print(f"  Epoch 50: {losses[49]:.6f}")

    check("initial loss > 0", losses[0] > 0.01, f"loss={losses[0]:.6f}")
    check("loss decreased", losses[-1] < losses[0],
          f"{losses[0]:.4f} → {losses[-1]:.4f}")
    # NOTE: With 135K+ params and only 16 samples, Adam WILL memorize → loss≈0 is valid
    check("no NaN in loss", not np.isnan(losses[-1]),
          f"final={losses[-1]:.6f}")


# ══════════════════════════════════════════════════════════════
#  TEST G: model.fit() API — end-to-end check
# ══════════════════════════════════════════════════════════════
def test_g_model_fit():
    print("\n" + "="*60)
    print("TEST G: model.fit() API — end-to-end (GPU)")
    print("="*60)
    np.random.seed(42)
    x_data = np.random.randn(32, 3).astype(np.float32)
    y_data = np.zeros((32, 2), dtype=np.float32)
    for i in range(32):
        y_data[i, i % 2] = 1.0

    model = Sequential([
        Input((3,)),
        Dense(3, 2, activation="softmax"),
    ], "cuda")

    loss_fn = Loss()
    optimizer = SGD(model, lr=0.05)

    history = model.fit(
        x_data, y_data,
        Optimizer=optimizer,
        Loss=loss_fn.categorical_cross_entropy,
        Epochs=20,
        batch_size=8,
        Loss_interval=10,
    )

    print(f"  history[0]  = {history[0]:.6f}")
    print(f"  history[-1] = {history[-1]:.6f}")
    check("history length == 20", len(history) == 20)
    check("loss decreased via fit()", history[-1] < history[0],
          f"{history[0]:.4f} → {history[-1]:.4f}")
    check("no NaN in history", not np.any(np.isnan(history)))


# ══════════════════════════════════════════════════════════════
#  TEST H: Large encoder-decoder (UNet-style) gradient check
# ══════════════════════════════════════════════════════════════
def test_h_large_unet():
    print("\n" + "="*60)
    print("TEST H: Large encoder-decoder gradient comparison")
    print("="*60)
    np.random.seed(42)

    # Encoder: 256→128→64 with relu, then softmax→5 classes
    x_np = np.random.randn(8, 256).astype(np.float32) * 0.1
    y_np = np.zeros((8, 5), dtype=np.float32)
    for i in range(8):
        y_np[i, i % 5] = 1.0

    W1 = np.random.randn(256, 128).astype(np.float32) * 0.02
    b1 = np.zeros((1, 128), dtype=np.float32)
    W2 = np.random.randn(128, 64).astype(np.float32) * 0.02
    b2 = np.zeros((1, 64), dtype=np.float32)
    W3 = np.random.randn(64, 128).astype(np.float32) * 0.02
    b3 = np.zeros((1, 128), dtype=np.float32)
    W4 = np.random.randn(128, 5).astype(np.float32) * 0.02
    b4 = np.zeros((1, 5), dtype=np.float32)

    # ── Seera ──
    model = Sequential([
        Input((256,)),
        Dense(256, 128, activation="relu"),
        Dense(128, 64, activation="relu"),
        Dense(64, 128, activation="relu"),
        Dense(128, 5, activation="softmax"),
    ], "cuda")

    model.model[1].set_weights(Tensor(W1.copy(), is_leaf=True, device="cuda"),
                                Tensor(b1.copy(), is_leaf=True, device="cuda"))
    model.model[2].set_weights(Tensor(W2.copy(), is_leaf=True, device="cuda"),
                                Tensor(b2.copy(), is_leaf=True, device="cuda"))
    model.model[3].set_weights(Tensor(W3.copy(), is_leaf=True, device="cuda"),
                                Tensor(b3.copy(), is_leaf=True, device="cuda"))
    model.model[4].set_weights(Tensor(W4.copy(), is_leaf=True, device="cuda"),
                                Tensor(b4.copy(), is_leaf=True, device="cuda"))

    X_batch = Tensor(x_np.copy(), is_leaf=True, device="cuda")
    y_batch = Tensor(y_np.copy(), device="cuda")
    pred = model.forward(X_batch)
    loss = Loss().categorical_cross_entropy(pred, y_batch)
    loss_val = float(to_np(loss.value).ravel()[0])

    model.zero_grad()
    autograd4nn(loss)

    dW1_s = to_np(model.model[1].weights.node.cp)
    dW2_s = to_np(model.model[2].weights.node.cp)
    dW3_s = to_np(model.model[3].weights.node.cp)
    dW4_s = to_np(model.model[4].weights.node.cp)

    print(f"  Seera loss = {loss_val:.6f}")
    for i, (name, dw) in enumerate(zip(
        ["dW1(256→128)", "dW2(128→64)", "dW3(64→128)", "dW4(128→5)"],
        [dW1_s, dW2_s, dW3_s, dW4_s]
    )):
        print(f"  {name}: shape={dw.shape} max={np.max(np.abs(dw)):.6f} "
              f"mean={np.mean(np.abs(dw)):.6f} nan={np.any(np.isnan(dw))}")

    check("encoder-decoder loss > 0", loss_val > 0.01, f"loss={loss_val:.6f}")
    check("dW1 non-zero", np.max(np.abs(dW1_s)) > 1e-8)
    check("dW2 non-zero", np.max(np.abs(dW2_s)) > 1e-8)
    check("dW3 non-zero", np.max(np.abs(dW3_s)) > 1e-8)
    check("dW4 non-zero", np.max(np.abs(dW4_s)) > 1e-8)
    check("no NaN in any gradient",
          not any(np.any(np.isnan(d)) for d in [dW1_s, dW2_s, dW3_s, dW4_s]))

    if HAS_TORCH:
        class UNetRef(tnn.Module):
            def __init__(self):
                super().__init__()
                self.enc1 = tnn.Linear(256, 128)
                self.enc2 = tnn.Linear(128, 64)
                self.dec1 = tnn.Linear(64, 128)
                self.out  = tnn.Linear(128, 5)
            def forward(self, x):
                x = torch.relu(self.enc1(x))
                x = torch.relu(self.enc2(x))
                x = torch.relu(self.dec1(x))
                x = torch.softmax(self.out(x), dim=-1)
                return x

        ref = UNetRef()
        with torch.no_grad():
            ref.enc1.weight.copy_(torch.tensor(W1.T))
            ref.enc1.bias.copy_(torch.tensor(b1.ravel()))
            ref.enc2.weight.copy_(torch.tensor(W2.T))
            ref.enc2.bias.copy_(torch.tensor(b2.ravel()))
            ref.dec1.weight.copy_(torch.tensor(W3.T))
            ref.dec1.bias.copy_(torch.tensor(b3.ravel()))
            ref.out.weight.copy_(torch.tensor(W4.T))
            ref.out.bias.copy_(torch.tensor(b4.ravel()))

        x_t = torch.tensor(x_np.copy())
        y_t = torch.tensor(y_np.copy())
        pred_t = ref(x_t)
        eps = 1e-15
        loss_t = (-y_t * torch.log(pred_t + eps)).sum(dim=-1).mean()
        loss_t.backward()

        dW1_pt = ref.enc1.weight.grad.numpy().T
        dW2_pt = ref.enc2.weight.grad.numpy().T
        dW3_pt = ref.dec1.weight.grad.numpy().T
        dW4_pt = ref.out.weight.grad.numpy().T

        d1 = max_diff(dW1_s, dW1_pt)
        d2 = max_diff(dW2_s, dW2_pt)
        d3 = max_diff(dW3_s, dW3_pt)
        d4 = max_diff(dW4_s, dW4_pt)
        d_loss = abs(loss_val - loss_t.item())

        print(f"\n  PyTorch loss = {loss_t.item():.6f}, diff = {d_loss:.6f}")
        print(f"  dW1 diff={d1:.6f}  dW2 diff={d2:.6f}")
        print(f"  dW3 diff={d3:.6f}  dW4 diff={d4:.6f}")

        check("UNet loss GPU==PT", d_loss < 0.01, f"diff={d_loss:.6f}")
        check("UNet dW1 GPU==PT", d1 < 0.05, f"diff={d1:.6f}")
        check("UNet dW4 GPU==PT", d4 < 0.05, f"diff={d4:.6f}")


# ══════════════════════════════════════════════════════════════
#  TEST I: Broadcast gradient reduction shape check
# ══════════════════════════════════════════════════════════════
def test_i_reduce_grad_shapes():
    print("\n" + "="*60)
    print("TEST I: _reduce_grad_gpu shape correctness")
    print("="*60)

    from Seera_Engine import autograd4nn as AG

    # (8, 16) → (1, 16) should keep shape (1, 16)
    g1 = cuten(np.random.randn(8, 16).astype(np.float32))
    r1 = AG._reduce_grad_gpu(g1, (1, 16))
    check("(8,16)→(1,16) shape", r1.shape == (1, 16), f"got {r1.shape}")

    # (4, 8, 16) → (8, 16) should keep shape (8, 16)
    g2 = cuten(np.random.randn(4, 8, 16).astype(np.float32))
    r2 = AG._reduce_grad_gpu(g2, (8, 16))
    check("(4,8,16)→(8,16) shape", r2.shape == (8, 16), f"got {r2.shape}")

    # (4, 1, 16) → (1, 1, 16) should give (1, 1, 16)
    g3 = cuten(np.random.randn(4, 1, 16).astype(np.float32))
    r3 = AG._reduce_grad_gpu(g3, (1, 1, 16))
    check("(4,1,16)→(1,1,16) shape", r3.shape == (1, 1, 16), f"got {r3.shape}")

    # (8,) → (1,) should give (1,)
    g4 = cuten(np.random.randn(8).astype(np.float32))
    r4 = AG._reduce_grad_gpu(g4, (1,))
    check("(8,)→(1,) shape", r4.shape == (1,), f"got {r4.shape}")

    # Verify values: sum of (8, 16) along axis 0 should match numpy
    test_data = np.random.randn(8, 16).astype(np.float32)
    g5 = cuten(test_data)
    r5 = AG._reduce_grad_gpu(g5, (1, 16))
    r5_np = r5.to_host_f32()
    expected = test_data.sum(axis=0, keepdims=True)
    d = max_diff(r5_np, expected)
    check("(8,16)→(1,16) values correct", d < 1e-3, f"diff={d:.6f}")


# ══════════════════════════════════════════════════════════════
#  TEST J: Training with SGD — 3 epochs loss progression
# ══════════════════════════════════════════════════════════════
def test_j_sgd_convergence():
    print("\n" + "="*60)
    print("TEST J: SGD convergence — simple network")
    print("="*60)
    np.random.seed(42)
    x_data = np.random.randn(32, 8).astype(np.float32)
    y_data = np.zeros((32, 4), dtype=np.float32)
    for i in range(32):
        y_data[i, i % 4] = 1.0

    model = Sequential([
        Input((8,)),
        Dense(8, 16, activation="relu"),
        Dense(16, 4, activation="softmax"),
    ], "cuda")

    loss_fn = Loss()
    optimizer = SGD(model, lr=0.1)
    losses = []

    for epoch in range(30):
        X_batch = Tensor(x_data, is_leaf=True, device="cuda")
        y_batch = Tensor(y_data, device="cuda")
        pred = model.forward(X_batch)
        loss = loss_fn.categorical_cross_entropy(pred, y_batch)
        loss_val = float(to_np(loss.value).ravel()[0])
        losses.append(loss_val)
        model.zero_grad()
        autograd4nn(loss)
        optimizer.step()

    print(f"  Epoch  1: {losses[0]:.6f}")
    print(f"  Epoch 15: {losses[14]:.6f}")
    print(f"  Epoch 30: {losses[-1]:.6f}")

    check("initial loss > 0", losses[0] > 0.01, f"loss={losses[0]:.6f}")
    check("loss decreased (SGD 2-layer)", losses[-1] < losses[0],
          f"{losses[0]:.4f} → {losses[-1]:.4f}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Seera GPU Deep Debug Suite                             ║")
    print("╚══════════════════════════════════════════════════════════╝")

    tests = [
        ("A", test_a_matmul_backward),
        ("B", test_b_softmax_cce),
        ("C", test_c_bce),
        ("D", test_d_multilayer_dense),
        ("E", test_e_adam_single_step),
        ("F", test_f_adam_convergence),
        ("G", test_g_model_fit),
        ("H", test_h_large_unet),
        ("I", test_i_reduce_grad_shapes),
        ("J", test_j_sgd_convergence),
    ]

    for tid, fn in tests:
        try:
            fn()
        except Exception:
            print(f"\n  💥 TEST {tid} CRASHED:")
            traceback.print_exc()
            total_fail += 1
        # Free GPU memory between tests to prevent exhaustion
        gc.collect()

    print("\n" + "═"*60)
    print(f"  RESULTS:  {total_pass} passed,  {total_fail} failed")
    print("═"*60)
    if total_fail == 0:
        print("  \033[92m🎉 ALL TESTS PASSED!\033[0m")
    else:
        print(f"  \033[91m⚠  {total_fail} test(s) failed\033[0m")
