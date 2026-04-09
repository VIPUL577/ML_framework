"""
═══════════════════════════════════════════════════════════════
  Seera ML Framework — GPU Test Suite (with PyTorch comparison)
  All tests run on CUDA. Gradients compared against PyTorch.
═══════════════════════════════════════════════════════════════
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from Seera_init import tensor as Tensor, _is_gpu
from Seera_Engine import autograd4nn
from Seera import Input, Dense, Sequential, Loss, SGD, Adam
from cuTen import cuten

# ── Styling ──
PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
total_pass = 0
total_fail = 0

def check(name, condition, detail=""):
    global total_pass, total_fail
    status = PASS if condition else FAIL
    if condition:
        total_pass += 1
    else:
        total_fail += 1
    print(f"  {status} {name}" + (f"  ({detail})" if detail else ""))
    return condition

def to_np(val):
    """Bring any value (cuten, ndarray, scalar) to host numpy."""
    if isinstance(val, cuten):
        return val.to_host_f32()
    return np.asarray(val, dtype=np.float32)

def max_diff(a, b):
    return float(np.max(np.abs(to_np(a).ravel() - to_np(b).ravel())))


# ══════════════════════════════════════════════════════════════
#  TEST 0: cuten scalar ops must NOT mutate in-place
# ══════════════════════════════════════════════════════════════
def test_0():
    print("\n" + "="*60)
    print("TEST 0: cuten scalar ops — no in-place mutation")
    print("="*60)

    data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    for op_name, op, expected in [
        ("mul *5",   lambda g: g * 5.0,   data * 5.0),
        ("add +10",  lambda g: g + 10.0,  data + 10.0),
        ("pow **2",  lambda g: g ** 2.0,  data ** 2.0),
        ("neg -",    lambda g: -g,        -data),
    ]:
        g = cuten(data.copy())
        orig = g.to_host_f32().copy()
        result = op(g)
        after = g.to_host_f32()
        check(f"{op_name}: original unchanged", np.allclose(orig, after))
        check(f"{op_name}: result correct", np.allclose(result.to_host_f32(), expected, atol=1e-5))


# ══════════════════════════════════════════════════════════════
#  TEST 1: Element-wise ops forward + backward (GPU vs PyTorch)
# ══════════════════════════════════════════════════════════════
def test_1():
    print("\n" + "="*60)
    print("TEST 1: Element-wise a*b+a — GPU backward vs PyTorch")
    print("="*60)

    np.random.seed(42)
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(2, 3).astype(np.float32)

    # GPU
    a = Tensor(a_np.copy(), is_leaf=True, device="cuda")
    b = Tensor(b_np.copy(), is_leaf=True, device="cuda")
    c = a * b + a
    loss = c.sum()
    autograd4nn(loss)

    try:
        import torch
        a_t = torch.tensor(a_np.copy(), requires_grad=True)
        b_t = torch.tensor(b_np.copy(), requires_grad=True)
        c_t = a_t * b_t + a_t
        loss_t = c_t.sum()
        loss_t.backward()

        d_fwd = max_diff(loss.value, loss_t.item())
        d_ga = max_diff(a.node.cp, a_t.grad.numpy())
        d_gb = max_diff(b.node.cp, b_t.grad.numpy())
        check("forward GPU==PT", d_fwd < 1e-3, f"diff={d_fwd:.6f}")
        check("grad_a GPU==PT", d_ga < 1e-3, f"diff={d_ga:.6f}")
        check("grad_b GPU==PT", d_gb < 1e-3, f"diff={d_gb:.6f}")
    except ImportError:
        check("grad_a non-zero", np.any(np.abs(to_np(a.node.cp)) > 1e-6))
        check("grad_b non-zero", np.any(np.abs(to_np(b.node.cp)) > 1e-6))


# ══════════════════════════════════════════════════════════════
#  TEST 2: Matmul + bias backward (GPU vs PyTorch)
# ══════════════════════════════════════════════════════════════
def test_2():
    print("\n" + "="*60)
    print("TEST 2: Matmul + bias backward (GPU vs PyTorch)")
    print("="*60)

    np.random.seed(42)
    x_np = np.random.randn(4, 3).astype(np.float32)
    W_np = np.random.randn(3, 2).astype(np.float32)
    b_np = np.zeros((1, 2), dtype=np.float32)

    x = Tensor(x_np.copy(), is_leaf=True, device="cuda")
    W = Tensor(W_np.copy(), is_leaf=True, device="cuda")
    b = Tensor(b_np.copy(), is_leaf=True, device="cuda")
    z = x.matmul(W) + b
    loss = z.sum()
    autograd4nn(loss)

    try:
        import torch
        x_t = torch.tensor(x_np.copy(), requires_grad=True)
        W_t = torch.tensor(W_np.copy(), requires_grad=True)
        b_t = torch.tensor(b_np.copy(), requires_grad=True)
        z_t = x_t @ W_t + b_t
        loss_t = z_t.sum()
        loss_t.backward()

        d_gW = max_diff(W.node.cp, W_t.grad.numpy())
        d_gb = max_diff(b.node.cp, b_t.grad.numpy())
        d_gx = max_diff(x.node.cp, x_t.grad.numpy())
        check("grad_W GPU==PT", d_gW < 1e-2, f"diff={d_gW:.6f}")
        check("grad_b GPU==PT", d_gb < 1e-2, f"diff={d_gb:.6f}")
        check("grad_x GPU==PT", d_gx < 1e-2, f"diff={d_gx:.6f}")
    except ImportError:
        check("grad_W non-zero", np.any(np.abs(to_np(W.node.cp)) > 1e-6))


# ══════════════════════════════════════════════════════════════
#  TEST 3: Softmax + CCE loss (GPU vs PyTorch)
# ══════════════════════════════════════════════════════════════
def test_3():
    print("\n" + "="*60)
    print("TEST 3: Softmax + CCE loss (GPU vs PyTorch)")
    print("="*60)

    np.random.seed(42)
    logits_np = np.random.randn(4, 3).astype(np.float32)
    y_np = np.zeros((4, 3), dtype=np.float32)
    y_np[0, 0] = y_np[1, 1] = y_np[2, 2] = y_np[3, 0] = 1.0

    logits = Tensor(logits_np.copy(), is_leaf=True, device="cuda")
    s = logits.softmax()
    y = Tensor(y_np.copy(), device="cuda")
    loss_fn = Loss()
    loss = loss_fn.categorical_cross_entropy(s, y)
    autograd4nn(loss)

    loss_val = float(to_np(loss.value).ravel()[0])
    grad = to_np(logits.node.cp)

    print(f"  GPU loss = {loss_val:.6f}")
    check("loss > 0", loss_val > 0.01, f"loss={loss_val:.6f}")
    check("grads non-zero", np.any(np.abs(grad) > 1e-6))

    try:
        import torch
        import torch.nn.functional as F

        logits_t = torch.tensor(logits_np.copy(), requires_grad=True)
        s_t = F.softmax(logits_t, dim=-1)
        y_t = torch.tensor(y_np.copy())
        loss_t = (-y_t * torch.log(s_t + 1e-15)).sum(dim=-1).mean()
        loss_t.backward()

        d_loss = abs(loss_val - loss_t.item())
        d_grad = max_diff(grad, logits_t.grad.numpy())
        check("loss GPU==PT", d_loss < 1e-3, f"diff={d_loss:.6f}")
        check("grad GPU==PT", d_grad < 1e-3, f"diff={d_grad:.6f}")
    except ImportError:
        pass


# ══════════════════════════════════════════════════════════════
#  TEST 4: Dense layer — identical weights, GPU vs PyTorch
# ══════════════════════════════════════════════════════════════
def test_4():
    print("\n" + "="*60)
    print("TEST 4: Dense layer (same weights, GPU vs PyTorch)")
    print("="*60)

    np.random.seed(42)
    W_np = np.random.randn(3, 2).astype(np.float32) * 0.1
    b_np = np.zeros((1, 2), dtype=np.float32)
    x_np = np.random.randn(4, 3).astype(np.float32)
    y_np = np.zeros((4, 2), dtype=np.float32)
    y_np[0, 0] = y_np[1, 1] = y_np[2, 0] = y_np[3, 1] = 1.0

    model = Sequential([
        Input((3,)),
        Dense(3, 2, activation="softmax"),
    ], "cuda")
    model.model[1].set_weights(
        Tensor(W_np.copy(), is_leaf=True, device="cuda"),
        Tensor(b_np.copy(), is_leaf=True, device="cuda"),
    )

    loss_fn = Loss()
    x = Tensor(x_np.copy(), is_leaf=True, device="cuda")
    y = Tensor(y_np.copy(), device="cuda")
    pred = model.forward(x)
    loss = loss_fn.categorical_cross_entropy(pred, y)
    model.zero_grad()
    autograd4nn(loss)

    W_layer, b_layer = model.model[1].get_weights()
    loss_val = float(to_np(loss.value).ravel()[0])
    grad_W = to_np(W_layer.node.cp)
    grad_b = to_np(b_layer.node.cp)

    print(f"  GPU loss = {loss_val:.6f}")
    print(f"  GPU grad_W:\n{grad_W}")
    check("loss > 0", loss_val > 0.01, f"loss={loss_val:.6f}")
    check("grad_W non-zero", np.any(np.abs(grad_W) > 1e-6))

    try:
        import torch
        import torch.nn.functional as F

        x_t = torch.tensor(x_np.copy())
        W_t = torch.tensor(W_np.copy(), requires_grad=True)
        b_t = torch.tensor(b_np.copy(), requires_grad=True)
        z_t = x_t @ W_t + b_t
        s_t = F.softmax(z_t, dim=-1)
        y_t = torch.tensor(y_np.copy())
        loss_t = (-y_t * torch.log(s_t + 1e-15)).sum(dim=-1).mean()
        loss_t.backward()

        d_loss = abs(loss_val - loss_t.item())
        d_gW = max_diff(grad_W, W_t.grad.numpy())
        d_gb = max_diff(grad_b, b_t.grad.numpy())
        print(f"  PyTorch loss = {loss_t.item():.6f}, grad_W:\n{W_t.grad.numpy()}")
        check("loss GPU==PT", d_loss < 1e-3, f"diff={d_loss:.6f}")
        check("grad_W GPU==PT", d_gW < 1e-2, f"diff={d_gW:.6f}")
        check("grad_b GPU==PT", d_gb < 1e-2, f"diff={d_gb:.6f}")
    except ImportError:
        pass


# ══════════════════════════════════════════════════════════════
#  TEST 5: MSE loss (GPU vs PyTorch)
# ══════════════════════════════════════════════════════════════
def test_5():
    print("\n" + "="*60)
    print("TEST 5: MSE loss backward (GPU vs PyTorch)")
    print("="*60)

    np.random.seed(42)
    pred_np = np.random.randn(4, 2).astype(np.float32)
    target_np = np.random.randn(4, 2).astype(np.float32)

    p = Tensor(pred_np.copy(), is_leaf=True, device="cuda")
    t = Tensor(target_np.copy(), device="cuda")
    loss_fn = Loss()
    loss = loss_fn.mse(p, t)
    autograd4nn(loss)
    loss_val = float(to_np(loss.value).ravel()[0])
    grad = to_np(p.node.cp)

    check("loss > 0", loss_val > 0.01, f"loss={loss_val:.6f}")

    try:
        import torch
        p_t = torch.tensor(pred_np.copy(), requires_grad=True)
        t_t = torch.tensor(target_np.copy())
        loss_t = ((p_t - t_t) ** 2).mean()
        loss_t.backward()

        d_loss = abs(loss_val - loss_t.item())
        d_grad = max_diff(grad, p_t.grad.numpy())
        check("loss GPU==PT", d_loss < 1e-2, f"diff={d_loss:.6f}")
        check("grad GPU==PT", d_grad < 1e-2, f"diff={d_grad:.6f}")
    except ImportError:
        check("grad non-zero", np.any(np.abs(grad) > 1e-6))


# ══════════════════════════════════════════════════════════════
#  TEST 6: Sum / Mean reduction backward on GPU
# ══════════════════════════════════════════════════════════════
def test_6():
    print("\n" + "="*60)
    print("TEST 6: Sum / Mean reduction backward (GPU vs expected)")
    print("="*60)

    np.random.seed(42)
    data_np = np.random.randn(3, 4).astype(np.float32)

    # Sum backward: gradient should be all 1s
    x = Tensor(data_np.copy(), is_leaf=True, device="cuda")
    s = x.sum()
    autograd4nn(s)
    grad_sum = to_np(x.node.cp)
    check("sum grad == 1.0", np.allclose(grad_sum, 1.0, atol=1e-5),
          f"grad[0,0]={grad_sum[0,0]:.6f}")

    # Mean backward: gradient should be 1/N
    x2 = Tensor(data_np.copy(), is_leaf=True, device="cuda")
    m = x2.mean()
    autograd4nn(m)
    grad_mean = to_np(x2.node.cp)
    expected = 1.0 / data_np.size
    check(f"mean grad ≈ 1/{data_np.size} = {expected:.6f}",
          np.allclose(grad_mean, expected, atol=1e-4),
          f"actual={grad_mean[0,0]:.6f}")

    # Sum(axis=-1) backward
    x3 = Tensor(data_np.copy(), is_leaf=True, device="cuda")
    sa = x3.sum(axis=-1)
    loss3 = sa.sum()
    autograd4nn(loss3)
    grad_axis = to_np(x3.node.cp)
    check("sum(axis=-1) grad == 1.0", np.allclose(grad_axis, 1.0, atol=1e-5))

    # Mean(axis=-1) backward
    x4 = Tensor(data_np.copy(), is_leaf=True, device="cuda")
    ma = x4.mean(axis=-1)
    loss4 = ma.sum()
    autograd4nn(loss4)
    grad_axis_mean = to_np(x4.node.cp)
    expected_axis = 1.0 / data_np.shape[-1]
    check(f"mean(axis=-1) grad ≈ 1/{data_np.shape[-1]} = {expected_axis:.4f}",
          np.allclose(grad_axis_mean, expected_axis, atol=1e-4),
          f"actual={grad_axis_mean[0,0]:.6f}")


# ══════════════════════════════════════════════════════════════
#  TEST 7: Sigmoid + MSE (GPU vs PyTorch)
# ══════════════════════════════════════════════════════════════
def test_7():
    print("\n" + "="*60)
    print("TEST 7: Sigmoid + MSE (GPU vs PyTorch)")
    print("="*60)

    np.random.seed(42)
    W_np = np.random.randn(3, 2).astype(np.float32) * 0.1
    b_np = np.zeros((1, 2), dtype=np.float32)
    x_np = np.array([[0.5, 1.0, -0.5]], dtype=np.float32)
    y_np = np.array([[1.0, 0.0]], dtype=np.float32)

    x = Tensor(x_np.copy(), is_leaf=True, device="cuda")
    W = Tensor(W_np.copy(), is_leaf=True, device="cuda")
    b = Tensor(b_np.copy(), is_leaf=True, device="cuda")
    z = x.matmul(W) + b
    out = z.sigmoid()
    loss = Loss().mse(out, Tensor(y_np.copy(), device="cuda"))
    autograd4nn(loss)

    loss_val = float(to_np(loss.value).ravel()[0])
    grad_W = to_np(W.node.cp)

    check("loss > 0", loss_val > 0.001, f"loss={loss_val:.6f}")

    try:
        import torch
        x_t = torch.tensor(x_np.copy(), requires_grad=True)
        W_t = torch.tensor(W_np.copy(), requires_grad=True)
        b_t = torch.tensor(b_np.copy(), requires_grad=True)
        z_t = x_t @ W_t + b_t
        out_t = torch.sigmoid(z_t)
        loss_t = ((out_t - torch.tensor(y_np)) ** 2).mean()
        loss_t.backward()

        d_gW = max_diff(grad_W, W_t.grad.numpy())
        d_gb = max_diff(to_np(b.node.cp), b_t.grad.numpy())
        check("grad_W GPU==PT", d_gW < 1e-3, f"diff={d_gW:.6f}")
        check("grad_b GPU==PT", d_gb < 1e-3, f"diff={d_gb:.6f}")
    except ImportError:
        check("grad_W non-zero", np.any(np.abs(grad_W) > 1e-6))


# ══════════════════════════════════════════════════════════════
#  TEST 8: Activations — log, sigmoid, relu, tanh (GPU vs PT)
# ══════════════════════════════════════════════════════════════
def test_8():
    print("\n" + "="*60)
    print("TEST 8: Activations forward+backward (GPU vs PyTorch)")
    print("="*60)

    np.random.seed(42)
    data_np = np.abs(np.random.randn(2, 3).astype(np.float32)) + 0.1

    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    for act_name, seera_fn, torch_fn_name in [
        ("log",     Tensor.log,     "log"),
        ("sigmoid", Tensor.sigmoid, "sigmoid"),
        ("relu",    Tensor.relu,    "relu"),
        ("tanh",    Tensor.tanh,    "tanh"),
    ]:
        x = Tensor(data_np.copy(), is_leaf=True, device="cuda")
        y = seera_fn(x)
        l = y.sum()
        autograd4nn(l)
        grad = to_np(x.node.cp)

        if has_torch:
            x_t = torch.tensor(data_np.copy(), requires_grad=True)
            y_t = getattr(torch, torch_fn_name)(x_t)
            l_t = y_t.sum()
            l_t.backward()

            d_fwd = max_diff(y.value, y_t.detach().numpy())
            d_grad = max_diff(grad, x_t.grad.numpy())
            check(f"{act_name} fwd GPU==PT", d_fwd < 1e-3, f"diff={d_fwd:.6f}")
            check(f"{act_name} grad GPU==PT", d_grad < 1e-3, f"diff={d_grad:.6f}")
        else:
            check(f"{act_name} grad non-zero", np.any(np.abs(grad) > 1e-6))


# ══════════════════════════════════════════════════════════════
#  TEST 9: Training loop — loss must decrease (GPU)
# ══════════════════════════════════════════════════════════════
def test_9():
    print("\n" + "="*60)
    print("TEST 9: Training loop — loss must decrease (GPU, SGD)")
    print("="*60)

    np.random.seed(42)
    x_data = np.random.randn(8, 3).astype(np.float32)
    y_data = np.zeros((8, 2), dtype=np.float32)
    for i in range(8):
        y_data[i, i % 2] = 1.0

    W_np = np.random.randn(3, 2).astype(np.float32) * 0.1
    b_np = np.zeros((1, 2), dtype=np.float32)

    model = Sequential([
        Input((3,)),
        Dense(3, 2, activation="softmax"),
    ], "cuda")
    model.model[1].set_weights(
        Tensor(W_np.copy(), is_leaf=True, device="cuda"),
        Tensor(b_np.copy(), is_leaf=True, device="cuda"),
    )

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
    print(f"  Epoch 30: {losses[29]:.6f}")

    check("initial loss > 0", losses[0] > 0.01, f"loss={losses[0]:.6f}")
    check("loss decreased", losses[-1] < losses[0],
          f"{losses[0]:.4f} → {losses[-1]:.4f}")
    check("loss not zero/nan", losses[-1] > 1e-10 and not np.isnan(losses[-1]))


# ══════════════════════════════════════════════════════════════
#  TEST 10: Training loop — Adam optimizer (GPU)
# ══════════════════════════════════════════════════════════════
def test_10():
    print("\n" + "="*60)
    print("TEST 10: Training loop — loss must decrease (GPU, Adam)")
    print("="*60)

    np.random.seed(42)
    x_data = np.random.randn(16, 400).astype(np.float32)
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
    optimizer = Adam(model, lr=0.01)
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
    print(f"  Epoch 30: {losses[29]:.6f}")

    check("initial loss > 0", losses[0] > 0.01, f"loss={losses[0]:.6f}")
    check("loss decreased", losses[-1] < losses[0],
          f"{losses[0]:.4f} → {losses[-1]:.4f}")
    check("loss not zero/nan", losses[-1] > 1e-10 and not np.isnan(losses[-1]))


# ══════════════════════════════════════════════════════════════
#  TEST 11: model.fit() API works end-to-end on GPU
# ══════════════════════════════════════════════════════════════
def test_11():
    print("\n" + "="*60)
    print("TEST 11: model.fit() API — end-to-end (GPU)")
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
        Epochs=10,
        batch_size=8,
        Loss_interval=5,
    )

    check("history length == 10", len(history) == 10)
    check("loss decreased via fit()", history[-1] < history[0],
          f"{history[0]:.4f} → {history[-1]:.4f}")
    check("no NaN in history", not np.any(np.isnan(history)))


# ══════════════════════════════════════════════════════════════
#  TEST 12: BCE loss (GPU vs PyTorch)
# ══════════════════════════════════════════════════════════════
def test_12():
    print("\n" + "="*60)
    print("TEST 12: Binary Cross Entropy (GPU vs PyTorch)")
    print("="*60)

    np.random.seed(42)
    pred_np = np.random.rand(4, 1).astype(np.float32) * 0.8 + 0.1  # range [0.1, 0.9]
    target_np = np.array([[1], [0], [1], [0]], dtype=np.float32)

    p = Tensor(pred_np.copy(), is_leaf=True, device="cuda")
    t = Tensor(target_np.copy(), device="cuda")
    loss = Loss().binary_cross_entropy(p, t)
    autograd4nn(loss)
    loss_val = float(to_np(loss.value).ravel()[0])
    grad = to_np(p.node.cp)

    check("loss > 0", loss_val > 0.01, f"loss={loss_val:.6f}")

    try:
        import torch
        p_t = torch.tensor(pred_np.copy(), requires_grad=True)
        t_t = torch.tensor(target_np.copy())
        eps = 1e-15
        loss_t = (-t_t * torch.log(p_t + eps) - (1 - t_t) * torch.log(1 - p_t + eps)).mean()
        loss_t.backward()

        d_loss = abs(loss_val - loss_t.item())
        d_grad = max_diff(grad, p_t.grad.numpy())
        check("loss GPU==PT", d_loss < 1e-3, f"diff={d_loss:.6f}")
        check("grad GPU==PT", d_grad < 1e-2, f"diff={d_grad:.6f}")
    except ImportError:
        check("grad non-zero", np.any(np.abs(grad) > 1e-6))


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Seera ML Framework — GPU Test Suite (CUDA + PyTorch)  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    test_0()
    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    test_6()
    test_7()
    test_8()
    test_9()
    test_10()
    test_11()
    test_12()

    print("\n" + "═"*60)
    print(f"  RESULTS:  {total_pass} passed,  {total_fail} failed")
    print("═"*60)
    if total_fail == 0:
        print("  \033[92m🎉 ALL TESTS PASSED!\033[0m")
    else:
        print(f"  \033[91m⚠  {total_fail} test(s) failed\033[0m")
