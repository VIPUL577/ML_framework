"""
Comprehensive GPU vs CPU vs PyTorch gradient comparison test suite.
Tests each component individually and then end-to-end training,
using IDENTICAL initial weights across all three backends.
"""
import sys, os
# _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, _PROJECT_ROOT)
# os.chdir(_PROJECT_ROOT)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

# ── Import Seera framework ──
from Seera_init import tensor as Tensor, _is_gpu
from Seera_Engine import autograd4nn
from Seera import Input, Dense, Sequential, Loss, SGD
from cuTen import cuten

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {name}" + (f"  ({detail})" if detail else ""))
    return condition

def max_diff(a, b):
    """Max absolute difference of two arrays (bring to host if needed)."""
    if isinstance(a, cuten):
        a = a.to_host_f32()
    if isinstance(b, cuten):
        b = b.to_host_f32()
    return float(np.max(np.abs(np.array(a).ravel() - np.array(b).ravel())))

# ══════════════════════════════════════════════════════════════
#  TEST 0: cuten scalar ops must NOT mutate in-place
# ══════════════════════════════════════════════════════════════
def test_0_cuten_no_inplace_mutation():
    print("\n" + "="*60)
    print("TEST 0: cuten scalar ops must NOT mutate in-place")
    print("="*60)

    data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    g = cuten(data.copy())
    original = g.to_host_f32().copy()

    # Test multiply
    result = g * 5.0
    after = g.to_host_f32()
    check("mul: original unchanged", np.allclose(original, after),
          f"orig={original.ravel()}, after={after.ravel()}")
    check("mul: result correct", np.allclose(result.to_host_f32(), data * 5.0))

    # Test add
    g2 = cuten(data.copy())
    original2 = g2.to_host_f32().copy()
    result2 = g2 + 10.0
    after2 = g2.to_host_f32()
    check("add: original unchanged", np.allclose(original2, after2))
    check("add: result correct", np.allclose(result2.to_host_f32(), data + 10.0))

    # Test pow
    g3 = cuten(data.copy())
    original3 = g3.to_host_f32().copy()
    result3 = g3 ** 2.0
    after3 = g3.to_host_f32()
    check("pow: original unchanged", np.allclose(original3, after3))
    check("pow: result correct", np.allclose(result3.to_host_f32(), data ** 2.0))

    # Test neg (uses * -1)
    g4 = cuten(data.copy())
    original4 = g4.to_host_f32().copy()
    result4 = -g4
    after4 = g4.to_host_f32()
    check("neg: original unchanged", np.allclose(original4, after4))
    check("neg: result correct", np.allclose(result4.to_host_f32(), -data))


# ══════════════════════════════════════════════════════════════
#  TEST 1: Element-wise ops forward + backward (CPU vs GPU)
# ══════════════════════════════════════════════════════════════
def test_1_elementwise_ops():
    print("\n" + "="*60)
    print("TEST 1: Element-wise ops — CPU vs GPU forward+backward")
    print("="*60)

    np.random.seed(42)
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(2, 3).astype(np.float32)

    # CPU
    a_cpu = Tensor(a_np.copy(), is_leaf=True)
    b_cpu = Tensor(b_np.copy(), is_leaf=True)
    c_cpu = a_cpu * b_cpu + a_cpu
    loss_cpu = c_cpu.sum()
    autograd4nn(loss_cpu)

    # GPU
    a_gpu = Tensor(a_np.copy(), is_leaf=True, device="cuda")
    b_gpu = Tensor(b_np.copy(), is_leaf=True, device="cuda")
    c_gpu = a_gpu * b_gpu + a_gpu
    loss_gpu = c_gpu.sum()
    autograd4nn(loss_gpu)

    d_fwd = max_diff(c_cpu.value, c_gpu.value.to_host_f32())
    d_ga = max_diff(a_cpu.node.cp, a_gpu.node.cp)
    d_gb = max_diff(b_cpu.node.cp, b_gpu.node.cp)

    check("forward match", d_fwd < 1e-3, f"diff={d_fwd:.6f}")
    check("grad_a match", d_ga < 1e-3, f"diff={d_ga:.6f}")
    check("grad_b match", d_gb < 1e-3, f"diff={d_gb:.6f}")


# ══════════════════════════════════════════════════════════════
#  TEST 2: Matmul + bias forward + backward (CPU vs GPU vs PyTorch)
# ══════════════════════════════════════════════════════════════
def test_2_matmul_backward():
    print("\n" + "="*60)
    print("TEST 2: Matmul + bias forward+backward (CPU vs GPU vs PyTorch)")
    print("="*60)

    np.random.seed(42)
    x_np = np.random.randn(4, 3).astype(np.float32)
    W_np = np.random.randn(3, 2).astype(np.float32)
    b_np = np.zeros((1, 2), dtype=np.float32)

    # CPU
    x_c = Tensor(x_np.copy(), is_leaf=True)
    W_c = Tensor(W_np.copy(), is_leaf=True)
    b_c = Tensor(b_np.copy(), is_leaf=True)
    z_c = x_c.matmul(W_c) + b_c
    loss_c = z_c.sum()
    autograd4nn(loss_c)

    # GPU
    x_g = Tensor(x_np.copy(), is_leaf=True, device="cuda")
    W_g = Tensor(W_np.copy(), is_leaf=True, device="cuda")
    b_g = Tensor(b_np.copy(), is_leaf=True, device="cuda")
    z_g = x_g.matmul(W_g) + b_g
    loss_g = z_g.sum()
    autograd4nn(loss_g)

    d_fwd = max_diff(z_c.value, z_g.value.to_host_f32())
    d_gW = max_diff(W_c.node.cp, W_g.node.cp)
    d_gb = max_diff(b_c.node.cp, b_g.node.cp)

    check("forward CPU==GPU", d_fwd < 1e-2, f"diff={d_fwd:.6f}")
    check("grad_W CPU==GPU", d_gW < 1e-2, f"diff={d_gW:.6f}")
    check("grad_b CPU==GPU", d_gb < 1e-2, f"diff={d_gb:.6f}")
    print(z_c)
    print(z_g)
    print(loss_c)
    print(loss_g)

    # PyTorch comparison
    try:
        import torch
        x_t = torch.tensor(x_np.copy(), requires_grad=True)
        W_t = torch.tensor(W_np.copy(), requires_grad=True)
        b_t = torch.tensor(b_np.copy(), requires_grad=True)
        z_t = x_t @ W_t + b_t
        loss_t = z_t.sum()
        loss_t.backward()

        d_gW_pt = max_diff(W_c.node.cp, W_t.grad.numpy())
        d_gb_pt = max_diff(b_c.node.cp, b_t.grad.numpy())
        check("grad_W CPU==PyTorch", d_gW_pt < 1e-4, f"diff={d_gW_pt:.6f}")
        check("grad_b CPU==PyTorch", d_gb_pt < 1e-4, f"diff={d_gb_pt:.6f}")
    except ImportError:
        print("  (PyTorch not available for comparison)")


# ══════════════════════════════════════════════════════════════
#  TEST 3: Softmax + CCE loss (CPU vs GPU vs PyTorch)
# ══════════════════════════════════════════════════════════════
def test_3_softmax_cce():
    print("\n" + "="*60)
    print("TEST 3: Softmax + CCE loss (CPU vs GPU vs PyTorch)")
    print("="*60)

    np.random.seed(42)
    logits_np = np.random.randn(4, 3).astype(np.float32)
    y_np = np.zeros((4, 3), dtype=np.float32)
    y_np[0, 0] = y_np[1, 1] = y_np[2, 2] = y_np[3, 0] = 1.0

    loss_fn = Loss()

    # CPU
    logits_c = Tensor(logits_np.copy(), is_leaf=True)
    s_c = logits_c.softmax()
    y_c = Tensor(y_np.copy())
    loss_c = loss_fn.categorical_cross_entropy(s_c, y_c)
    autograd4nn(loss_c)
    loss_val_c = float(loss_c.value)
    grad_c = logits_c.node.cp.copy()

    # GPU
    logits_g = Tensor(logits_np.copy(), is_leaf=True, device="cuda")
    s_g = logits_g.softmax()
    y_g = Tensor(y_np.copy(), device="cuda")
    loss_g = loss_fn.categorical_cross_entropy(s_g, y_g)
    autograd4nn(loss_g)
    loss_val_g = float(loss_g.value.to_host_f32().ravel()[0])
    grad_g = logits_g.node.cp.to_host_f32()

    d_loss = s_c.value - s_g.value.to_host_f32()
    d_grad = max_diff(grad_c, grad_g)

    print(f"  CPU loss = {loss_val_c:.6f}")
    print(f"  GPU loss = {loss_val_g:.6f}")
    print(f"s_c: {s_c}")
    print(f"s_g: {s_g}")
    
    check("loss CPU==GPU", d_loss.sum() < 1e-3, f"diff={d_loss.sum()}")
    check("grad_logits CPU==GPU", d_grad < 1e-3, f"diff={d_grad:.6f}")
    check("loss is non-zero", loss_val_c > 0.01, f"loss={loss_val_c:.6f}")
    check("grads are non-zero", np.any(np.abs(grad_c) > 1e-6))

    # PyTorch comparison
    try:
        import torch
        import torch.nn.functional as F

        logits_t = torch.tensor(logits_np.copy(), requires_grad=True)
        s_t = F.softmax(logits_t, dim=-1)
        y_t = torch.tensor(y_np.copy())
        loss_t = (-y_t * torch.log(s_t + 1e-15)).sum(dim=-1).mean()
        loss_t.backward()

        d_loss_pt = abs(loss_val_c - loss_t.item())
        d_grad_pt = max_diff(grad_c, logits_t.grad.numpy())
        check("loss CPU==PyTorch", d_loss_pt < 1e-4, f"diff={d_loss_pt:.6f}")
        check("grad CPU==PyTorch", d_grad_pt < 1e-4, f"diff={d_grad_pt:.6f}")
    except ImportError:
        print("  (PyTorch not available for comparison)")


# ══════════════════════════════════════════════════════════════
#  TEST 4: Dense layer forward+backward with identical weights
# ══════════════════════════════════════════════════════════════
def test_4_dense_layer():
    print("\n" + "="*60)
    print("TEST 4: Dense layer forward+backward (same weights, CPU vs GPU vs PyTorch)")
    print("="*60)

    np.random.seed(42)
    W_np = np.random.randn(300, 2).astype(np.float32) * 0.1
    b_np = np.zeros((1, 2), dtype=np.float32)
    x_np = np.random.randn(4, 300).astype(np.float32)
    y_np = np.zeros((4, 2), dtype=np.float32)
    y_np[0, 0] = y_np[1, 1] = y_np[2, 0] = y_np[3, 1] = 1.0

    loss_fn = Loss()

    # ── CPU model ──
    model_cpu = Sequential([
        Input((3,)),
        Dense(300, 2, activation="softmax"),
    ], "cpu")
    model_cpu.model[1].set_weights(W_np.copy(), b_np.copy())

    x_c = Tensor(x_np.copy(), is_leaf=True)
    y_c = Tensor(y_np.copy())
    pred_c = model_cpu.forward(x_c)
    loss_c = loss_fn.categorical_cross_entropy(pred_c, y_c)
    model_cpu.zero_grad()
    autograd4nn(loss_c)
    W_cpu, b_cpu = model_cpu.model[1].get_weights()

    # ── GPU model ──
    model_gpu = Sequential([
        Input((3,)),
        Dense(300, 2, activation="softmax"),
    ], "cuda")
    # Set same weights AFTER moving to GPU
    model_gpu.model[1].set_weights(
        Tensor(W_np.copy(), is_leaf=True, device="cuda"),
        Tensor(b_np.copy(), is_leaf=True, device="cuda"),
    )

    x_g = Tensor(x_np.copy(), is_leaf=True, device="cuda")
    y_g = Tensor(y_np.copy(), device="cuda")
    pred_g = model_gpu.forward(x_g)
    loss_g = loss_fn.categorical_cross_entropy(pred_g, y_g)
    model_gpu.zero_grad()
    print(f"   GPU loss={loss_g}")
    autograd4nn(loss_g)
    print(f"   GPU loss={loss_g}")
    
    W_gpu, b_gpu = model_gpu.model[1].get_weights()

    loss_val_c = float(loss_c.value)
    ttc = ((-y_c*(pred_c.log())).sum(axis=-1)).mean()
    
    ttg = ((-y_g*(pred_g.log())).sum(axis=-1)).mean()
    print(ttg.value.to_host_f32)
    print(ttc.value)
    
    loss_val_g = float(loss_g.value.to_host_f32())

    grad_W_c = W_cpu.node.cp
    grad_b_c = b_cpu.node.cp
    grad_W_g = W_gpu.node.cp.to_host_f32()
    grad_b_g = b_gpu.node.cp.to_host_f32()

    d_loss = abs(loss_val_c - loss_val_g)
    d_gW = max_diff(grad_W_c, grad_W_g)
    d_gb = max_diff(grad_b_c, grad_b_g)

    # print(f"  CPU grad_W:\n{grad_W_c}")
    # print(f"  GPU grad_W:\n{grad_W_g}")

    check("loss CPU==GPU", d_loss < 1e-2, f"diff={d_loss:.6f}")
    check("grad_W CPU==GPU", d_gW < 1e-2, f"diff={d_gW:.6f}")
    check("grad_b CPU==GPU", d_gb < 1e-2, f"diff={d_gb:.6f}")
    check("loss is non-zero", loss_val_c > 0.01, f"loss={loss_val_c:.6f}")
    check("grad_W non-zero", np.any(np.abs(grad_W_c) > 1e-6))

    # PyTorch comparison
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

        d_gW_pt = max_diff(grad_W_c, W_t.grad.numpy())
        d_gb_pt = max_diff(grad_b_c, b_t.grad.numpy())
        d_loss_pt = abs(loss_val_c - loss_t.item())
        # print(f"  PyTorch loss={loss_t.item():.6f}")
        # print(f"  PyTorch grad_W:\n{W_t.grad.numpy()}")
        check("loss CPU==PyTorch", d_loss_pt < 1e-4, f"diff={d_loss_pt:.6f}")
        check("grad_W CPU==PyTorch", d_gW_pt < 1e-3, f"diff={d_gW_pt:.6f}")
        check("grad_b CPU==PyTorch", d_gb_pt < 1e-3, f"diff={d_gb_pt:.6f}")
    except ImportError:
        print("  (PyTorch not available for comparison)")


# ══════════════════════════════════════════════════════════════
#  TEST 5: MSE loss forward+backward (CPU vs GPU vs PyTorch)
# ══════════════════════════════════════════════════════════════
def test_5_mse_loss():
    print("\n" + "="*60)
    print("TEST 5: MSE loss forward+backward (CPU vs GPU vs PyTorch)")
    print("="*60)

    np.random.seed(42)
    pred_np = np.random.randn(4, 2).astype(np.float32)
    target_np = np.random.randn(4, 2).astype(np.float32)

    loss_fn = Loss()

    # CPU
    p_c = Tensor(pred_np.copy(), is_leaf=True)
    t_c = Tensor(target_np.copy())
    loss_c = loss_fn.mse(p_c, t_c)
    autograd4nn(loss_c)
    loss_val_c = float(loss_c.value)
    grad_c = p_c.node.cp.copy()

    # GPU
    p_g = Tensor(pred_np.copy(), is_leaf=True, device="cuda")
    t_g = Tensor(target_np.copy(), device="cuda")
    loss_g = loss_fn.mse(p_g, t_g)
    autograd4nn(loss_g)
    loss_val_g = float(loss_g.value.to_host_f32().ravel()[0])
    grad_g = p_g.node.cp.to_host_f32()

    d_loss = abs(loss_val_c - loss_val_g)
    d_grad = max_diff(grad_c, grad_g)

    print(f"  CPU loss={loss_val_c:.6f}, GPU loss={loss_val_g:.6f}")
    check("loss CPU==GPU", d_loss < 1e-2, f"diff={d_loss:.6f}")
    check("grad CPU==GPU", d_grad < 1e-2, f"diff={d_grad:.6f}")
    check("loss non-zero", loss_val_c > 0.01)

    try:
        import torch
        p_t = torch.tensor(pred_np.copy(), requires_grad=True)
        t_t = torch.tensor(target_np.copy())
        loss_t = ((p_t - t_t) ** 2).mean()
        loss_t.backward()

        d_loss_pt = abs(loss_val_c - loss_t.item())
        d_grad_pt = max_diff(grad_c, p_t.grad.numpy())
        check("loss CPU==PyTorch", d_loss_pt < 1e-4, f"diff={d_loss_pt:.6f}")
        check("grad CPU==PyTorch", d_grad_pt < 1e-4, f"diff={d_grad_pt:.6f}")
    except ImportError:
        print("  (PyTorch not available for comparison)")


# ══════════════════════════════════════════════════════════════
#  TEST 6: Sum/Mean backward on GPU
# ══════════════════════════════════════════════════════════════
def test_6_reduction_backward():
    print("\n" + "="*60)
    print("TEST 6: Sum/Mean reduction backward (CPU vs GPU)")
    print("="*60)

    np.random.seed(42)
    data_np = np.random.randn(3, 4).astype(np.float32)

    # Sum backward
    x_c = Tensor(data_np.copy(), is_leaf=True)
    s_c = x_c.sum()
    autograd4nn(s_c)

    x_g = Tensor(data_np.copy(), is_leaf=True, device="cuda")
    s_g = x_g.sum()
    autograd4nn(s_g)

    d_sum_fwd = max_diff(s_c.value, s_g.value)
    d_sum_grad = max_diff(x_c.node.cp, x_g.node.cp)
    check("sum fwd CPU==GPU", d_sum_fwd < 1e-3, f"diff={d_sum_fwd:.6f}")
    check("sum grad CPU==GPU", d_sum_grad < 1e-3, f"diff={d_sum_grad:.6f}")
    check("sum grad == 1.0", np.allclose(x_c.node.cp, 1.0),
          f"grad={x_c.node.cp.ravel()[:4]}...")

    # Mean backward
    x_c2 = Tensor(data_np.copy(), is_leaf=True)
    m_c = x_c2.mean()
    autograd4nn(m_c)

    x_g2 = Tensor(data_np.copy(), is_leaf=True, device="cuda")
    m_g = x_g2.mean()
    autograd4nn(m_g)

    expected_grad = 1.0 / data_np.size
    d_mean_fwd = max_diff(m_c.value, m_g.value)
    d_mean_grad = max_diff(x_c2.node.cp, x_g2.node.cp)
    check("mean fwd CPU==GPU", d_mean_fwd < 1e-3, f"diff={d_mean_fwd:.6f}")
    check("mean grad CPU==GPU", d_mean_grad < 1e-3, f"diff={d_mean_grad:.6f}")
    check(f"mean grad ≈ 1/N = {expected_grad:.6f}",
          np.allclose(x_c2.node.cp, expected_grad, atol=1e-5),
          f"actual={x_c2.node.cp.ravel()[0]:.6f}")

    # Sum along axis
    x_c3 = Tensor(data_np.copy(), is_leaf=True)
    sa_c = x_c3.sum(axis=-1)
    loss_c3 = sa_c.sum()
    autograd4nn(loss_c3)

    x_g3 = Tensor(data_np.copy(), is_leaf=True, device="cuda")
    sa_g = x_g3.sum(axis=-1)
    loss_g3 = sa_g.sum()
    autograd4nn(loss_g3)

    d_axis_grad = max_diff(x_c3.node.cp, x_g3.node.cp)
    check("sum(axis=-1) grad CPU==GPU", d_axis_grad < 1e-3, f"diff={d_axis_grad:.6f}")


# ══════════════════════════════════════════════════════════════
#  TEST 7: Sigmoid + MSE backward (CPU vs GPU vs PyTorch)
# ══════════════════════════════════════════════════════════════
def test_7_sigmoid_mse():
    print("\n" + "="*60)
    print("TEST 7: Sigmoid + MSE (CPU vs GPU vs PyTorch)")
    print("="*60)

    np.random.seed(42)
    W_np = np.random.randn(3, 2).astype(np.float32) * 0.1
    b_np = np.zeros((1, 2), dtype=np.float32)
    x_np = np.array([[0.5, 1.0, -0.5]], dtype=np.float32)
    y_np = np.array([[1.0, 0.0]], dtype=np.float32)
    loss_fn = Loss()

    # CPU
    x_c = Tensor(x_np.copy(), is_leaf=True)
    W_c = Tensor(W_np.copy(), is_leaf=True)
    b_c = Tensor(b_np.copy(), is_leaf=True)
    z_c = x_c.matmul(W_c) + b_c
    out_c = z_c.sigmoid()
    loss_c = loss_fn.mse(out_c, Tensor(y_np.copy()))
    autograd4nn(loss_c)

    # GPU
    x_g = Tensor(x_np.copy(), is_leaf=True, device="cuda")
    W_g = Tensor(W_np.copy(), is_leaf=True, device="cuda")
    b_g = Tensor(b_np.copy(), is_leaf=True, device="cuda")
    z_g = x_g.matmul(W_g) + b_g
    out_g = z_g.sigmoid()
    loss_g = loss_fn.mse(out_g, Tensor(y_np.copy(), device="cuda"))
    autograd4nn(loss_g)

    loss_val_c = float(loss_c.value)
    loss_val_g = float(loss_g.value.to_host_f32().ravel()[0])
    d_loss = abs(loss_val_c - loss_val_g)
    d_gW = max_diff(W_c.node.cp, W_g.node.cp)
    d_gb = max_diff(b_c.node.cp, b_g.node.cp)

    print(f"  CPU loss={loss_val_c:.6f}, GPU loss={loss_val_g:.6f}")
    check("loss CPU==GPU", d_loss < 1e-2, f"diff={d_loss:.6f}")
    check("grad_W CPU==GPU", d_gW < 1e-2, f"diff={d_gW:.6f}")
    check("grad_b CPU==GPU", d_gb < 1e-2, f"diff={d_gb:.6f}")

    try:
        import torch
        x_t = torch.tensor(x_np.copy(), requires_grad=True)
        W_t = torch.tensor(W_np.copy(), requires_grad=True)
        b_t = torch.tensor(b_np.copy(), requires_grad=True)
        z_t = x_t @ W_t + b_t
        out_t = torch.sigmoid(z_t)
        loss_t = ((out_t - torch.tensor(y_np)) ** 2).mean()
        loss_t.backward()

        d_gW_pt = max_diff(W_c.node.cp, W_t.grad.numpy())
        d_gb_pt = max_diff(b_c.node.cp, b_t.grad.numpy())
        check("grad_W CPU==PyTorch", d_gW_pt < 1e-4, f"diff={d_gW_pt:.6f}")
        check("grad_b CPU==PyTorch", d_gb_pt < 1e-4, f"diff={d_gb_pt:.6f}")
    except ImportError:
        print("  (PyTorch not available for comparison)")


# ══════════════════════════════════════════════════════════════
#  TEST 8: Training loop — loss should decrease (GPU)
# ══════════════════════════════════════════════════════════════
def test_8_training_loop():
    print("\n" + "="*60)
    print("TEST 8: Training loop — loss must decrease (GPU)")
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

    for epoch in range(20):
        X_batch = Tensor(x_data, is_leaf=True, device="cuda")
        y_batch = Tensor(y_data, device="cuda")
        pred = model.forward(X_batch)
        loss = loss_fn.categorical_cross_entropy(pred, y_batch)
        loss_val = float(loss.value.to_host_f32().ravel()[0])
        losses.append(loss_val)

        model.zero_grad()
        autograd4nn(loss)
        optimizer.step()

    print(f"  Epoch  1 loss: {losses[0]:.6f}")
    print(f"  Epoch 10 loss: {losses[9]:.6f}")
    print(f"  Epoch 20 loss: {losses[19]:.6f}")

    check("initial loss > 0", losses[0] > 0.01, f"loss={losses[0]:.6f}")
    check("loss decreased", losses[-1] < losses[0],
          f"{losses[0]:.4f} → {losses[-1]:.4f}")
    check("loss not zero", losses[-1] > 1e-10, f"final={losses[-1]:.6f}")


# ══════════════════════════════════════════════════════════════
#  TEST 9: Log + activations forward (CPU vs GPU)
# ══════════════════════════════════════════════════════════════
def test_9_activations():
    print("\n" + "="*60)
    print("TEST 9: Activation forward + backward (CPU vs GPU)")
    print("="*60)

    np.random.seed(42)
    data_np = np.abs(np.random.randn(2, 3).astype(np.float32)) + 0.1

    for act_name, act_fn in [("log", Tensor.log), ("sigmoid", Tensor.sigmoid),
                              ("relu", Tensor.relu), ("tanh", Tensor.tanh)]:
        x_c = Tensor(data_np.copy(), is_leaf=True)
        y_c = act_fn(x_c)
        l_c = y_c.sum()
        autograd4nn(l_c)

        x_g = Tensor(data_np.copy(), is_leaf=True, device="cuda")
        y_g = act_fn(x_g)
        l_g = y_g.sum()
        autograd4nn(l_g)

        d_fwd = max_diff(y_c.value, y_g.value)
        d_grad = max_diff(x_c.node.cp, x_g.node.cp)
        check(f"{act_name} fwd CPU==GPU", d_fwd < 1e-3, f"diff={d_fwd:.6f}")
        check(f"{act_name} grad CPU==GPU", d_grad < 1e-3, f"diff={d_grad:.6f}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Seera ML Framework — GPU vs CPU vs PyTorch Test Suite  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    test_0_cuten_no_inplace_mutation()
    test_1_elementwise_ops()
    test_2_matmul_backward()
    test_3_softmax_cce()
    test_4_dense_layer()
    test_5_mse_loss()
    test_6_reduction_backward()
    test_7_sigmoid_mse()
    test_8_training_loop()
    test_9_activations()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
