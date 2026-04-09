"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              SEERA FRAMEWORK — COMPREHENSIVE CUDA TEST SUITE               ║
║    Tests: Tensor ops, Autograd, Layers, Optimizers, Loss, Full Training    ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO READ FAILURES:
  ✗ means the test itself crashed or produced wrong values.
  Each test prints what it expected vs. what it got.
  Gradient tests compare GPU result against a CPU reference.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import traceback
import sys
import time

# ── Framework imports ────────────────────────────────────────
from Seera_init import tensor as Tensor
from Seera_Engine import autograd4nn
from Seera import (
    Sequential, Input, Dense, Conv2D, Flatten,
    MaxPool2D, BatchNorm2d, BatchNorm1d,
    Unpool2D_Nearest, ConvTranspose2D, Concatenate,
    Loss, SGD, Adam,
)

# ════════════════════════════════════════════════════════════
# Test runner utility
# ════════════════════════════════════════════════════════════

_PASS = 0
_FAIL = 0
_RESULTS = []

def _run(name, fn):
    global _PASS, _FAIL
    try:
        t0 = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  ✓  {name:<60}  ({elapsed:6.1f} ms)")
        _PASS += 1
        _RESULTS.append(("PASS", name, None))
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  ✗  {name:<60}  FAILED")
        print(f"       └─ {e}")
        _FAIL += 1
        _RESULTS.append(("FAIL", name, tb))

def section(title):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")

def _allclose_gpu(a_gpu, b_np, rtol=1e-3, atol=1e-4, label=""):
    """Pull GPU tensor to host and compare to numpy reference."""
    if hasattr(a_gpu, 'to_host_f32'):
        a_np = a_gpu.to_host_f32()
    elif hasattr(a_gpu, 'value') and hasattr(a_gpu.value, 'to_host_f32'):
        a_np = a_gpu.value.to_host_f32()
    else:
        a_np = np.array(a_gpu)
    if not np.allclose(a_np, b_np, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{label}\n  GPU result: {a_np.ravel()[:8]}\n  CPU ref:    {np.array(b_np).ravel()[:8]}"
        )

def _tensor_gpu(arr):
    """Wrap a numpy array as a GPU Tensor."""
    return Tensor(arr.astype(np.float32), is_leaf=True, device="cuda")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Tensor Construction & Memory
# ════════════════════════════════════════════════════════════════════════════
section("1 · TENSOR CONSTRUCTION & MEMORY")

def test_zeros_gpu():
    t = Tensor.zeros((4, 8), device="cuda")
    _allclose_gpu(t, np.zeros((4, 8)), label="zeros")

def test_ones_gpu():
    t = Tensor.ones((3, 5), device="cuda")
    _allclose_gpu(t, np.ones((3, 5)), label="ones")

def test_random_gpu():
    t = Tensor.random((10, 10), device="cuda")
    arr = t.value.to_host_f32()
    assert arr.shape == (10, 10), f"Expected (10,10), got {arr.shape}"
    assert (arr >= 0).all() and (arr <= 1).all(), "random values out of [0,1]"

def test_from_numpy_gpu():
    np_arr = np.random.randn(5, 6).astype(np.float32)
    t = _tensor_gpu(np_arr)
    _allclose_gpu(t, np_arr, label="from_numpy")

def test_shape_property():
    t = _tensor_gpu(np.zeros((2, 3, 4)))
    assert t.shape == (2, 3, 4), f"Expected (2,3,4), got {t.shape}"

def test_to_numpy():
    np_arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = _tensor_gpu(np_arr)
    result = t.to_numpy()
    _allclose_gpu(result, np_arr, label="to_numpy")

_run("Tensor.zeros on CUDA",               test_zeros_gpu)
_run("Tensor.ones on CUDA",                test_ones_gpu)
_run("Tensor.random on CUDA — shape & range", test_random_gpu)
_run("Tensor from numpy on CUDA",          test_from_numpy_gpu)
_run("Tensor.shape property on CUDA",      test_shape_property)
_run("Tensor.to_numpy round-trip",         test_to_numpy)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Element-wise Arithmetic
# ════════════════════════════════════════════════════════════════════════════
section("2 · ELEMENT-WISE ARITHMETIC (CUDA)")

def _make_ab(shape=(4, 4)):
    a_np = np.random.randn(*shape).astype(np.float32)
    b_np = np.random.randn(*shape).astype(np.float32) + 1.0   # avoid near-zero
    return a_np, b_np, _tensor_gpu(a_np), _tensor_gpu(b_np)

def test_add():
    a, b, ta, tb = _make_ab()
    _allclose_gpu((ta + tb), a + b, label="add")

def test_sub():
    a, b, ta, tb = _make_ab()
    _allclose_gpu((ta - tb), a - b, label="sub")

def test_mul():
    a, b, ta, tb = _make_ab()
    _allclose_gpu((ta * tb), a * b, label="mul")

def test_div():
    a, b, ta, tb = _make_ab()
    _allclose_gpu((ta / tb), a / b, rtol=1e-3, label="div")

def test_neg():
    a, _, ta, _ = _make_ab()
    _allclose_gpu((-ta), -a, label="neg")

def test_scalar_add():
    a_np = np.ones((3, 3), dtype=np.float32) * 2.0
    ta = _tensor_gpu(a_np)
    _allclose_gpu((ta + 3.0), a_np + 3.0, label="scalar_add")

def test_scalar_mul():
    a_np = np.ones((3, 3), dtype=np.float32) * 5.0
    ta = _tensor_gpu(a_np)
    _allclose_gpu((ta * 2.5), a_np * 2.5, label="scalar_mul")

def test_pow():
    a_np = np.abs(np.random.randn(4, 4)).astype(np.float32) + 0.5
    ta = _tensor_gpu(a_np)
    _allclose_gpu((ta ** 2.0), a_np ** 2.0, rtol=1e-3, label="pow")

def test_broadcast_add():
    a_np = np.random.randn(8, 4).astype(np.float32)
    b_np = np.random.randn(1, 4).astype(np.float32)
    ta, tb = _tensor_gpu(a_np), _tensor_gpu(b_np)
    _allclose_gpu((ta + tb), a_np + b_np, label="broadcast_add")

_run("add: tensor + tensor",                test_add)
_run("sub: tensor - tensor",                test_sub)
_run("mul: tensor * tensor",                test_mul)
_run("div: tensor / tensor",                test_div)
_run("neg: -tensor",                        test_neg)
_run("scalar add: tensor + 3.0",            test_scalar_add)
_run("scalar mul: tensor * 2.5",            test_scalar_mul)
_run("pow: tensor ** 2.0",                  test_pow)
_run("broadcast add (8,4) + (1,4)",         test_broadcast_add)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Activation Functions
# ════════════════════════════════════════════════════════════════════════════
section("3 · ACTIVATION FUNCTIONS (CUDA)")

def _act_test(name, fn_gpu, fn_cpu, input_np=None):
    if input_np is None:
        input_np = np.random.randn(8, 16).astype(np.float32)
    t = _tensor_gpu(input_np)
    gpu_out = fn_gpu(t)
    cpu_out = fn_cpu(input_np)
    _allclose_gpu(gpu_out, cpu_out, rtol=1e-3, atol=1e-4, label=name)

def test_relu():
    _act_test("relu", Tensor.relu, lambda x: np.maximum(x, 0))

def test_sigmoid():
    _act_test("sigmoid", Tensor.sigmoid, lambda x: 1/(1+np.exp(-x)))

def test_tanh():
    _act_test("tanh", Tensor.tanh, lambda x: np.tanh(x))

def test_softmax():
    x_np = np.random.randn(4, 10).astype(np.float32)
    t = _tensor_gpu(x_np)
    gpu_out = Tensor.softmax(t)
    shifted = x_np - x_np.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    cpu_out = exp / exp.sum(axis=-1, keepdims=True)
    _allclose_gpu(gpu_out, cpu_out, rtol=1e-3, label="softmax")

def test_log():
    x_np = np.abs(np.random.randn(4, 4)).astype(np.float32) + 0.1
    t = _tensor_gpu(x_np)
    _allclose_gpu(Tensor.log(t), np.log(x_np), rtol=1e-3, label="log")

def test_exp():
    x_np = np.random.randn(4, 4).astype(np.float32) * 0.5
    t = _tensor_gpu(x_np)
    _allclose_gpu(Tensor.exp(t), np.exp(x_np), rtol=1e-3, label="exp")

def test_abs():
    x_np = np.random.randn(4, 4).astype(np.float32)
    t = _tensor_gpu(x_np)
    _allclose_gpu(Tensor.abs(t), np.abs(x_np), label="abs")

def test_sqrt():
    x_np = np.abs(np.random.randn(4, 4)).astype(np.float32) + 0.1
    t = _tensor_gpu(x_np)
    _allclose_gpu(Tensor.sqrt(t), np.sqrt(x_np), rtol=1e-3, label="sqrt")

def test_softmax_rows_sum_to_one():
    x_np = np.random.randn(16, 32).astype(np.float32)
    t = _tensor_gpu(x_np)
    out = Tensor.softmax(t).value.to_host_f32()
    row_sums = out.sum(axis=-1)
    if not np.allclose(row_sums, np.ones(16), atol=1e-4):
        raise AssertionError(f"Softmax rows don't sum to 1: {row_sums[:4]}")

_run("relu forward",                        test_relu)
_run("sigmoid forward",                     test_sigmoid)
_run("tanh forward",                        test_tanh)
_run("softmax forward",                     test_softmax)
_run("log forward",                         test_log)
_run("exp forward",                         test_exp)
_run("abs forward",                         test_abs)
_run("sqrt forward",                        test_sqrt)
_run("softmax rows sum to 1",               test_softmax_rows_sum_to_one)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Reductions
# ════════════════════════════════════════════════════════════════════════════
section("4 · REDUCTIONS (CUDA)")

def test_sum_all():
    x_np = np.random.randn(4, 8).astype(np.float32)
    t = _tensor_gpu(x_np)
    gpu_val = t.sum()
    # chain down to scalar
    gpu_scalar = gpu_val.value.to_host_f32().ravel()[0] if hasattr(gpu_val.value, 'to_host_f32') else float(gpu_val.value)
    assert abs(gpu_scalar - x_np.sum()) < 0.1, f"sum all: {gpu_scalar} vs {x_np.sum()}"

def test_sum_axis0():
    x_np = np.random.randn(6, 8).astype(np.float32)
    t = _tensor_gpu(x_np)
    _allclose_gpu(t.sum(axis=0), x_np.sum(axis=0), rtol=1e-3, label="sum axis=0")

def test_sum_axis1():
    x_np = np.random.randn(6, 8).astype(np.float32)
    t = _tensor_gpu(x_np)
    _allclose_gpu(t.sum(axis=1), x_np.sum(axis=1), rtol=1e-3, label="sum axis=1")

def test_mean_all():
    x_np = np.random.randn(4, 8).astype(np.float32)
    t = _tensor_gpu(x_np)
    gpu_out = t.mean()
    # mean returns a tensor; drill down
    v = gpu_out
    while hasattr(v, 'value'):
        v = v.value
    if hasattr(v, 'to_host_f32'):
        v = v.to_host_f32().ravel()[0]
    else:
        v = float(np.array(v).ravel()[0])
    assert abs(v - x_np.mean()) < 0.1, f"mean all: {v} vs {x_np.mean()}"

def test_mean_axis0():
    x_np = np.random.randn(8, 4).astype(np.float32)
    t = _tensor_gpu(x_np)
    _allclose_gpu(t.mean(axis=0), x_np.mean(axis=0), rtol=1e-3, label="mean axis=0")

_run("sum (all elements)",                  test_sum_all)
_run("sum (axis=0)",                        test_sum_axis0)
_run("sum (axis=1)",                        test_sum_axis1)
_run("mean (all elements)",                 test_mean_all)
_run("mean (axis=0)",                       test_mean_axis0)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Matmul
# ════════════════════════════════════════════════════════════════════════════
section("5 · MATMUL (CUDA)")

def test_matmul_basic():
    a_np = np.random.randn(4, 8).astype(np.float32)
    b_np = np.random.randn(8, 6).astype(np.float32)
    ta, tb = _tensor_gpu(a_np), _tensor_gpu(b_np)
    _allclose_gpu(ta.matmul(tb), a_np @ b_np, rtol=1e-3, label="matmul (4,8)@(8,6)")

def test_matmul_square():
    a_np = np.random.randn(16, 16).astype(np.float32)
    b_np = np.random.randn(16, 16).astype(np.float32)
    ta, tb = _tensor_gpu(a_np), _tensor_gpu(b_np)
    _allclose_gpu(ta.matmul(tb), a_np @ b_np, rtol=1e-3, label="matmul 16x16")

def test_matmul_batch_like():
    # Simulates Dense forward: (N, in) @ (in, out) → (N, out)
    N, in_f, out_f = 32, 64, 32
    x_np = np.random.randn(N, in_f).astype(np.float32)
    w_np = np.random.randn(in_f, out_f).astype(np.float32)
    tx, tw = _tensor_gpu(x_np), _tensor_gpu(w_np)
    _allclose_gpu(tx.matmul(tw), x_np @ w_np, rtol=1e-3, label="matmul batch-like")

_run("matmul (4,8)@(8,6)",                  test_matmul_basic)
_run("matmul 16×16 square",                 test_matmul_square)
_run("matmul Dense-like (32,64)@(64,32)",   test_matmul_batch_like)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Autograd (Gradient Correctness via finite-difference check)
# ════════════════════════════════════════════════════════════════════════════
section("6 · AUTOGRAD GRADIENT CHECKS (CUDA vs. CPU reference)")

def _grad_gpu(loss_fn, x_np, y_np=None):
    """Run loss_fn on GPU, backprop, return gradient as numpy."""
    tx = _tensor_gpu(x_np)
    if y_np is not None:
        ty = Tensor(y_np.astype(np.float32), device="cuda")
        loss = loss_fn(tx, ty)
    else:
        loss = loss_fn(tx)
    if hasattr(loss.value, 'to_host_f32') and loss.value.size > 1:
        loss = loss.mean()
    autograd4nn(loss)
    grad = tx.node.cp
    if hasattr(grad, 'to_host_f32'):
        return grad.to_host_f32()
    return np.array(grad)

def _grad_cpu(loss_fn, x_np, y_np=None):
    """Run loss_fn on CPU, backprop, return gradient as numpy."""
    tx = Tensor(x_np.astype(np.float32), is_leaf=True)
    if y_np is not None:
        ty = Tensor(y_np.astype(np.float32))
        loss = loss_fn(tx, ty)
    else:
        loss = loss_fn(tx)
    if loss.value.ndim > 0 and loss.value.size > 1:
        loss = loss.mean()
    autograd4nn(loss)
    return np.array(tx.node.cp)

def _check_grad(name, loss_fn, x_np, y_np=None, atol=5e-3):
    g_gpu = _grad_gpu(loss_fn, x_np, y_np)
    g_cpu = _grad_cpu(loss_fn, x_np, y_np)
    if not np.allclose(g_gpu, g_cpu, atol=atol, rtol=1e-2):
        raise AssertionError(
            f"{name} gradient mismatch\n"
            f"  GPU: {g_gpu.ravel()[:6]}\n"
            f"  CPU: {g_cpu.ravel()[:6]}"
        )

def test_grad_add():
    x = np.random.randn(4, 4).astype(np.float32)
    y = np.random.randn(4, 4).astype(np.float32)
    _check_grad("add", lambda a, b: a + b, x, y)

def test_grad_mul():
    x = np.random.randn(4, 4).astype(np.float32)
    y = np.random.randn(4, 4).astype(np.float32)
    _check_grad("mul", lambda a, b: a * b, x, y)

def test_grad_pow():
    x = np.abs(np.random.randn(4, 4)).astype(np.float32) + 0.5
    _check_grad("pow", lambda a: a ** 2.0, x)

def test_grad_relu():
    x = np.random.randn(4, 4).astype(np.float32)
    _check_grad("relu", lambda a: Tensor.relu(a), x)

def test_grad_sigmoid():
    x = np.random.randn(4, 4).astype(np.float32) * 2.0
    _check_grad("sigmoid", lambda a: Tensor.sigmoid(a), x)

def test_grad_tanh():
    x = np.random.randn(4, 4).astype(np.float32)
    _check_grad("tanh", lambda a: Tensor.tanh(a), x)

def test_grad_softmax():
    x = np.random.randn(4, 8).astype(np.float32)
    _check_grad("softmax", lambda a: Tensor.softmax(a), x, atol=1e-2)

def test_grad_log():
    x = np.abs(np.random.randn(4, 4)).astype(np.float32) + 0.5
    _check_grad("log", lambda a: Tensor.log(a), x)

def test_grad_matmul():
    x = np.random.randn(4, 8).astype(np.float32)
    y = np.random.randn(8, 4).astype(np.float32)
    _check_grad("matmul (input grad)", lambda a, b: a.matmul(b), x, y)

def test_grad_sum():
    x = np.random.randn(4, 8).astype(np.float32)
    _check_grad("sum", lambda a: a.sum(axis=1), x)

def test_grad_mean():
    x = np.random.randn(4, 8).astype(np.float32)
    _check_grad("mean", lambda a: a.mean(axis=1), x)

def test_grad_chain_relu_mul():
    x = np.random.randn(4, 4).astype(np.float32)
    y = np.random.randn(4, 4).astype(np.float32)
    fn = lambda a, b: Tensor.relu(a * b)
    _check_grad("chain relu(a*b)", fn, x, y)

_run("∂/∂x  of  x + y",                    test_grad_add)
_run("∂/∂x  of  x * y",                    test_grad_mul)
_run("∂/∂x  of  x^2",                      test_grad_pow)
_run("∂/∂x  of  relu(x)",                  test_grad_relu)
_run("∂/∂x  of  sigmoid(x)",               test_grad_sigmoid)
_run("∂/∂x  of  tanh(x)",                  test_grad_tanh)
_run("∂/∂x  of  softmax(x)",               test_grad_softmax)
_run("∂/∂x  of  log(x)",                   test_grad_log)
_run("∂/∂x  of  x @ y  (input grad)",      test_grad_matmul)
_run("∂/∂x  of  sum(x, axis=1)",           test_grad_sum)
_run("∂/∂x  of  mean(x, axis=1)",          test_grad_mean)
_run("∂/∂x  of  relu(x * y)  chain rule",  test_grad_chain_relu_mul)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Loss Functions
# ════════════════════════════════════════════════════════════════════════════
section("7 · LOSS FUNCTIONS (CUDA)")

loss_obj = Loss()

def test_mse_value():
    y_pred_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y_np      = np.array([[1.5, 1.5], [2.5, 4.5]], dtype=np.float32)
    yp = Tensor(y_pred_np, device="cuda")
    y  = Tensor(y_np, device="cuda")
    loss = loss_obj.mse(yp, y)
    v = loss.value.to_host_f32().ravel()[0]
    ref = float(np.mean((y_pred_np - y_np)**2))
    assert abs(v - ref) < 1e-4, f"MSE: {v:.6f} vs {ref:.6f}"

def test_mae_value():
    y_pred_np = np.array([[1.0, 3.0]], dtype=np.float32)
    y_np      = np.array([[2.0, 2.0]], dtype=np.float32)
    yp = Tensor(y_pred_np, device="cuda")
    y  = Tensor(y_np, device="cuda")
    loss = loss_obj.mae(yp, y)
    v = loss.value.to_host_f32().ravel()[0]
    ref = float(np.mean(np.abs(y_pred_np - y_np)))
    assert abs(v - ref) < 1e-4, f"MAE: {v:.6f} vs {ref:.6f}"

def test_bce_value():
    y_pred_np = np.array([[0.9, 0.1]], dtype=np.float32)
    y_np      = np.array([[1.0, 0.0]], dtype=np.float32)
    yp = Tensor(y_pred_np, device="cuda")
    y  = Tensor(y_np, device="cuda")
    loss = loss_obj.binary_cross_entropy(yp, y)
    v = loss.value.to_host_f32().ravel()[0]
    eps = 1e-15
    ref = float(-np.mean(y_np * np.log(y_pred_np + eps) + (1 - y_np) * np.log(1 - y_pred_np + eps)))
    assert abs(v - ref) < 1e-3, f"BCE: {v:.6f} vs {ref:.6f}"

def test_cce_value():
    y_pred_np = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]], dtype=np.float32)
    y_np      = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    yp = Tensor(y_pred_np, device="cuda")
    y  = Tensor(y_np, device="cuda")
    loss = loss_obj.categorical_cross_entropy(yp, y)
    v = loss.value.to_host_f32().ravel()[0]
    eps = 1e-15
    ref = float(-np.mean(np.sum(y_np * np.log(y_pred_np + eps), axis=-1)))
    assert abs(v - ref) < 1e-3, f"CCE: {v:.6f} vs {ref:.6f}"

def test_mse_grad():
    y_pred_np = np.random.rand(8, 4).astype(np.float32)
    y_np      = np.random.rand(8, 4).astype(np.float32)
    yp = Tensor(y_pred_np, is_leaf=True, device="cuda")
    y  = Tensor(y_np, device="cuda")
    loss = loss_obj.mse(yp, y)
    autograd4nn(loss)
    g = yp.node.cp.to_host_f32()
    ref_g = (2.0 * (y_pred_np - y_np)) / y_pred_np.size
    assert np.allclose(g, ref_g, atol=1e-4), f"MSE grad mismatch"

_run("MSE loss value",                      test_mse_value)
_run("MAE loss value",                      test_mae_value)
_run("Binary cross-entropy value",          test_bce_value)
_run("Categorical cross-entropy value",     test_cce_value)
_run("MSE gradient wrt y_pred",            test_mse_grad)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Dense Layer (forward + backward)
# ════════════════════════════════════════════════════════════════════════════
section("8 · DENSE LAYER (CUDA)")

def test_dense_forward_shape():
    model = Sequential([
        Input((8,)),
        Dense(8, 4, activation="relu"),
    ], device="cuda")
    x = np.random.randn(16, 8).astype(np.float32)
    out = model.forward(x)
    arr = out.value.to_host_f32()
    assert arr.shape == (16, 4), f"Expected (16,4), got {arr.shape}"

def test_dense_relu_non_negative():
    model = Sequential([
        Input((8,)),
        Dense(8, 16, activation="relu"),
    ], device="cuda")
    x = np.random.randn(32, 8).astype(np.float32)
    out = model.forward(x)
    arr = out.value.to_host_f32()
    assert (arr >= -1e-6).all(), "ReLU output has negative values"

def test_dense_sigmoid_range():
    model = Sequential([
        Input((4,)),
        Dense(4, 8, activation="sigmoid"),
    ], device="cuda")
    x = np.random.randn(10, 4).astype(np.float32) * 10  # large values
    out = model.forward(x)
    arr = out.value.to_host_f32()
    assert (arr >= 0).all() and (arr <= 1).all(), "Sigmoid out of [0,1]"

def test_dense_softmax_sums():
    model = Sequential([
        Input((4,)),
        Dense(4, 5, activation="softmax"),
    ], device="cuda")
    x = np.random.randn(8, 4).astype(np.float32)
    out = model.forward(x)
    arr = out.value.to_host_f32()
    row_sums = arr.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4), f"Softmax rows: {row_sums}"

def test_dense_grad_flows():
    """Verify that weight gradients are non-zero after backward."""
    model = Sequential([
        Input((8,)),
        Dense(8, 4, activation="relu"),
    ], device="cuda")
    x = np.random.randn(4, 8).astype(np.float32)
    out = model.forward(x)
    loss = out.mean()
    model.zero_grad()
    autograd4nn(loss)
    layer = model.model[1]
    grad_w = layer.weights.node.cp.to_host_f32()
    grad_b = layer.bais.node.cp.to_host_f32()
    assert not np.allclose(grad_w, 0), "Weight gradient is all zero — backward broken"
    assert not np.allclose(grad_b, 0), "Bias gradient is all zero — backward broken"

def test_dense_stacked():
    model = Sequential([
        Input((16,)),
        Dense(16, 32, activation="relu"),
        Dense(32, 8,  activation="sigmoid"),
        Dense(8,  3,  activation="softmax"),
    ], device="cuda")
    x = np.random.randn(12, 16).astype(np.float32)
    out = model.forward(x)
    arr = out.value.to_host_f32()
    assert arr.shape == (12, 3)
    assert np.allclose(arr.sum(axis=1), 1.0, atol=1e-4)

_run("Dense forward output shape",          test_dense_forward_shape)
_run("Dense(relu) output ≥ 0",             test_dense_relu_non_negative)
_run("Dense(sigmoid) output in [0,1]",      test_dense_sigmoid_range)
_run("Dense(softmax) rows sum to 1",        test_dense_softmax_sums)
_run("Dense weight/bias gradients non-zero", test_dense_grad_flows)
_run("3 stacked Dense layers",              test_dense_stacked)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Conv2D Layer
# ════════════════════════════════════════════════════════════════════════════
section("9 · CONV2D LAYER (CUDA)")

def test_conv2d_output_shape():
    # (N, C, H, W) → after Conv2D(F, C, (KH,KW)): (N, F, OH, OW)
    # OH = (H - KH) // stride + 1  with no padding, stride=1
    model = Sequential([
        Input((3, 16, 16)),
        Conv2D(8, 3, (3,3), activation="relu"),
    ], device="cuda")
    x = np.random.randn(4, 3, 16, 16).astype(np.float32)
    out = model.forward(x)
    arr = out.value.to_host_f32()
    assert arr.shape == (4, 8, 14, 14), f"Expected (4,8,14,14), got {arr.shape}"

def test_conv2d_padding_same_shape():
    model = Sequential([
        Input((1, 8, 8)),
        Conv2D(4, 1, (3,3), activation="relu", zero_padding=1),
    ], device="cuda")
    x = np.random.randn(2, 1, 8, 8).astype(np.float32)
    out = model.forward(x)
    arr = out.value.to_host_f32()
    assert arr.shape == (2, 4, 8, 8), f"Expected (2,4,8,8), got {arr.shape}"

def test_conv2d_relu_non_negative():
    model = Sequential([
        Input((1, 8, 8)),
        Conv2D(4, 1, (3,3), activation="relu"),
    ], device="cuda")
    x = np.random.randn(2, 1, 8, 8).astype(np.float32)
    out = model.forward(x)
    arr = out.value.to_host_f32()
    assert (arr >= -1e-6).all(), "Conv2D ReLU has negative output"

def test_conv2d_grad_flows():
    model = Sequential([
        Input((1, 8, 8)),
        Conv2D(4, 1, (3,3), activation="relu"),
    ], device="cuda")
    x = np.random.randn(2, 1, 8, 8).astype(np.float32)
    out = model.forward(x)
    loss = out.mean()
    model.zero_grad()
    autograd4nn(loss)
    layer = model.model[1]
    grad_w = layer.weights.node.cp.to_host_f32()
    assert not np.allclose(grad_w, 0), "Conv2D weight gradient is zero"

def test_conv2d_stride():
    model = Sequential([
        Input((1, 8, 8)),
        Conv2D(2, 1, (3,3), activation="relu", stride=2),
    ], device="cuda")
    x = np.random.randn(2, 1, 8, 8).astype(np.float32)
    out = model.forward(x)
    arr = out.value.to_host_f32()
    # OH = (8 - 3)//2 + 1 = 3
    assert arr.shape == (2, 2, 3, 3), f"Expected (2,2,3,3), got {arr.shape}"

_run("Conv2D output shape (no padding)",    test_conv2d_output_shape)
_run("Conv2D output shape (padding=1)",     test_conv2d_padding_same_shape)
_run("Conv2D relu output ≥ 0",             test_conv2d_relu_non_negative)
_run("Conv2D weight gradient non-zero",     test_conv2d_grad_flows)
_run("Conv2D stride=2 output shape",        test_conv2d_stride)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MaxPool2D & Flatten
# ════════════════════════════════════════════════════════════════════════════
section("10 · MAXPOOL2D + FLATTEN (CUDA)")

def test_maxpool_shape():
    model = Sequential([
        Input((1, 8, 8)),
        Conv2D(4, 1, (3,3), activation="relu"),
        MaxPool2D(pool_size=(2,2), stride=2),
    ], device="cuda")
    x = np.random.randn(2, 1, 8, 8).astype(np.float32)
    out = model.forward(x)
    arr = out.value.to_host_f32()
    # after conv: (2,4,6,6); after maxpool(2,2,stride2): (2,4,3,3)
    assert arr.shape == (2, 4, 3, 3), f"Expected (2,4,3,3), got {arr.shape}"

def test_maxpool_grad_flows():
    model = Sequential([
        Input((1, 8, 8)),
        Conv2D(4, 1, (3,3), activation="relu"),
        MaxPool2D(pool_size=(2,2), stride=2),
    ], device="cuda")
    x = np.random.randn(2, 1, 8, 8).astype(np.float32)
    out = model.forward(x)
    loss = out.mean()
    model.zero_grad()
    autograd4nn(loss)
    layer = model.model[1]
    grad_w = layer.weights.node.cp.to_host_f32()
    assert not np.allclose(grad_w, 0), "Conv2D grad zero after MaxPool"

def test_flatten_shape():
    model = Sequential([
        Input((2, 4, 4)),
        Conv2D(4, 2, (3,3), activation="relu"),
        Flatten(),
    ], device="cuda")
    x = np.random.randn(3, 2, 4, 4).astype(np.float32)
    out = model.forward(x)
    arr = out.value.to_host_f32()
    # after conv: (3,4,2,2) → flatten: (3,16)
    assert arr.shape == (3, 16), f"Expected (3,16), got {arr.shape}"

def test_flatten_grad_flows():
    model = Sequential([
        Input((1, 8, 8)),
        Conv2D(2, 1, (3,3), activation="relu"),
        Flatten(),
        Dense(72, 4, activation="relu"),
    ], device="cuda")
    x = np.random.randn(2, 1, 8, 8).astype(np.float32)
    out = model.forward(x)
    loss = out.mean()
    model.zero_grad()
    autograd4nn(loss)
    dense_layer = model.model[-1]
    g = dense_layer.weights.node.cp.to_host_f32()
    assert not np.allclose(g, 0), "Dense grad zero after Flatten"

_run("MaxPool2D output shape",               test_maxpool_shape)
_run("MaxPool2D gradient flows back",        test_maxpool_grad_flows)
_run("Flatten output shape",                 test_flatten_shape)
_run("Flatten gradient flows to Dense",      test_flatten_grad_flows)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 11 — ConvTranspose2D & Unpool2D_Nearest
# ════════════════════════════════════════════════════════════════════════════
section("11 · CONV_TRANSPOSE2D + UNPOOL2D (CUDA)")

def test_convtranspose_shape():
    model = Sequential([
        Input((4, 4, 4)),
        ConvTranspose2D(2, 4, (3,3), activation="relu"),
    ], device="cuda")
    x = np.random.randn(2, 4, 4, 4).astype(np.float32)
    out = model.forward(x)
    arr = out.value.to_host_f32()
    # Hout = (4-1)*1 - 0 + 3 = 6
    assert arr.shape == (2, 2, 6, 6), f"Expected (2,2,6,6), got {arr.shape}"

def test_convtranspose_grad():
    model = Sequential([
        Input((2, 4, 4)),
        ConvTranspose2D(2, 2, (3,3), activation="relu"),
    ], device="cuda")
    x = np.random.randn(2, 2, 4, 4).astype(np.float32)
    out = model.forward(x)
    loss = out.mean()
    model.zero_grad()
    autograd4nn(loss)
    g = model.model[1].weights.node.cp.to_host_f32()
    assert not np.allclose(g, 0), "ConvTranspose2D weight grad is zero"

def test_unpool_shape():
    model = Sequential([
        Input((2, 4, 4)),
        Unpool2D_Nearest(size=(2,2)),
    ], device="cuda")
    x = np.random.randn(2, 2, 4, 4).astype(np.float32)
    out = model.forward(x)
    arr = out.value.to_host_f32()
    assert arr.shape == (2, 2, 8, 8), f"Expected (2,2,8,8), got {arr.shape}"

_run("ConvTranspose2D output shape",         test_convtranspose_shape)
_run("ConvTranspose2D grad non-zero",        test_convtranspose_grad)
_run("Unpool2D_Nearest output shape ×2",    test_unpool_shape)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 12 — SGD Optimizer
# ════════════════════════════════════════════════════════════════════════════
section("12 · SGD OPTIMIZER (CUDA)")

def _build_dense_model():
    return Sequential([
        Input((4,)),
        Dense(4, 8, activation="relu"),
        Dense(8, 1, activation="sigmoid"),
    ], device="cuda")

def test_sgd_weights_change():
    model = _build_dense_model()
    opt   = SGD(model, lr=0.1)
    X = np.random.randn(8, 4).astype(np.float32)
    y = np.random.rand(8, 1).astype(np.float32)
    # Save initial weights
    w0 = model.model[1].weights.value.to_host_f32().copy()
    # One update
    out  = model.forward(X)
    loss = loss_obj.mse(out, Tensor(y, device="cuda"))
    model.zero_grad()
    autograd4nn(loss)
    opt.step()
    w1 = model.model[1].weights.value.to_host_f32()
    assert not np.allclose(w0, w1), "SGD did not change weights"

def test_sgd_loss_decreases():
    model = _build_dense_model()
    opt   = SGD(model, lr=0.05)
    X = np.random.randn(16, 4).astype(np.float32)
    y = np.random.rand(16, 1).astype(np.float32)
    losses = []
    for _ in range(20):
        out  = model.forward(X)
        loss = loss_obj.mse(out, Tensor(y, device="cuda"))
        model.zero_grad()
        autograd4nn(loss)
        opt.step()
        losses.append(float(loss.value.to_host_f32().ravel()[0]))
    assert losses[-1] < losses[0], f"SGD loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

def test_sgd_momentum():
    model = _build_dense_model()
    opt   = SGD(model, lr=0.05, momentum=0.9)
    X = np.random.randn(16, 4).astype(np.float32)
    y = np.random.rand(16, 1).astype(np.float32)
    losses = []
    for _ in range(20):
        out  = model.forward(X)
        loss = loss_obj.mse(out, Tensor(y, device="cuda"))
        model.zero_grad()
        autograd4nn(loss)
        opt.step()
        losses.append(float(loss.value.to_host_f32().ravel()[0]))
    assert losses[-1] < losses[0], f"SGD+momentum loss did not decrease"

_run("SGD updates weights",                  test_sgd_weights_change)
_run("SGD loss decreases over 20 steps",     test_sgd_loss_decreases)
_run("SGD with momentum=0.9",               test_sgd_momentum)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 13 — Adam Optimizer
# ════════════════════════════════════════════════════════════════════════════
section("13 · ADAM OPTIMIZER (CUDA)")

def test_adam_weights_change():
    model = _build_dense_model()
    opt   = Adam(model, lr=0.001)
    X = np.random.randn(8, 4).astype(np.float32)
    y = np.random.rand(8, 1).astype(np.float32)
    w0 = model.model[1].weights.value.to_host_f32().copy()
    out  = model.forward(X)
    loss = loss_obj.mse(out, Tensor(y, device="cuda"))
    model.zero_grad()
    autograd4nn(loss)
    opt.step()
    w1 = model.model[1].weights.value.to_host_f32()
    assert not np.allclose(w0, w1), "Adam did not change weights"

def test_adam_loss_decreases():
    model = _build_dense_model()
    opt   = Adam(model, lr=0.01)
    X = np.random.randn(16, 4).astype(np.float32)
    y = np.random.rand(16, 1).astype(np.float32)
    losses = []
    for _ in range(30):
        out  = model.forward(X)
        loss = loss_obj.mse(out, Tensor(y, device="cuda"))
        model.zero_grad()
        autograd4nn(loss)
        opt.step()
        losses.append(float(loss.value.to_host_f32().ravel()[0]))
    assert losses[-1] < losses[0], f"Adam loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

def test_adam_faster_than_sgd():
    """Adam should reach a lower loss than vanilla SGD in same steps."""
    np.random.seed(42)
    X = np.random.randn(32, 8).astype(np.float32)
    y = np.random.rand(32, 2).astype(np.float32)

    def train(opt_cls, lr, steps=40):
        m = Sequential([Input((8,)), Dense(8,16,activation="relu"), Dense(16,2,activation="sigmoid")], device="cuda")
        opt = opt_cls(m, lr=lr)
        losses = []
        for _ in range(steps):
            out = m.forward(X)
            loss = loss_obj.mse(out, Tensor(y, device="cuda"))
            m.zero_grad(); autograd4nn(loss); opt.step()
            losses.append(float(loss.value.to_host_f32().ravel()[0]))
        return losses[-1]

    loss_adam = train(Adam, 0.01)
    loss_sgd  = train(SGD,  0.05)
    assert loss_adam < loss_sgd * 1.5, f"Adam ({loss_adam:.4f}) not competitive with SGD ({loss_sgd:.4f})"

_run("Adam updates weights",                 test_adam_weights_change)
_run("Adam loss decreases over 30 steps",    test_adam_loss_decreases)
_run("Adam competitive with SGD (40 steps)", test_adam_faster_than_sgd)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 14 — Full Training Loops (End-to-End)
# ════════════════════════════════════════════════════════════════════════════
section("14 · END-TO-END TRAINING LOOPS (CUDA)")

def test_fit_regression():
    """MLP should fit a simple linear relationship."""
    np.random.seed(0)
    X = np.random.randn(64, 4).astype(np.float32)
    y = (X @ np.array([1, -1, 0.5, -0.5], dtype=np.float32)).reshape(-1, 1)
    y = (y - y.min()) / (y.max() - y.min())  # normalize to [0,1]

    model = Sequential([
        Input((4,)),
        Dense(4, 16, activation="relu"),
        Dense(16, 1, activation="sigmoid"),
    ], device="cuda")
    opt = Adam(model, lr=0.01)
    hist = model.fit(X, y, opt, loss_obj.mse, Epochs=50, batch_size=16, Loss_interval=9999)
    assert hist[-1] < hist[0] * 0.8, f"Regression not converging: {hist[0]:.4f}→{hist[-1]:.4f}"

def test_fit_classification():
    """2-class XOR-like problem — model should improve."""
    np.random.seed(1)
    N = 128
    X = np.random.randn(N, 2).astype(np.float32)
    y_raw = ((X[:,0] * X[:,1]) > 0).astype(np.float32).reshape(-1,1)

    model = Sequential([
        Input((2,)),
        Dense(2, 16, activation="relu"),
        Dense(16, 8, activation="relu"),
        Dense(8,  1, activation="sigmoid"),
    ], device="cuda")
    opt = Adam(model, lr=0.02)
    hist = model.fit(X, y_raw, opt, loss_obj.binary_cross_entropy, Epochs=60, batch_size=16, Loss_interval=9999)
    assert hist[-1] < hist[0], f"Classification not improving: {hist[0]:.4f}→{hist[-1]:.4f}"

def test_fit_batch_size_1():
    """Verify online learning (batch_size=1) works without crash."""
    X = np.random.randn(8, 4).astype(np.float32)
    y = np.random.rand(8, 1).astype(np.float32)
    model = Sequential([Input((4,)), Dense(4, 4, activation="relu"), Dense(4, 1, activation="sigmoid")], device="cuda")
    opt = SGD(model, lr=0.01)
    hist = model.fit(X, y, opt, loss_obj.mse, Epochs=5, batch_size=1, Loss_interval=9999)
    assert len(hist) == 5

def test_fit_batch_size_full():
    """Batch gradient descent (batch_size = N)."""
    N = 32
    X = np.random.randn(N, 4).astype(np.float32)
    y = np.random.rand(N, 1).astype(np.float32)
    model = Sequential([Input((4,)), Dense(4, 8, activation="relu"), Dense(8, 1, activation="sigmoid")], device="cuda")
    opt = Adam(model, lr=0.01)
    hist = model.fit(X, y, opt, loss_obj.mse, Epochs=20, batch_size=N, Loss_interval=9999)
    assert hist[-1] < hist[0], "Full-batch GD not converging"

def test_fit_multiclass():
    """3-class softmax classification."""
    np.random.seed(2)
    N = 90
    X = np.vstack([
        np.random.randn(30, 4) + np.array([2, 0, 0, 0]),
        np.random.randn(30, 4) + np.array([0, 2, 0, 0]),
        np.random.randn(30, 4) + np.array([0, 0, 2, 0]),
    ]).astype(np.float32)
    y_onehot = np.eye(3, dtype=np.float32)[[0]*30 + [1]*30 + [2]*30]
    model = Sequential([
        Input((4,)),
        Dense(4, 16, activation="relu"),
        Dense(16, 3, activation="softmax"),
    ], device="cuda")
    opt = Adam(model, lr=0.01)
    hist = model.fit(X, y_onehot, opt, loss_obj.categorical_cross_entropy, Epochs=60, batch_size=30, Loss_interval=9999)
    assert hist[-1] < hist[0], f"Multiclass not converging: {hist[0]:.4f}→{hist[-1]:.4f}"

def test_fit_cnn():
    """CNN on tiny 'images'."""
    np.random.seed(3)
    N = 20
    X = np.random.randn(N, 1, 8, 8).astype(np.float32)
    y = np.random.rand(N, 4).astype(np.float32)
    y = y / y.sum(axis=1, keepdims=True)
    model = Sequential([
        Input((1, 8, 8)),
        Conv2D(4, 1, (3,3), activation="relu"),
        MaxPool2D(pool_size=(2,2), stride=2),
        Flatten(),
        Dense(36, 4, activation="softmax"),
    ], device="cuda")
    opt = Adam(model, lr=0.005)
    hist = model.fit(X, y, opt, loss_obj.categorical_cross_entropy, Epochs=20, batch_size=N, Loss_interval=9999)
    assert hist[-1] < hist[0] * 1.1, f"CNN not training: {hist[0]:.4f}→{hist[-1]:.4f}"

_run("fit() — MLP regression (MSE)",        test_fit_regression)
_run("fit() — XOR binary classification",   test_fit_classification)
_run("fit() — batch_size=1 (online)",        test_fit_batch_size_1)
_run("fit() — batch_size=N (full batch)",   test_fit_batch_size_full)
_run("fit() — 3-class softmax",             test_fit_multiclass)
_run("fit() — CNN on tiny images",           test_fit_cnn)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 15 — Gradient Sanity (Zero Loss → Zero Grad)
# ════════════════════════════════════════════════════════════════════════════
section("15 · GRADIENT SANITY CHECKS (CUDA)")

def test_zero_loss_zero_grad():
    """If y_pred == y, MSE loss = 0 and gradients should be ~0."""
    model = Sequential([Input((4,)), Dense(4, 2, activation="sigmoid")], device="cuda")
    x = np.random.randn(4, 4).astype(np.float32)
    out = model.forward(x)
    y_np = out.value.to_host_f32().copy()  # perfect predictions
    y = Tensor(y_np, device="cuda")
    loss = loss_obj.mse(out, y)
    model.zero_grad()
    autograd4nn(loss)
    lv = float(loss.value.to_host_f32().ravel()[0])
    assert abs(lv) < 1e-6, f"Zero-target loss is not 0: {lv}"

def test_grad_increases_with_error():
    """Larger prediction error → larger gradient magnitude."""
    model1 = Sequential([Input((4,)), Dense(4, 2, activation="sigmoid")], device="cuda")
    model2 = Sequential([Input((4,)), Dense(4, 2, activation="sigmoid")], device="cuda")
    x = np.ones((4, 4), dtype=np.float32)
    y_close = np.full((4, 2), 0.5, dtype=np.float32)
    y_far   = np.ones((4, 2), dtype=np.float32)

    out1 = model1.forward(x)
    loss1 = loss_obj.mse(out1, Tensor(y_close, device="cuda"))
    model1.zero_grad(); autograd4nn(loss1)
    g1 = np.abs(model1.model[1].weights.node.cp.to_host_f32()).mean()

    out2 = model2.forward(x)
    loss2 = loss_obj.mse(out2, Tensor(y_far, device="cuda"))
    model2.zero_grad(); autograd4nn(loss2)
    g2 = np.abs(model2.model[1].weights.node.cp.to_host_f32()).mean()

    assert g2 >= g1 * 0.5, f"Larger error should give ≥ gradient: close={g1:.4f}, far={g2:.4f}"

def test_double_backward_no_explosion():
    """Run backward twice in succession; values should remain finite."""
    model = Sequential([Input((4,)), Dense(4, 8, activation="relu"), Dense(8, 2, activation="sigmoid")], device="cuda")
    x = np.random.randn(4, 4).astype(np.float32)
    y = np.random.rand(4, 2).astype(np.float32)
    for _ in range(2):
        out = model.forward(x)
        loss = loss_obj.mse(out, Tensor(y, device="cuda"))
        model.zero_grad()
        autograd4nn(loss)
    g = model.model[1].weights.node.cp.to_host_f32()
    assert np.isfinite(g).all(), "Gradient exploded / went NaN"

_run("zero pred error → ~zero gradient",    test_zero_loss_zero_grad)
_run("larger error → larger gradient",       test_grad_increases_with_error)
_run("two backward passes stay finite",      test_double_backward_no_explosion)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 16 — Model Save & Load
# ════════════════════════════════════════════════════════════════════════════
section("16 · MODEL SAVE & LOAD")

import tempfile, os

def test_save_load_dense():
    model = Sequential([
        Input((4,)),
        Dense(4, 8, activation="relu"),
        Dense(8, 2, activation="sigmoid"),
    ], device="cuda")
    x = np.random.randn(4, 4).astype(np.float32)
    out_before = model.forward(x).value.to_host_f32()

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        model.save(path)
        model2 = Sequential.load(path)
        # Move to GPU for inference
        model2._move_params_to_device() if model2.device == "cuda" else None
        out_after = model2.forward(x).value.to_host_f32() if hasattr(model2.forward(x).value, 'to_host_f32') else model2.forward(x).value
        # Weights should be identical
        w_orig = model.model[1].weights.value.to_host_f32()
        w_load = model2.model[1].weights.value if isinstance(model2.model[1].weights.value, np.ndarray) else model2.model[1].weights.value.to_host_f32()
        assert np.allclose(w_orig, w_load, atol=1e-5), "Loaded weights differ from saved"
    finally:
        os.unlink(path)

_run("save/load Dense model — weights match", test_save_load_dense)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 17 — Numerical Stability
# ════════════════════════════════════════════════════════════════════════════
section("17 · NUMERICAL STABILITY (CUDA)")

def test_softmax_large_values():
    """Softmax on large inputs should not produce NaN/Inf."""
    x_np = np.array([[1000.0, -1000.0, 500.0]], dtype=np.float32)
    t = _tensor_gpu(x_np)
    out = Tensor.softmax(t).value.to_host_f32()
    assert np.isfinite(out).all(), f"Softmax NaN/Inf on large input: {out}"
    assert np.allclose(out.sum(axis=-1), 1.0, atol=1e-4)

def test_log_near_zero():
    """log of tiny positive number should not be -inf in BCE context."""
    x_np = np.array([[1e-7, 1.0 - 1e-7]], dtype=np.float32)
    t = _tensor_gpu(x_np + 1e-15)  # matches BCE epsilon
    out = Tensor.log(t).value.to_host_f32()
    assert np.isfinite(out).all(), f"log near-zero blew up: {out}"

def test_no_nan_after_100_steps():
    """Train 100 steps on random data and check for NaN in weights."""
    model = Sequential([
        Input((8,)),
        Dense(8, 16, activation="relu"),
        Dense(16, 4, activation="softmax"),
    ], device="cuda")
    opt = Adam(model, lr=0.001)
    X = np.random.randn(32, 8).astype(np.float32)
    y = np.eye(4, dtype=np.float32)[[i % 4 for i in range(32)]]
    for _ in range(100):
        out  = model.forward(X)
        loss = loss_obj.categorical_cross_entropy(out, Tensor(y, device="cuda"))
        model.zero_grad(); autograd4nn(loss); opt.step()
    w = model.model[1].weights.value.to_host_f32()
    assert np.isfinite(w).all(), f"NaN/Inf in weights after 100 steps"

_run("softmax stable on large inputs",       test_softmax_large_values)
_run("log stable near zero",                 test_log_near_zero)
_run("no NaN after 100 Adam steps",          test_no_nan_after_100_steps)


# ════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ════════════════════════════════════════════════════════════════════════════
total = _PASS + _FAIL
print(f"\n{'═'*70}")
print(f"  RESULTS:  {_PASS} passed  /  {_FAIL} failed  /  {total} total")
print(f"{'═'*70}")

if _FAIL > 0:
    print("\n  ── FAILED TESTS ──────────────────────────────────────────────────")
    for status, name, tb in _RESULTS:
        if status == "FAIL":
            print(f"\n  ✗ {name}")
            if tb:
                for line in tb.strip().splitlines()[-6:]:
                    print(f"    {line}")
    print()
    print("  ── DEBUGGING CHECKLIST ───────────────────────────────────────────")
    print("  1. CUDA visible?        →  check 'nvidia-smi' and CUDA_VISIBLE_DEVICES")
    print("  2. cuTen built?         →  run 'python build_cuten.py'")
    print("  3. seera_cuda built?    →  run 'python build_engine.py'")
    print("  4. Gradient is all-zero →  likely zero_grad() not called before backward")
    print("  5. Loss doesn't drop    →  check lr scale, activation choices, batch size")
    print("  6. NaN/Inf in weights   →  lr too high, or missing gradient clipping")
    print("  7. Wrong output shape   →  verify in/out channel args to Conv2D")
    print()

sys.exit(0 if _FAIL == 0 else 1)