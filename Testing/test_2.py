"""
╔══════════════════════════════════════════════════════════════════╗
║         SEERA vs PYTORCH — RUTHLESS GRADIENT TEST SUITE         ║
║         All tests run on CUDA. Grads compared to PyTorch.       ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python test_seera_cuda.py                # run all tests
    python test_seera_cuda.py -v             # verbose
    python test_seera_cuda.py TestGrads.test_relu   # single test

Requirements:
    pip install torch pytest
    Seera (Seera_init.py, Seera_Engine.py, Seera.py) on PYTHONPATH
    CUDA GPU with seera_cuda / cuTen compiled

What we test:
  - Forward values  : Seera output == PyTorch output  (atol=1e-4)
  - Backward grads  : Seera grad   == PyTorch grad    (atol=1e-4)
  - Gradient of gradient is *non-zero* (model is actually learning)
  - Training step   : loss decreases after one SGD / Adam step
  - Numerically checks with finite-difference (gradcheck) where feasible
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sys
import math
import unittest
import numpy as np

# ── Guard: PyTorch is needed for reference ──
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("PyTorch is required to run this test suite.  pip install torch")

# ── Guard: CUDA must be available in PyTorch ──
if not torch.cuda.is_available():
    sys.exit("A CUDA GPU is required to run this test suite.")

DEVICE = torch.device("cuda")

# ── Import Seera ──
try:
    from Seera_init import tensor as Tensor
    from Seera_Engine import autograd4nn
    from Seera import (
        Sequential, Dense, Conv2D, ConvTranspose2D,
        Flatten, MaxPool2D, BatchNorm1d, BatchNorm2d,
        Input, Loss, SGD, Adam,
    )
except ImportError as e:
    sys.exit(f"Could not import Seera: {e}\n"
             "Make sure Seera_init.py / Seera_Engine.py / Seera.py are on PYTHONPATH.")

# ── Check cuTen / seera_cuda ──
try:
    from cuTen import cuten
    import seera_cuda
except ImportError as e:
    sys.exit(f"CUDA backend not found: {e}\n"
             "Compile seera_cuda and cuTen before running GPU tests.")


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

ATOL_FWD  = 1e-4   # forward value tolerance
ATOL_GRAD = 1e-4   # gradient tolerance
RTOL      = 1e-3   # relative tolerance


def seera_to_np(val):
    """Bring a Seera tensor value (cuten or ndarray) to float32 numpy."""
    if isinstance(val, cuten):
        return val.to_host_f32()
    return np.array(val, dtype=np.float32)


def seera_grad_to_np(t: Tensor) -> np.ndarray:
    """Extract accumulated gradient from a Seera Tensor."""
    return seera_to_np(t.node.cp)


def torch_tensor(arr, requires_grad=False):
    """Build a float32 CUDA torch.Tensor from numpy."""
    return torch.tensor(arr, dtype=torch.float32,
                        device=DEVICE, requires_grad=requires_grad)


def assert_close(name, seera_val, torch_val,
                 atol=ATOL_FWD, rtol=RTOL):
    """Compare numpy arrays, print a helpful diff on failure."""
    s = np.asarray(seera_val, dtype=np.float32)
    t = torch_val.detach().cpu().numpy().astype(np.float32)
    if s.shape != t.shape:
        raise AssertionError(
            f"[{name}] shape mismatch: Seera {s.shape} vs PyTorch {t.shape}")
    max_diff = np.abs(s - t).max()
    rel_diff = max_diff / (np.abs(t).max() + 1e-8)
    if not np.allclose(s, t, atol=atol, rtol=rtol):
        raise AssertionError(
            f"[{name}] MISMATCH  max_abs={max_diff:.6f}  max_rel={rel_diff:.6f}\n"
            f"  Seera  : {s.ravel()[:8]}\n"
            f"  PyTorch: {t.ravel()[:8]}")


def run_seera_backward(seera_loss_tensor):
    """Run Seera backward and return the autograd object."""
    return autograd4nn(seera_loss_tensor)


# ══════════════════════════════════════════════════════════════════
#  1. ELEMENTWISE OPERATIONS & GRADIENTS
# ══════════════════════════════════════════════════════════════════

class TestElementwiseOps(unittest.TestCase):
    """Forward + gradient for every scalar/elementwise op."""

    def _run(self, name, seera_fn, torch_fn, shape=(4, 8)):
        rng = np.random.RandomState(42)
        # Use positive values for ops that need it (log, sqrt)
        raw = np.abs(rng.randn(*shape).astype(np.float32)) + 0.5

        # Seera (GPU)
        s_t = Tensor(raw, is_leaf=True, device="cuda")
        s_out = seera_fn(s_t)
        run_seera_backward(s_out.sum())  # scalar loss
        s_fwd = seera_to_np(s_out.value)
        s_grad = seera_grad_to_np(s_t)

        # PyTorch (GPU)
        p_t = torch_tensor(raw, requires_grad=True)
        p_out = torch_fn(p_t)
        p_scalar = p_out.sum()
        p_scalar.backward()
        p_fwd = p_out.detach().cpu().numpy()
        p_grad = p_t.grad.cpu().numpy()

        assert_close(f"{name}_fwd", s_fwd, torch.tensor(p_fwd))
        assert_close(f"{name}_grad", s_grad, torch.tensor(p_grad))

    def test_relu(self):
        rng = np.random.RandomState(0)
        raw = rng.randn(4, 8).astype(np.float32)
        s_t = Tensor(raw, is_leaf=True, device="cuda")
        s_out = s_t.relu()
        run_seera_backward(s_out.sum())
        p_t = torch_tensor(raw, requires_grad=True)
        (F.relu(p_t).sum()).backward()
        assert_close("relu_fwd", seera_to_np(s_out.value),
                     F.relu(torch_tensor(raw)))
        assert_close("relu_grad", seera_grad_to_np(s_t), p_t.grad)

    def test_sigmoid(self):
        self._run("sigmoid",
                  lambda t: t.sigmoid(),
                  lambda t: torch.sigmoid(t))

    def test_tanh(self):
        self._run("tanh",
                  lambda t: t.tanh(),
                  lambda t: torch.tanh(t))

    def test_exp(self):
        rng = np.random.RandomState(1)
        raw = rng.randn(4, 4).astype(np.float32) * 0.5
        self._run("exp",
                  lambda t: t.exp(),
                  lambda t: torch.exp(t),
                  shape=(4, 4))

    def test_log(self):
        self._run("log",
                  lambda t: t.log(),
                  lambda t: torch.log(t))

    def test_sqrt(self):
        self._run("sqrt",
                  lambda t: t.sqrt(),
                  lambda t: torch.sqrt(t))

    def test_abs(self):
        rng = np.random.RandomState(3)
        raw = rng.randn(4, 8).astype(np.float32)
        s_t = Tensor(raw, is_leaf=True, device="cuda")
        s_out = s_t.abs()
        run_seera_backward(s_out.sum())
        p_t = torch_tensor(raw, requires_grad=True)
        torch.abs(p_t).sum().backward()
        assert_close("abs_fwd", seera_to_np(s_out.value),
                     torch.abs(torch_tensor(raw)))
        assert_close("abs_grad", seera_grad_to_np(s_t), p_t.grad)

    def test_pow2(self):
        rng = np.random.RandomState(4)
        raw = rng.randn(4, 8).astype(np.float32)
        s_t = Tensor(raw, is_leaf=True, device="cuda")
        s_out = s_t ** 2
        run_seera_backward(s_out.sum())
        p_t = torch_tensor(raw, requires_grad=True)
        (p_t ** 2).sum().backward()
        assert_close("pow2_fwd", seera_to_np(s_out.value),
                     (torch_tensor(raw) ** 2))
        assert_close("pow2_grad", seera_grad_to_np(s_t), p_t.grad)

    def test_pow3(self):
        rng = np.random.RandomState(5)
        raw = rng.randn(4, 8).astype(np.float32) * 0.5
        s_t = Tensor(raw, is_leaf=True, device="cuda")
        s_out = s_t ** 3
        run_seera_backward(s_out.sum())
        p_t = torch_tensor(raw, requires_grad=True)
        (p_t ** 3).sum().backward()
        assert_close("pow3_fwd", seera_to_np(s_out.value),
                     (torch_tensor(raw) ** 3))
        assert_close("pow3_grad", seera_grad_to_np(s_t), p_t.grad)

    def test_add(self):
        rng = np.random.RandomState(6)
        a = rng.randn(4, 8).astype(np.float32)
        b = rng.randn(4, 8).astype(np.float32)
        sa = Tensor(a, is_leaf=True, device="cuda")
        sb = Tensor(b, is_leaf=True, device="cuda")
        run_seera_backward((sa + sb).sum())
        pa = torch_tensor(a, requires_grad=True)
        pb = torch_tensor(b, requires_grad=True)
        (pa + pb).sum().backward()
        assert_close("add_grad_a", seera_grad_to_np(sa), pa.grad)
        assert_close("add_grad_b", seera_grad_to_np(sb), pb.grad)

    def test_mul(self):
        rng = np.random.RandomState(7)
        a = rng.randn(4, 8).astype(np.float32)
        b = rng.randn(4, 8).astype(np.float32)
        sa = Tensor(a, is_leaf=True, device="cuda")
        sb = Tensor(b, is_leaf=True, device="cuda")
        run_seera_backward((sa * sb).sum())
        pa = torch_tensor(a, requires_grad=True)
        pb = torch_tensor(b, requires_grad=True)
        (pa * pb).sum().backward()
        assert_close("mul_grad_a", seera_grad_to_np(sa), pa.grad)
        assert_close("mul_grad_b", seera_grad_to_np(sb), pb.grad)

    def test_sub(self):
        rng = np.random.RandomState(8)
        a = rng.randn(4, 8).astype(np.float32)
        b = rng.randn(4, 8).astype(np.float32)
        sa = Tensor(a, is_leaf=True, device="cuda")
        sb = Tensor(b, is_leaf=True, device="cuda")
        run_seera_backward((sa - sb).sum())
        pa = torch_tensor(a, requires_grad=True)
        pb = torch_tensor(b, requires_grad=True)
        (pa - pb).sum().backward()
        assert_close("sub_grad_a", seera_grad_to_np(sa), pa.grad)
        assert_close("sub_grad_b", seera_grad_to_np(sb), pb.grad)

    def test_div(self):
        rng = np.random.RandomState(9)
        a = rng.randn(4, 8).astype(np.float32)
        b = np.abs(rng.randn(4, 8).astype(np.float32)) + 0.5
        sa = Tensor(a, is_leaf=True, device="cuda")
        sb = Tensor(b, is_leaf=True, device="cuda")
        run_seera_backward((sa / sb).sum())
        pa = torch_tensor(a, requires_grad=True)
        pb = torch_tensor(b, requires_grad=True)
        (pa / pb).sum().backward()
        assert_close("div_grad_a", seera_grad_to_np(sa), pa.grad)
        assert_close("div_grad_b", seera_grad_to_np(sb), pb.grad)

    def test_clip(self):
        rng = np.random.RandomState(10)
        raw = rng.randn(4, 8).astype(np.float32)
        s_t = Tensor(raw, is_leaf=True, device="cuda")
        s_out = s_t.clip(-0.5, 0.5)
        run_seera_backward(s_out.sum())
        p_t = torch_tensor(raw, requires_grad=True)
        torch.clamp(p_t, -0.5, 0.5).sum().backward()
        assert_close("clip_fwd", seera_to_np(s_out.value),
                     torch.clamp(torch_tensor(raw), -0.5, 0.5))
        assert_close("clip_grad", seera_grad_to_np(s_t), p_t.grad)

    def test_neg(self):
        rng = np.random.RandomState(11)
        raw = rng.randn(4, 8).astype(np.float32)
        s_t = Tensor(raw, is_leaf=True, device="cuda")
        run_seera_backward((-s_t).sum())
        p_t = torch_tensor(raw, requires_grad=True)
        (-p_t).sum().backward()
        assert_close("neg_grad", seera_grad_to_np(s_t), p_t.grad)


# ══════════════════════════════════════════════════════════════════
#  2. REDUCTIONS
# ══════════════════════════════════════════════════════════════════

class TestReductions(unittest.TestCase):
    """sum / mean gradient flows."""

    def test_sum_all(self):
        raw = np.random.randn(3, 4, 5).astype(np.float32)
        s_t = Tensor(raw, is_leaf=True, device="cuda")
        run_seera_backward(s_t.sum())
        p_t = torch_tensor(raw, requires_grad=True)
        p_t.sum().backward()
        assert_close("sum_all_grad", seera_grad_to_np(s_t), p_t.grad)

    def test_sum_axis0(self):
        raw = np.random.randn(4, 6).astype(np.float32)
        s_t = Tensor(raw, is_leaf=True, device="cuda")
        run_seera_backward(s_t.sum(axis=0))
        p_t = torch_tensor(raw, requires_grad=True)
        p_t.sum(dim=0).sum().backward()
        # Seera sums further to scalar automatically, so compare grad shape
        sg = seera_grad_to_np(s_t)
        pg = p_t.grad.cpu().numpy()
        assert sg.shape == pg.shape, f"sum_ax0 shape {sg.shape} vs {pg.shape}"
        assert_close("sum_ax0_grad", sg, torch.tensor(pg))

    def test_sum_axis1(self):
        raw = np.random.randn(4, 6).astype(np.float32)
        s_t = Tensor(raw, is_leaf=True, device="cuda")
        run_seera_backward(s_t.sum(axis=1))
        p_t = torch_tensor(raw, requires_grad=True)
        p_t.sum(dim=1).sum().backward()
        assert_close("sum_ax1_grad", seera_grad_to_np(s_t), p_t.grad)

    def test_mean_all(self):
        raw = np.random.randn(3, 5).astype(np.float32)
        s_t = Tensor(raw, is_leaf=True, device="cuda")
        run_seera_backward(s_t.mean())
        p_t = torch_tensor(raw, requires_grad=True)
        p_t.mean().backward()
        assert_close("mean_all_grad", seera_grad_to_np(s_t), p_t.grad)

    def test_mean_axis0(self):
        raw = np.random.randn(5, 4).astype(np.float32)
        s_t = Tensor(raw, is_leaf=True, device="cuda")
        run_seera_backward(s_t.mean(axis=0))
        p_t = torch_tensor(raw, requires_grad=True)
        p_t.mean(dim=0).sum().backward()
        assert_close("mean_ax0_grad", seera_grad_to_np(s_t), p_t.grad)


# ══════════════════════════════════════════════════════════════════
#  3. MATMUL
# ══════════════════════════════════════════════════════════════════

class TestMatmul(unittest.TestCase):

    def _matmul_case(self, name, M, K, N):
        rng = np.random.RandomState(42)
        a = rng.randn(M, K).astype(np.float32)
        b = rng.randn(K, N).astype(np.float32)

        sa = Tensor(a, is_leaf=True, device="cuda")
        sb = Tensor(b, is_leaf=True, device="cuda")
        s_out = sa.matmul(sb)
        run_seera_backward(s_out.sum())

        pa = torch_tensor(a, requires_grad=True)
        pb = torch_tensor(b, requires_grad=True)
        (pa @ pb).sum().backward()

        assert_close(f"{name}_fwd", seera_to_np(s_out.value), pa @ pb)
        assert_close(f"{name}_grad_a", seera_grad_to_np(sa), pa.grad)
        assert_close(f"{name}_grad_b", seera_grad_to_np(sb), pb.grad)

    def test_matmul_square(self):      self._matmul_case("sq",   8, 8, 8)
    def test_matmul_tall(self):        self._matmul_case("tall", 16, 4, 8)
    def test_matmul_wide(self):        self._matmul_case("wide", 4, 32, 16)
    def test_matmul_batch_like(self):  self._matmul_case("batch",32, 64, 10)


# ══════════════════════════════════════════════════════════════════
#  4. SOFTMAX
# ══════════════════════════════════════════════════════════════════

class TestSoftmax(unittest.TestCase):

    def _softmax_case(self, name, shape):
        rng = np.random.RandomState(99)
        raw = rng.randn(*shape).astype(np.float32)
        # upstream gradient (random, to test VJP properly)
        upstream = rng.randn(*shape).astype(np.float32)

        s_t = Tensor(raw, is_leaf=True, device="cuda")
        s_out = s_t.softmax()
        # simulate arbitrary upstream by multiplying and summing
        s_upstream = Tensor(upstream, device="cuda")
        run_seera_backward((s_out * s_upstream).sum())

        p_t = torch_tensor(raw, requires_grad=True)
        p_out = F.softmax(p_t, dim=-1)
        (p_out * torch_tensor(upstream)).sum().backward()

        assert_close(f"{name}_fwd", seera_to_np(s_out.value), p_out)
        assert_close(f"{name}_grad", seera_grad_to_np(s_t), p_t.grad)

    def test_softmax_2d(self):   self._softmax_case("sm_2d",   (8, 10))
    def test_softmax_batch(self): self._softmax_case("sm_bat",  (32, 10))
    def test_softmax_large(self): self._softmax_case("sm_large",(16, 100))

    def test_softmax_sums_to_one(self):
        raw = np.random.randn(16, 10).astype(np.float32)
        s_t = Tensor(raw, is_leaf=True, device="cuda")
        s_out = seera_to_np(s_t.softmax().value)
        row_sums = s_out.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(16), atol=1e-5,
                                   err_msg="Softmax rows don't sum to 1")


# ══════════════════════════════════════════════════════════════════
#  5. CONV2D FORWARD + BACKWARD
# ══════════════════════════════════════════════════════════════════

class TestConv2D(unittest.TestCase):

    def _conv_case(self, name, N, C, H, W, F_out, KH, KW, stride, pad):
        rng = np.random.RandomState(42)
        x_np  = rng.randn(N, C, H, W).astype(np.float32) * 0.1
        w_np  = rng.randn(F_out, C, KH, KW).astype(np.float32) * 0.1

        # Seera
        sx = Tensor(x_np, is_leaf=True, device="cuda")
        sw = Tensor(w_np, is_leaf=True, device="cuda")
        s_out = sx.conv2d(sw, stride=(stride, stride), padding=(pad, pad))
        run_seera_backward(s_out.sum())

        # PyTorch
        px = torch_tensor(x_np, requires_grad=True)
        pw = torch_tensor(w_np, requires_grad=True)
        p_out = F.conv2d(px, pw, stride=stride, padding=pad)
        p_out.sum().backward()

        assert_close(f"{name}_fwd", seera_to_np(s_out.value), p_out)
        assert_close(f"{name}_grad_x", seera_grad_to_np(sx), px.grad, atol=2e-4)
        assert_close(f"{name}_grad_w", seera_grad_to_np(sw), pw.grad, atol=2e-4)

    def test_conv_basic(self):
        self._conv_case("conv_basic", 2, 3, 8, 8, 4, 3, 3, 1, 0)

    def test_conv_stride2(self):
        self._conv_case("conv_s2", 2, 3, 8, 8, 4, 3, 3, 2, 0)

    def test_conv_padded(self):
        self._conv_case("conv_pad", 2, 3, 8, 8, 4, 3, 3, 1, 1)

    def test_conv_stride_pad(self):
        self._conv_case("conv_sp", 4, 1, 16, 16, 8, 5, 5, 2, 2)

    def test_conv_1x1(self):
        self._conv_case("conv_1x1", 4, 16, 8, 8, 32, 1, 1, 1, 0)

    def test_conv_depthwise_like(self):
        self._conv_case("conv_deep", 2, 8, 12, 12, 8, 3, 3, 1, 1)

    def test_conv_rectangular_kernel(self):
        # Non-square kernel: 1×3
        rng = np.random.RandomState(77)
        x_np = rng.randn(2, 4, 8, 8).astype(np.float32) * 0.1
        w_np = rng.randn(8, 4, 1, 3).astype(np.float32) * 0.1
        sx = Tensor(x_np, is_leaf=True, device="cuda")
        sw = Tensor(w_np, is_leaf=True, device="cuda")
        s_out = sx.conv2d(sw, stride=(1, 1), padding=(0, 1))
        run_seera_backward(s_out.sum())
        px = torch_tensor(x_np, requires_grad=True)
        pw = torch_tensor(w_np, requires_grad=True)
        F.conv2d(px, pw, stride=1, padding=(0, 1)).sum().backward()
        assert_close("conv_rect_fwd", seera_to_np(s_out.value),
                     F.conv2d(torch_tensor(x_np), torch_tensor(w_np),
                              stride=1, padding=(0, 1)))
        assert_close("conv_rect_grad_x", seera_grad_to_np(sx), px.grad, atol=2e-4)
        assert_close("conv_rect_grad_w", seera_grad_to_np(sw), pw.grad, atol=2e-4)


# ══════════════════════════════════════════════════════════════════
#  6. CONV TRANSPOSE 2D
# ══════════════════════════════════════════════════════════════════

class TestConvTranspose2D(unittest.TestCase):

    def _convT_case(self, name, N, Cin, H, W, Cout, KH, KW, stride, pad):
        rng = np.random.RandomState(42)
        x_np = rng.randn(N, Cin, H, W).astype(np.float32) * 0.1
        # Seera ConvTranspose weight: (Cin, Cout, KH, KW)
        w_np_seera = rng.randn(Cin, Cout, KH, KW).astype(np.float32) * 0.1
        # PyTorch weight: (Cin, Cout, KH, KW) — same convention
        w_np_torch = w_np_seera.copy()

        sx = Tensor(x_np, is_leaf=True, device="cuda")
        sw = Tensor(w_np_seera, is_leaf=True, device="cuda")
        s_out = sx.conv_transpose2d(sw, stride=(stride, stride), padding=(pad, pad))
        run_seera_backward(s_out.sum())

        px = torch_tensor(x_np, requires_grad=True)
        pw = torch_tensor(w_np_torch, requires_grad=True)
        p_out = F.conv_transpose2d(px, pw, stride=stride, padding=pad)
        p_out.sum().backward()

        assert_close(f"{name}_fwd",    seera_to_np(s_out.value), p_out, atol=2e-4)
        assert_close(f"{name}_grad_x", seera_grad_to_np(sx), px.grad, atol=2e-4)
        assert_close(f"{name}_grad_w", seera_grad_to_np(sw), pw.grad, atol=2e-4)

    def test_convT_basic(self):   self._convT_case("cT_basic", 2, 4, 4, 4, 8, 3, 3, 1, 0)
    def test_convT_stride2(self): self._convT_case("cT_s2",    2, 4, 4, 4, 8, 4, 4, 2, 0)
    def test_convT_padded(self):  self._convT_case("cT_pad",   2, 4, 4, 4, 8, 3, 3, 1, 1)


# ══════════════════════════════════════════════════════════════════
#  7. MAXPOOL2D
# ══════════════════════════════════════════════════════════════════

class TestMaxPool2D(unittest.TestCase):

    def _pool_case(self, name, N, C, H, W, KH, KW, stride, pad):
        rng = np.random.RandomState(42)
        x_np = rng.randn(N, C, H, W).astype(np.float32)

        sx = Tensor(x_np, is_leaf=True, device="cuda")
        s_out = sx.maxpool2d(kernelsize=(KH, KW),
                             stride=(stride, stride),
                             padding=(pad, pad))
        run_seera_backward(s_out.sum())

        px = torch_tensor(x_np, requires_grad=True)
        p_out = F.max_pool2d(px, kernel_size=(KH, KW),
                             stride=stride, padding=pad)
        p_out.sum().backward()

        assert_close(f"{name}_fwd",  seera_to_np(s_out.value), p_out)
        assert_close(f"{name}_grad", seera_grad_to_np(sx), px.grad, atol=2e-4)

    def test_pool_2x2_s1(self):  self._pool_case("pool_2x2_s1", 2, 4, 8, 8, 2, 2, 2, 0)
    def test_pool_3x3(self):     self._pool_case("pool_3x3",    2, 4, 9, 9, 3, 3, 3, 0)
    def test_pool_padded(self):  self._pool_case("pool_pad",    2, 4, 8, 8, 2, 2, 2, 1)


# ══════════════════════════════════════════════════════════════════
#  8. BATCHNORM  (CPU backward — still check GPU forward value)
# ══════════════════════════════════════════════════════════════════

class TestBatchNorm(unittest.TestCase):
    """BatchNorm uses CPU backward internally; we test CPU path here."""

    def test_batchnorm1d_fwd(self):
        rng = np.random.RandomState(42)
        x  = rng.randn(16, 32).astype(np.float32)
        g  = np.ones(32,  dtype=np.float32)
        b  = np.zeros(32, dtype=np.float32)
        rm = np.zeros(32, dtype=np.float32)
        rv = np.ones(32,  dtype=np.float32)

        sx = Tensor(x, is_leaf=True)
        sg = Tensor(g, is_leaf=True)
        sb = Tensor(b, is_leaf=True)
        s_out = sx.batchnorm(sg, sb, rm, rv,
                             training=True, momentum=0.1, eps=1e-5, mode="1d")

        px = torch_tensor(x)
        p_bn = nn.BatchNorm1d(32, momentum=0.1, eps=1e-5).to(DEVICE)
        with torch.no_grad():
            p_bn.weight.fill_(1.0)
            p_bn.bias.fill_(0.0)
        p_out = p_bn(px)
        assert_close("bn1d_fwd", seera_to_np(s_out.value), p_out)

    def test_batchnorm1d_grad(self):
        rng = np.random.RandomState(7)
        x  = rng.randn(8, 16).astype(np.float32)
        g  = rng.randn(16).astype(np.float32) * 0.5 + 1.0
        b  = rng.randn(16).astype(np.float32) * 0.1
        rm = np.zeros(16, dtype=np.float32)
        rv = np.ones(16, dtype=np.float32)

        sx = Tensor(x,   is_leaf=True)
        sg = Tensor(g,   is_leaf=True)
        sb = Tensor(b,   is_leaf=True)
        s_out = sx.batchnorm(sg, sb, rm.copy(), rv.copy(),
                             training=True, momentum=0.1, eps=1e-5, mode="1d")
        run_seera_backward(s_out.sum())

        px = torch_tensor(x, requires_grad=True)
        p_bn = nn.BatchNorm1d(16, momentum=0.1, eps=1e-5)
        with torch.no_grad():
            p_bn.weight.copy_(torch.tensor(g))
            p_bn.bias.copy_(torch.tensor(b))
        p_out = p_bn(px)
        p_out.sum().backward()

        assert_close("bn1d_grad_x", seera_grad_to_np(sx), px.grad, atol=1e-3)

    def test_batchnorm2d_fwd(self):
        rng = np.random.RandomState(13)
        x  = rng.randn(4, 8, 6, 6).astype(np.float32)
        g  = np.ones(8,  dtype=np.float32)
        b  = np.zeros(8, dtype=np.float32)
        rm = np.zeros(8, dtype=np.float32)
        rv = np.ones(8,  dtype=np.float32)

        sx = Tensor(x, is_leaf=True)
        sg = Tensor(g, is_leaf=True)
        sb = Tensor(b, is_leaf=True)
        s_out = sx.batchnorm(sg, sb, rm, rv,
                             training=True, momentum=0.1, eps=1e-5, mode="2d")

        px = torch_tensor(x)
        p_bn = nn.BatchNorm2d(8, momentum=0.1, eps=1e-5)
        with torch.no_grad():
            p_bn.weight.fill_(1.0)
            p_bn.bias.fill_(0.0)
        p_out = p_bn(px.cpu())
        assert_close("bn2d_fwd", seera_to_np(s_out.value),
                     p_out.to(DEVICE), atol=1e-3)


# ══════════════════════════════════════════════════════════════════
#  9. DENSE LAYER GRADIENT FLOW (end-to-end)
# ══════════════════════════════════════════════════════════════════

class TestDenseGradient(unittest.TestCase):
    """Single Dense layer: check weight gradients against PyTorch linear."""

    def _dense_case(self, name, N, in_u, out_u, act):
        rng = np.random.RandomState(42)
        x_np = rng.randn(N, in_u).astype(np.float32)
        w_np = rng.randn(in_u, out_u).astype(np.float32) * 0.1
        b_np = rng.randn(1, out_u).astype(np.float32)    * 0.1

        torch_acts = {"relu": F.relu, "sigmoid": torch.sigmoid,
                      "tanh": torch.tanh}

        # Seera
        sx  = Tensor(x_np, is_leaf=True, device="cuda")
        sw  = Tensor(w_np, is_leaf=True, device="cuda")
        sb  = Tensor(b_np, is_leaf=True, device="cuda")
        s_z = sx.matmul(sw) + sb
        acts_map = {"relu": lambda t: t.relu(),
                    "sigmoid": lambda t: t.sigmoid(),
                    "tanh": lambda t: t.tanh()}
        s_out = acts_map[act](s_z)
        run_seera_backward(s_out.sum())

        # PyTorch
        px = torch_tensor(x_np, requires_grad=True)
        pw = torch_tensor(w_np, requires_grad=True)
        pb = torch_tensor(b_np, requires_grad=True)
        p_z = px @ pw + pb
        p_out = torch_acts[act](p_z)
        p_out.sum().backward()

        assert_close(f"{name}_fwd",    seera_to_np(s_out.value), p_out)
        assert_close(f"{name}_grad_w", seera_grad_to_np(sw),     pw.grad)
        assert_close(f"{name}_grad_b", seera_grad_to_np(sb),     pb.grad)
        assert_close(f"{name}_grad_x", seera_grad_to_np(sx),     px.grad)

    def test_dense_relu(self):     self._dense_case("d_relu",    16, 32, 16, "relu")
    def test_dense_sigmoid(self):  self._dense_case("d_sigmoid", 16, 32, 16, "sigmoid")
    def test_dense_tanh(self):     self._dense_case("d_tanh",    16, 32, 16, "tanh")
    def test_dense_large(self):    self._dense_case("d_large",   64, 256, 128, "relu")


# ══════════════════════════════════════════════════════════════════
#  10. LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════

class TestLossFunctions(unittest.TestCase):

    def test_mse_value_and_grad(self):
        rng = np.random.RandomState(0)
        pred = np.abs(rng.randn(8, 4).astype(np.float32))
        tgt  = np.abs(rng.randn(8, 4).astype(np.float32))

        sp = Tensor(pred, is_leaf=True, device="cuda")
        st = Tensor(tgt,  device="cuda")
        s_loss = ((sp - st) ** 2).mean()
        run_seera_backward(s_loss)

        pp = torch_tensor(pred, requires_grad=True)
        pt = torch_tensor(tgt)
        p_loss = F.mse_loss(pp, pt)
        p_loss.backward()

        assert_close("mse_val",  seera_to_np(s_loss.value),
                     p_loss.unsqueeze(0))
        assert_close("mse_grad", seera_grad_to_np(sp), pp.grad)

    def test_bce_value_and_grad(self):
        rng = np.random.RandomState(1)
        pred = (rng.randn(8, 1).astype(np.float32) * 0.3 + 0.5).clip(0.01, 0.99)
        tgt  = (rng.randint(0, 2, (8, 1))).astype(np.float32)

        sp = Tensor(pred, is_leaf=True, device="cuda")
        st = Tensor(tgt,  device="cuda")
        eps = 1e-7
        s_loss = (-(st * (sp + eps).log()) - ((1 - st) * ((1 - sp + eps).log()))).mean()
        run_seera_backward(s_loss)

        pp = torch_tensor(pred, requires_grad=True)
        pt = torch_tensor(tgt)
        p_loss = F.binary_cross_entropy(pp, pt)
        p_loss.backward()

        assert_close("bce_val",  seera_to_np(s_loss.value),
                     p_loss.unsqueeze(0), atol=1e-4)
        assert_close("bce_grad", seera_grad_to_np(sp), pp.grad, atol=1e-3)

    def test_cce_value_and_grad(self):
        rng = np.random.RandomState(2)
        logits = rng.randn(8, 10).astype(np.float32)
        labels = np.zeros((8, 10), dtype=np.float32)
        labels[np.arange(8), rng.randint(0, 10, 8)] = 1.0

        # Seera: softmax → CCE
        sl = Tensor(logits, is_leaf=True, device="cuda")
        sl_t = Tensor(labels, device="cuda")
        eps = 1e-7
        sp = sl.softmax()
        s_loss = (-(sl_t * (sp + eps).log())).sum(axis=-1).mean()
        run_seera_backward(s_loss)

        # PyTorch
        pl = torch_tensor(logits, requires_grad=True)
        pt = torch_tensor(labels)
        p_out = F.softmax(pl, dim=-1)
        p_loss = -(pt * torch.log(p_out + 1e-7)).sum(dim=-1).mean()
        p_loss.backward()

        assert_close("cce_val",  seera_to_np(s_loss.value),
                     p_loss.unsqueeze(0), atol=1e-3)
        assert_close("cce_grad", seera_grad_to_np(sl), pl.grad, atol=2e-3)


# ══════════════════════════════════════════════════════════════════
#  11. SEQUENTIAL MODEL — GRADIENT FLOW & TRAINING STEP
# ══════════════════════════════════════════════════════════════════

class TestSequentialModel(unittest.TestCase):
    """Build a small Sequential model on CUDA and check:
       (a) gradients are non-zero after backward
       (b) loss decreases after one optimizer step
    """

    def _build_mlp(self):
        return Sequential([
            Input((8,)),
            Dense(8,  16, activation="relu"),
            Dense(16,  8,  activation="relu"),
            Dense(8,   4,  activation="sigmoid"),
        ], device="cuda")

    def _build_cnn(self):
        return Sequential([
            Input((1, 8, 8)),
            Conv2D(4, 1, (3, 3), activation="relu", stride=1, zero_padding=1),
            MaxPool2D(pool_size=(2, 2), stride=2),
            Flatten(),
            Dense(64, 4, activation="sigmoid"),
        ], device="cuda")

    def test_mlp_grads_nonzero(self):
        model = self._build_mlp()
        rng = np.random.RandomState(0)
        X = rng.randn(4, 8).astype(np.float32)
        y = rng.randn(4, 4).astype(np.float32)
        ypred = model.forward(X)
        st = Tensor(y, device="cuda")
        loss = ((ypred - st) ** 2).mean()
        model.zero_grad()
        autograd4nn(loss)
        for layer in model.model:
            if hasattr(layer, "get_weights"):
                W, B = layer.get_weights()
                wg = seera_grad_to_np(W)
                bg = seera_grad_to_np(B)
                self.assertFalse(
                    np.allclose(wg, 0),
                    f"Weight gradient is all-zero in {layer} — model not training!")
                self.assertFalse(
                    np.allclose(bg, 0),
                    f"Bias gradient is all-zero in {layer} — model not training!")

    def test_mlp_loss_decreases_sgd(self):
        model = self._build_mlp()
        opt   = SGD(model, lr=0.05, momentum=0.0)
        rng   = np.random.RandomState(1)
        X     = rng.randn(16, 8).astype(np.float32)
        y     = rng.randn(16, 4).astype(np.float32)

        losses = []
        for _ in range(5):
            ypred = model.forward(X)
            st    = Tensor(y, device="cuda")
            loss  = ((ypred - st) ** 2).mean()
            lv    = float(seera_to_np(loss.value).ravel()[0])
            losses.append(lv)
            model.zero_grad()
            autograd4nn(loss)
            opt.step()

        self.assertLess(losses[-1], losses[0],
                        f"Loss did not decrease: {losses}")

    def test_mlp_loss_decreases_adam(self):
        model = self._build_mlp()
        opt   = Adam(model, lr=0.01)
        rng   = np.random.RandomState(2)
        X     = rng.randn(16, 8).astype(np.float32)
        y     = rng.randn(16, 4).astype(np.float32)

        losses = []
        for _ in range(5):
            ypred = model.forward(X)
            st    = Tensor(y, device="cuda")
            loss  = ((ypred - st) ** 2).mean()
            lv    = float(seera_to_np(loss.value).ravel()[0])
            losses.append(lv)
            model.zero_grad()
            autograd4nn(loss)
            opt.step()

        self.assertLess(losses[-1], losses[0],
                        f"Adam: loss did not decrease: {losses}")

    def test_cnn_grads_nonzero(self):
        model = self._build_cnn()
        rng = np.random.RandomState(3)
        X = rng.randn(4, 1, 8, 8).astype(np.float32)
        y = rng.randn(4, 4).astype(np.float32)
        ypred = model.forward(X)
        st = Tensor(y, device="cuda")
        loss = ((ypred - st) ** 2).mean()
        model.zero_grad()
        autograd4nn(loss)
        for layer in model.model:
            if hasattr(layer, "get_weights"):
                W, B = layer.get_weights()
                wg = seera_grad_to_np(W)
                self.assertFalse(
                    np.allclose(wg, 0),
                    f"Weight gradient is all-zero in {layer}")

    def test_cnn_loss_decreases_adam(self):
        model = self._build_cnn()
        opt   = Adam(model, lr=0.01)
        rng   = np.random.RandomState(4)
        X     = rng.randn(8, 1, 8, 8).astype(np.float32)
        y     = rng.randn(8, 4).astype(np.float32)

        losses = []
        for _ in range(8):
            ypred = model.forward(X)
            st    = Tensor(y, device="cuda")
            loss  = ((ypred - st) ** 2).mean()
            lv    = float(seera_to_np(loss.value).ravel()[0])
            losses.append(lv)
            model.zero_grad()
            autograd4nn(loss)
            opt.step()

        self.assertLess(losses[-1], losses[0],
                        f"CNN Adam: loss did not decrease: {losses}")


# ══════════════════════════════════════════════════════════════════
#  12. CHAIN RULE & COMPOSITE EXPRESSION TESTS
# ══════════════════════════════════════════════════════════════════

class TestChainRule(unittest.TestCase):
    """Complex composite expressions — really stresses the tape."""

    def test_sigmoid_of_matmul(self):
        rng = np.random.RandomState(55)
        a = rng.randn(8, 16).astype(np.float32)
        b = rng.randn(16, 4).astype(np.float32)

        sa = Tensor(a, is_leaf=True, device="cuda")
        sb = Tensor(b, is_leaf=True, device="cuda")
        s_out = sa.matmul(sb).sigmoid()
        run_seera_backward(s_out.sum())

        pa = torch_tensor(a, requires_grad=True)
        pb = torch_tensor(b, requires_grad=True)
        (torch.sigmoid(pa @ pb)).sum().backward()

        assert_close("chain_smm_grad_a", seera_grad_to_np(sa), pa.grad)
        assert_close("chain_smm_grad_b", seera_grad_to_np(sb), pb.grad)

    def test_relu_squared_sum(self):
        rng = np.random.RandomState(66)
        raw = rng.randn(4, 8).astype(np.float32)
        s_t = Tensor(raw, is_leaf=True, device="cuda")
        s_out = (s_t.relu() ** 2).sum()
        run_seera_backward(s_out)
        p_t = torch_tensor(raw, requires_grad=True)
        (F.relu(p_t) ** 2).sum().backward()
        assert_close("relu_sq_grad", seera_grad_to_np(s_t), p_t.grad)

    def test_exp_log_identity(self):
        """exp(log(x)) ≈ x for positive x; gradient should be ~1."""
        raw = np.abs(np.random.randn(4, 4).astype(np.float32)) + 0.5
        s_t = Tensor(raw, is_leaf=True, device="cuda")
        s_out = s_t.log().exp()
        run_seera_backward(s_out.sum())
        p_t = torch_tensor(raw, requires_grad=True)
        torch.exp(torch.log(p_t)).sum().backward()
        assert_close("exp_log_fwd",  seera_to_np(s_out.value),
                     torch.exp(torch.log(torch_tensor(raw))), atol=1e-4)
        assert_close("exp_log_grad", seera_grad_to_np(s_t), p_t.grad, atol=1e-4)

    def test_multi_op_chain(self):
        """(tanh(a * b) + sigmoid(c)) ^ 2"""
        rng = np.random.RandomState(77)
        a = rng.randn(4, 4).astype(np.float32) * 0.5
        b = rng.randn(4, 4).astype(np.float32) * 0.5
        c = rng.randn(4, 4).astype(np.float32)

        sa = Tensor(a, is_leaf=True, device="cuda")
        sb = Tensor(b, is_leaf=True, device="cuda")
        sc = Tensor(c, is_leaf=True, device="cuda")
        s_out = ((sa * sb).tanh() + sc.sigmoid()) ** 2
        run_seera_backward(s_out.sum())

        pa = torch_tensor(a, requires_grad=True)
        pb = torch_tensor(b, requires_grad=True)
        pc = torch_tensor(c, requires_grad=True)
        ((pa * pb).tanh() + pc.sigmoid()).pow(2).sum().backward()

        assert_close("chain_multi_a", seera_grad_to_np(sa), pa.grad)
        assert_close("chain_multi_b", seera_grad_to_np(sb), pb.grad)
        assert_close("chain_multi_c", seera_grad_to_np(sc), pc.grad)


# ══════════════════════════════════════════════════════════════════
#  13. NUMERICAL GRADIENT CHECK (finite differences)
# ══════════════════════════════════════════════════════════════════

class TestNumericalGradCheck(unittest.TestCase):
    """
    Perturb each element by ±eps and check (f(x+e) - f(x-e)) / (2*eps)
    matches Seera's analytic gradient.  Runs only on small tensors.
    """

    def _fd_check(self, name, seera_fn, shape, eps=1e-3, atol=1e-2):
        rng = np.random.RandomState(0)
        x = np.abs(rng.randn(*shape).astype(np.float32)) + 0.5

        # Analytic gradient
        sx = Tensor(x.copy(), is_leaf=True, device="cuda")
        s_out = seera_fn(sx)
        run_seera_backward(s_out.sum())
        analytic = seera_grad_to_np(sx).ravel()

        # Numerical gradient
        numeric = np.zeros_like(x.ravel())
        x_flat = x.ravel()
        for i in range(x_flat.size):
            xp = x_flat.copy(); xp[i] += eps
            xm = x_flat.copy(); xm[i] -= eps
            fp = seera_to_np(seera_fn(
                Tensor(xp.reshape(shape), is_leaf=True, device="cuda")).value).sum()
            fm = seera_to_np(seera_fn(
                Tensor(xm.reshape(shape), is_leaf=True, device="cuda")).value).sum()
            numeric[i] = (fp - fm) / (2 * eps)

        max_diff = np.abs(analytic - numeric).max()
        if max_diff > atol:
            raise AssertionError(
                f"[{name}] Finite-diff check FAILED  max_diff={max_diff:.6f}\n"
                f"  analytic: {analytic[:8]}\n"
                f"  numeric : {numeric[:8]}")

    def test_fd_sigmoid(self):
        self._fd_check("fd_sigmoid", lambda t: t.sigmoid(), (3, 4))

    def test_fd_tanh(self):
        self._fd_check("fd_tanh", lambda t: t.tanh(), (3, 4))

    def test_fd_relu(self):
        # avoid x≈0 where relu is non-smooth
        rng = np.random.RandomState(10)
        shape = (3, 4)
        x = rng.randn(*shape).astype(np.float32)
        x[np.abs(x) < 0.3] += 0.5  # keep away from 0
        eps = 1e-3
        sx = Tensor(x.copy(), is_leaf=True, device="cuda")
        run_seera_backward(sx.relu().sum())
        analytic = seera_grad_to_np(sx).ravel()
        numeric = np.zeros_like(analytic)
        xf = x.ravel()
        for i in range(xf.size):
            xp = xf.copy(); xp[i] += eps
            xm = xf.copy(); xm[i] -= eps
            fp = seera_to_np(
                Tensor(xp.reshape(shape), is_leaf=True, device="cuda")
                .relu().value).sum()
            fm = seera_to_np(
                Tensor(xm.reshape(shape), is_leaf=True, device="cuda")
                .relu().value).sum()
            numeric[i] = (fp - fm) / (2 * eps)
        max_diff = np.abs(analytic - numeric).max()
        self.assertLess(max_diff, 0.02, f"fd_relu max_diff={max_diff}")

    def test_fd_sqrt(self):
        self._fd_check("fd_sqrt", lambda t: t.sqrt(), (2, 4))

    def test_fd_log(self):
        self._fd_check("fd_log", lambda t: t.log(), (2, 4))

    def test_fd_pow2(self):
        self._fd_check("fd_pow2", lambda t: t ** 2, (3, 4))

    def test_fd_matmul(self):
        """Grad w.r.t. first argument of matmul."""
        rng = np.random.RandomState(42)
        a = rng.randn(4, 6).astype(np.float32)
        b = rng.randn(6, 3).astype(np.float32) * 0.3
        eps = 1e-3; atol = 2e-2
        sb = Tensor(b, is_leaf=True, device="cuda")
        sa = Tensor(a.copy(), is_leaf=True, device="cuda")
        run_seera_backward(sa.matmul(sb).sum())
        analytic = seera_grad_to_np(sa).ravel()
        numeric = np.zeros_like(analytic)
        af = a.ravel()
        for i in range(af.size):
            xp = af.copy(); xp[i] += eps
            xm = af.copy(); xm[i] -= eps
            fp = seera_to_np(
                Tensor(xp.reshape(a.shape), is_leaf=True, device="cuda")
                .matmul(Tensor(b, is_leaf=True, device="cuda")).value).sum()
            fm = seera_to_np(
                Tensor(xm.reshape(a.shape), is_leaf=True, device="cuda")
                .matmul(Tensor(b, is_leaf=True, device="cuda")).value).sum()
            numeric[i] = (fp - fm) / (2 * eps)
        max_diff = np.abs(analytic - numeric).max()
        self.assertLess(max_diff, atol,
                        f"fd_matmul max_diff={max_diff}")


# ══════════════════════════════════════════════════════════════════
#  14. ZERO_GRAD SANITY
# ══════════════════════════════════════════════════════════════════

class TestZeroGrad(unittest.TestCase):
    """Gradients must be cleanly zeroed between steps."""

    def test_grads_zero_after_zero_grad(self):
        model = Sequential([
            Input((4,)),
            Dense(4, 8, activation="relu"),
            Dense(8, 2, activation="sigmoid"),
        ], device="cuda")
        rng = np.random.RandomState(0)
        X = rng.randn(4, 4).astype(np.float32)
        y = rng.randn(4, 2).astype(np.float32)
        ypred = model.forward(X)
        st = Tensor(y, device="cuda")
        loss = ((ypred - st) ** 2).mean()
        autograd4nn(loss)
        model.zero_grad()
        for layer in model.model:
            if hasattr(layer, "get_weights"):
                W, B = layer.get_weights()
                wg = seera_grad_to_np(W)
                bg = seera_grad_to_np(B)
                np.testing.assert_allclose(
                    wg, 0, atol=1e-7,
                    err_msg=f"Weight grad not zeroed in {layer}")
                np.testing.assert_allclose(
                    bg, 0, atol=1e-7,
                    err_msg=f"Bias grad not zeroed in {layer}")

    def test_grad_accumulation_without_zero_grad(self):
        """Running backward twice without zeroing should double the grad."""
        rng = np.random.RandomState(1)
        raw = rng.randn(4, 8).astype(np.float32)
        w   = rng.randn(8, 4).astype(np.float32)
        sw  = Tensor(w, is_leaf=True, device="cuda")

        # First pass
        sx1 = Tensor(raw, is_leaf=True, device="cuda")
        out1 = sx1.matmul(sw)
        run_seera_backward(out1.sum())
        grad1 = seera_grad_to_np(sw).copy()

        # Second pass (no zero_grad — grad should accumulate)
        sx2 = Tensor(raw, is_leaf=True, device="cuda")
        out2 = sx2.matmul(sw)
        run_seera_backward(out2.sum())
        grad2 = seera_grad_to_np(sw)

        np.testing.assert_allclose(
            grad2, 2 * grad1, atol=1e-5,
            err_msg="Grad accumulation incorrect (expected 2×)")


# ══════════════════════════════════════════════════════════════════
#  15. BROADCAST GRADIENT REDUCTION
# ══════════════════════════════════════════════════════════════════

class TestBroadcast(unittest.TestCase):
    """Bias broadcast: (N, out) + (1, out) — bias grad must sum over batch."""

    def test_bias_broadcast_dense(self):
        rng = np.random.RandomState(0)
        x = rng.randn(8, 4).astype(np.float32)
        w = rng.randn(4, 6).astype(np.float32)
        b = rng.randn(1, 6).astype(np.float32)

        sx = Tensor(x, is_leaf=True, device="cuda")
        sw = Tensor(w, is_leaf=True, device="cuda")
        sb = Tensor(b, is_leaf=True, device="cuda")
        run_seera_backward((sx.matmul(sw) + sb).sum())

        px = torch_tensor(x, requires_grad=True)
        pw = torch_tensor(w, requires_grad=True)
        pb = torch_tensor(b, requires_grad=True)
        (px @ pw + pb).sum().backward()

        assert_close("bcast_grad_b", seera_grad_to_np(sb), pb.grad)
        assert_close("bcast_grad_w", seera_grad_to_np(sw), pw.grad)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Pretty header
    print("╔" + "═"*66 + "╗")
    print("║   SEERA DEEP LEARNING FRAMEWORK — CUDA GRADIENT TEST SUITE   ║")
    print("╚" + "═"*66 + "╝")
    try:
        import seera_cuda
        dev = "CUDA (seera_cuda loaded)"
    except Exception:
        dev = "CUDA backend missing — tests will fail"
    print(f"  Device : {dev}")
    print(f"  PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print()

    unittest.main(verbosity=2)