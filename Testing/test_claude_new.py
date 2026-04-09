"""
╔══════════════════════════════════════════════════════════════════════════╗
║   SEERA — EXTREME CONV2D / MAXPOOL2D / CONVTRANSPOSE2D TEST SUITE       ║
║   All forward values + ALL gradients compared element-by-element        ║
║   against PyTorch on CUDA.                                              ║
║                                                                         ║
║   Sections                                                              ║
║   ─────────────────────────────────────────────────────────────────     ║
║   A. Conv2D  — 30 cases (shapes, strides, pads, asymmetric, dilation)  ║
║   B. MaxPool2D — 20 cases                                               ║
║   C. ConvTranspose2D — 20 cases                                         ║
║   D. Chained sequences  (conv → pool → convT)                           ║
║   E. Full architectures — LeNet-5, VGG-mini, ResNet-block,             ║
║      U-Net encoder–decoder, AlexNet-mini, SqueezeNet-block,            ║
║      MobileNet-block, FCN-mini                                          ║
║   F. Numerical finite-difference double-check (conv fwd+bwd)           ║
║   G. Pathological cases — single-pixel, batch=1, large channels        ║
║   H. Gradient accumulation & zero_grad hygiene                         ║
║                                                                         ║
║   Run:  python test_seera_conv_extreme.py [-v]                         ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sys
import unittest
import numpy as np

# ── PyTorch ──────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("pip install torch  (CUDA build required)")

if not torch.cuda.is_available():
    sys.exit("A CUDA GPU is required.")

TORCH_DEVICE = torch.device("cuda")

# ── Seera ─────────────────────────────────────────────────────────────────
try:
    from Seera_init import tensor as Tensor
    from Seera_Engine import autograd4nn
    from Seera import (
        Sequential, Input, Dense, Conv2D, ConvTranspose2D,
        Flatten, MaxPool2D, Concatenate, Adam, SGD, Loss as SeeraLoss,
    )
    from cuTen import cuten
    import seera_cuda
except ImportError as exc:
    sys.exit(f"Seera / cuTen / seera_cuda import failed: {exc}")


# ════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ════════════════════════════════════════════════════════════════════════════

# Tolerances — we use float32 throughout; CUDA kernels may differ by ~1 ULP
ATOL_FWD  = 5e-4
ATOL_GRAD = 1e-3   # conv backward accumulates more error
RTOL      = 1e-3


def s2np(val):
    """Seera tensor value → float32 numpy (works for cuten and ndarray)."""
    if isinstance(val, cuten):
        return val.to_host_f32()
    return np.asarray(val, dtype=np.float32)


def grad_np(t: Tensor) -> np.ndarray:
    """Extract accumulated gradient from Seera Tensor leaf."""
    return s2np(t.node.cp)


def pt(arr, requires_grad=False):
    """numpy → float32 cuda torch.Tensor."""
    return torch.tensor(arr, dtype=torch.float32,
                        device=TORCH_DEVICE, requires_grad=requires_grad)


def _check(tag, s_arr, t_tensor,
           atol=ATOL_GRAD, rtol=RTOL):
    """Assert Seera array ≈ PyTorch tensor, print a diff on failure."""
    s = np.asarray(s_arr, dtype=np.float32)
    t = t_tensor.detach().cpu().numpy().astype(np.float32)
    if s.shape != t.shape:
        raise AssertionError(
            f"[{tag}] shape mismatch  Seera={s.shape}  PyTorch={t.shape}")
    max_abs = float(np.abs(s - t).max())
    max_rel = max_abs / (float(np.abs(t).max()) + 1e-8)
    if not np.allclose(s, t, atol=atol, rtol=rtol):
        raise AssertionError(
            f"[{tag}] MISMATCH  max_abs={max_abs:.6f}  max_rel={max_rel:.4f}\n"
            f"  Seera  : {s.ravel()[:10]}\n"
            f"  PyTorch: {t.ravel()[:10]}")


def seera_backward(loss_tensor: Tensor):
    """Run Seera backward pass."""
    autograd4nn(loss_tensor)


# ── shared weight init so Seera and PyTorch always start identical ────────
RNG = np.random.RandomState(2024)

def rand(*shape):
    return RNG.randn(*shape).astype(np.float32) * 0.1

def rand_pos(*shape):
    return (np.abs(RNG.randn(*shape)) + 0.3).astype(np.float32) * 0.1


# ════════════════════════════════════════════════════════════════════════════
#  BASE MIXIN  — shared conv / pool / convT comparison logic
# ════════════════════════════════════════════════════════════════════════════

class ConvTestMixin:
    """Shared helpers used by all conv-related test classes."""

    # ── Conv2D ──────────────────────────────────────────────────────────
    def _conv2d(self, tag, N, C, H, W, F_out, KH, KW,
                stride=1, pad=0, upstream_shape=None):
        """
        Run conv2d forward + backward in Seera and PyTorch with *identical*
        weights, compare fwd output, dX, dW, dB.
        upstream_shape: if given, use a random upstream gradient instead of
                        .sum() — this exercises the full VJP path.
        """
        x_np = rand(N, C, H, W)
        w_np = rand(F_out, C, KH, KW)
        b_np = rand(1, F_out, 1, 1)

        OH = (H + 2 * pad - KH) // stride + 1
        OW = (W + 2 * pad - KW) // stride + 1
        up_np = rand(N, F_out, OH, OW) if upstream_shape is None else rand(*upstream_shape)

        # ── Seera ──
        sx = Tensor(x_np,  is_leaf=True, device="cuda")
        sw = Tensor(w_np,  is_leaf=True, device="cuda")
        sb = Tensor(b_np,  is_leaf=True, device="cuda")
        s_conv = sx.conv2d(sw, stride=(stride, stride), padding=(pad, pad))
        s_out  = s_conv + sb
        s_up   = Tensor(up_np, device="cuda")
        seera_backward((s_out * s_up).sum())

        # ── PyTorch ──
        px = pt(x_np, requires_grad=True)
        pw = pt(w_np, requires_grad=True)
        pb = pt(b_np, requires_grad=True)
        p_out = F.conv2d(px, pw, bias=pb.view(F_out), stride=stride, padding=pad)
        (p_out * pt(up_np)).sum().backward()

        _check(f"{tag}_fwd",    s2np(s_out.value), p_out,  atol=ATOL_FWD)
        _check(f"{tag}_dX",     grad_np(sx),       px.grad)
        _check(f"{tag}_dW",     grad_np(sw),       pw.grad)
        _check(f"{tag}_dB",     grad_np(sb),       pb.grad)

    # ── MaxPool2D ────────────────────────────────────────────────────────
    def _maxpool(self, tag, N, C, H, W, KH, KW, stride, pad=0):
        x_np = rand(N, C, H, W)
        OH   = (H + 2 * pad - KH) // stride + 1
        OW   = (W + 2 * pad - KW) // stride + 1
        up_np = rand(N, C, OH, OW)

        sx   = Tensor(x_np, is_leaf=True, device="cuda")
        s_p  = sx.maxpool2d(kernelsize=(KH, KW),
                            stride=(stride, stride),
                            padding=(pad, pad))
        s_up = Tensor(up_np, device="cuda")
        seera_backward((s_p * s_up).sum())

        px   = pt(x_np, requires_grad=True)
        p_p  = F.max_pool2d(px, kernel_size=(KH, KW),
                            stride=stride, padding=pad)
        (p_p * pt(up_np)).sum().backward()

        _check(f"{tag}_fwd",  s2np(s_p.value), p_p,   atol=ATOL_FWD)
        _check(f"{tag}_dX",   grad_np(sx),     px.grad)

    # ── ConvTranspose2D ──────────────────────────────────────────────────
    def _convT(self, tag, N, Cin, H, W, Cout, KH, KW, stride=1, pad=0):
        """
        Seera ConvTranspose weight layout: (Cin, Cout, KH, KW)
        PyTorch ConvTranspose weight layout: (Cin, Cout, KH, KW)  ← same
        """
        x_np = rand(N, Cin, H, W)
        w_np = rand(Cin, Cout, KH, KW)
        b_np = rand(1, Cout, 1, 1)

        Hout = (H - 1) * stride - 2 * pad + KH
        Wout = (W - 1) * stride - 2 * pad + KW
        up_np = rand(N, Cout, Hout, Wout)

        # Seera
        sx  = Tensor(x_np,  is_leaf=True, device="cuda")
        sw  = Tensor(w_np,  is_leaf=True, device="cuda")
        sb  = Tensor(b_np,  is_leaf=True, device="cuda")
        s_cT = sx.conv_transpose2d(sw, stride=(stride, stride), padding=(pad, pad))
        s_out = s_cT + sb
        s_up  = Tensor(up_np, device="cuda")
        seera_backward((s_out * s_up).sum())

        # PyTorch
        px  = pt(x_np, requires_grad=True)
        pw  = pt(w_np, requires_grad=True)
        pb  = pt(b_np, requires_grad=True)
        p_out = F.conv_transpose2d(px, pw,
                                   bias=pb.view(Cout),
                                   stride=stride, padding=pad)
        (p_out * pt(up_np)).sum().backward()

        _check(f"{tag}_fwd", s2np(s_out.value), p_out,   atol=ATOL_FWD)
        _check(f"{tag}_dX",  grad_np(sx),       px.grad, atol=ATOL_GRAD)
        _check(f"{tag}_dW",  grad_np(sw),       pw.grad, atol=ATOL_GRAD)
        _check(f"{tag}_dB",  grad_np(sb),       pb.grad, atol=ATOL_GRAD)


# ════════════════════════════════════════════════════════════════════════════
#  A. CONV2D — 30 CASES
# ════════════════════════════════════════════════════════════════════════════

class TestConv2D(ConvTestMixin, unittest.TestCase):

    # ── stride-1, no padding ──────────────────────────────────────────────
    def test_c01_3x3_s1_p0(self):     self._conv2d("c01", 2, 3,  8,  8,  4, 3, 3)
    def test_c02_5x5_s1_p0(self):     self._conv2d("c02", 2, 3,  8,  8,  4, 5, 5)
    def test_c03_1x1_s1_p0(self):     self._conv2d("c03", 4, 16, 8,  8,  32, 1, 1)
    def test_c04_7x7_s1_p0(self):     self._conv2d("c04", 2, 3,  12, 12, 8, 7, 7)

    # ── same-padding (p = k//2) ───────────────────────────────────────────
    def test_c05_3x3_s1_p1(self):     self._conv2d("c05", 2, 3,  8,  8,  4, 3, 3, pad=1)
    def test_c06_5x5_s1_p2(self):     self._conv2d("c06", 2, 3,  8,  8,  4, 5, 5, pad=2)
    def test_c07_7x7_s1_p3(self):     self._conv2d("c07", 2, 3,  14, 14, 8, 7, 7, pad=3)

    # ── stride-2 ──────────────────────────────────────────────────────────
    def test_c08_3x3_s2_p0(self):     self._conv2d("c08", 2, 3,  8,  8,  4, 3, 3, stride=2)
    def test_c09_3x3_s2_p1(self):     self._conv2d("c09", 2, 3,  8,  8,  4, 3, 3, stride=2, pad=1)
    def test_c10_5x5_s2_p2(self):     self._conv2d("c10", 2, 3,  10, 10, 4, 5, 5, stride=2, pad=2)
    def test_c11_3x3_s3_p1(self):     self._conv2d("c11", 2, 4,  12, 12, 8, 3, 3, stride=3, pad=1)

    # ── batch sizes ───────────────────────────────────────────────────────
    def test_c12_batch1(self):        self._conv2d("c12", 1, 3,  8,  8,  4, 3, 3, pad=1)
    def test_c13_batch8(self):        self._conv2d("c13", 8, 3,  8,  8,  4, 3, 3, pad=1)
    def test_c14_batch32(self):       self._conv2d("c14", 32, 3, 8,  8,  4, 3, 3, pad=1)

    # ── channel depth ─────────────────────────────────────────────────────
    def test_c15_deep_channels(self): self._conv2d("c15", 2, 64, 8,  8,  64, 3, 3, pad=1)
    def test_c16_1ch_in(self):        self._conv2d("c16", 4, 1,  16, 16, 8,  3, 3, pad=1)
    def test_c17_1ch_out(self):       self._conv2d("c17", 4, 8,  8,  8,  1,  3, 3, pad=1)

    # ── non-square spatial ────────────────────────────────────────────────
    def test_c18_rect_input(self):    self._conv2d("c18", 2, 3,  4,  16, 4,  3, 3, pad=1)
    def test_c19_tall_input(self):    self._conv2d("c19", 2, 3,  16, 4,  4,  3, 3, pad=1)

    # ── non-square kernels ────────────────────────────────────────────────
    def test_c20_1x3_kernel(self):    self._conv2d("c20", 2, 4,  8,  8,  8,  1, 3, pad=0)
    def test_c21_3x1_kernel(self):    self._conv2d("c21", 2, 4,  8,  8,  8,  3, 1, pad=0)
    def test_c22_1x5_kernel(self):    self._conv2d("c22", 2, 4,  8,  8,  8,  1, 5, pad=0)

    # ── large spatial ─────────────────────────────────────────────────────
    def test_c23_32x32(self):         self._conv2d("c23", 2, 3,  32, 32, 8,  3, 3, pad=1)
    def test_c24_64x64(self):         self._conv2d("c24", 2, 1,  64, 64, 4,  5, 5, stride=2, pad=2)

    # ── random upstream gradient (not just .sum()) ────────────────────────
    def test_c25_random_upstream(self):
        self._conv2d("c25", 4, 3, 8, 8, 4, 3, 3, stride=1, pad=1,
                     upstream_shape=(4, 4, 8, 8))

    def test_c26_upstream_stride2(self):
        self._conv2d("c26", 4, 3, 8, 8, 4, 3, 3, stride=2, pad=1,
                     upstream_shape=(4, 4, 4, 4))

    # ── edge: kernel == input spatial ─────────────────────────────────────
    def test_c27_global_conv(self):   self._conv2d("c27", 2, 4, 5, 5, 8, 5, 5)

    # ── many output channels, small spatial ───────────────────────────────
    def test_c28_many_filt(self):     self._conv2d("c28", 2, 3, 6, 6, 128, 3, 3, pad=1)

    # ── stride larger than kernel (produces gaps — unusual but valid) ──────
    def test_c29_stride_gt_kernel(self):  self._conv2d("c29", 2, 4, 12, 12, 8, 2, 2, stride=3)

    # ── accumulation: run backward twice, check grads double ──────────────
    def test_c30_grad_accumulates(self):
        x_np = rand(2, 3, 8, 8)
        w_np = rand(4, 3, 3, 3)
        sx = Tensor(x_np, is_leaf=True, device="cuda")
        sw = Tensor(w_np, is_leaf=True, device="cuda")
        # first pass
        seera_backward(sx.conv2d(sw, stride=(1,1), padding=(1,1)).sum())
        g1_w = grad_np(sw).copy()
        g1_x = grad_np(sx).copy()
        # second pass (no zero_grad)
        seera_backward(sx.conv2d(sw, stride=(1,1), padding=(1,1)).sum())
        g2_w = grad_np(sw)
        g2_x = grad_np(sx)
        np.testing.assert_allclose(g2_w, 2 * g1_w, atol=1e-4,
                                   err_msg="conv dW not accumulating correctly")
        np.testing.assert_allclose(g2_x, 2 * g1_x, atol=1e-4,
                                   err_msg="conv dX not accumulating correctly")


# ════════════════════════════════════════════════════════════════════════════
#  B. MAXPOOL2D — 20 CASES
# ════════════════════════════════════════════════════════════════════════════

class TestMaxPool2D(ConvTestMixin, unittest.TestCase):

    # ── stride = kernel (non-overlapping) ────────────────────────────────
    def test_p01_2x2_s2(self):   self._maxpool("p01", 2, 4,  8,  8,  2, 2, 2)
    def test_p02_3x3_s3(self):   self._maxpool("p02", 2, 4,  9,  9,  3, 3, 3)
    def test_p03_4x4_s4(self):   self._maxpool("p03", 2, 4,  16, 16, 4, 4, 4)

    # ── overlapping stride ────────────────────────────────────────────────
    def test_p04_2x2_s1(self):   self._maxpool("p04", 2, 4,  8,  8,  2, 2, 1)
    def test_p05_3x3_s2(self):   self._maxpool("p05", 2, 4,  8,  8,  3, 3, 2)
    def test_p06_3x3_s1(self):   self._maxpool("p06", 2, 4,  8,  8,  3, 3, 1)

    # ── padding ───────────────────────────────────────────────────────────
    def test_p07_2x2_s2_p1(self): self._maxpool("p07", 2, 4,  8,  8,  2, 2, 2, pad=1)
    def test_p08_3x3_s2_p1(self): self._maxpool("p08", 2, 4,  8,  8,  3, 3, 2, pad=1)

    # ── batch sizes ───────────────────────────────────────────────────────
    def test_p09_batch1(self):    self._maxpool("p09", 1, 8,  8,  8,  2, 2, 2)
    def test_p10_batch16(self):   self._maxpool("p10", 16, 8, 8,  8,  2, 2, 2)

    # ── channel depth ─────────────────────────────────────────────────────
    def test_p11_deep(self):      self._maxpool("p11", 2, 64, 8,  8,  2, 2, 2)
    def test_p12_1ch(self):       self._maxpool("p12", 2, 1,  8,  8,  2, 2, 2)

    # ── non-square spatial ────────────────────────────────────────────────
    def test_p13_rect(self):      self._maxpool("p13", 2, 4,  4,  16, 2, 2, 2)
    def test_p14_tall(self):      self._maxpool("p14", 2, 4,  16, 4,  2, 2, 2)

    # ── large spatial ─────────────────────────────────────────────────────
    def test_p15_32x32(self):     self._maxpool("p15", 2, 4,  32, 32, 2, 2, 2)
    def test_p16_64x64(self):     self._maxpool("p16", 2, 4,  64, 64, 4, 4, 4)

    # ── global pool equivalent (k == spatial) ────────────────────────────
    def test_p17_global(self):    self._maxpool("p17", 4, 8,  4,  4,  4, 4, 4)

    # ── upstream gradient not all-ones ────────────────────────────────────
    def test_p18_random_upstream(self):
        """Verify the mask-based scatter works for any upstream gradient."""
        N, C, H, W, K, S = 2, 4, 8, 8, 2, 2
        x_np  = rand(N, C, H, W)
        up_np = rand(N, C, H//S, W//S)
        sx   = Tensor(x_np, is_leaf=True, device="cuda")
        s_p  = sx.maxpool2d(kernelsize=(K, K), stride=(S, S))
        seera_backward((s_p * Tensor(up_np, device="cuda")).sum())
        px = pt(x_np, requires_grad=True)
        (F.max_pool2d(px, K, stride=S) * pt(up_np)).sum().backward()
        _check("p18_dX", grad_np(sx), px.grad)

    # ── ties: when two elements are equal the backward should still work ──
    def test_p19_ties(self):
        """Force ties by making all elements in each window equal."""
        N, C, H, W = 2, 4, 8, 8
        # constant input → every element ties for max → grad still valid
        x_np = np.ones((N, C, H, W), dtype=np.float32)
        sx   = Tensor(x_np, is_leaf=True, device="cuda")
        s_p  = sx.maxpool2d(kernelsize=(2, 2), stride=(2, 2))
        seera_backward(s_p.sum())
        # grads must be non-negative and sum to total output elements
        g = grad_np(sx)
        self.assertGreaterEqual(float(g.min()), 0,
                                "Ties: negative gradient in maxpool backward")
        # each output element's gradient must have landed somewhere in input
        self.assertAlmostEqual(float(g.sum()), float(s_p.value.size if hasattr(s_p.value, 'size') else np.prod(s2np(s_p.value).shape)),
                               places=3,
                               msg="Ties: gradient mass doesn't match output size")

    # ── conv → pool chain: dX must flow all the way back ─────────────────
    def test_p20_conv_pool_chain(self):
        x_np = rand(2, 3, 8, 8)
        w_np = rand(4, 3, 3, 3)
        sx   = Tensor(x_np, is_leaf=True, device="cuda")
        sw   = Tensor(w_np, is_leaf=True, device="cuda")
        s_c  = sx.conv2d(sw, stride=(1, 1), padding=(1, 1))
        s_p  = s_c.maxpool2d(kernelsize=(2, 2), stride=(2, 2))
        seera_backward(s_p.sum())

        px = pt(x_np, requires_grad=True)
        pw = pt(w_np, requires_grad=True)
        F.max_pool2d(F.conv2d(px, pw, padding=1), 2, stride=2).sum().backward()

        _check("cp_dX",  grad_np(sx), px.grad)
        _check("cp_dW",  grad_np(sw), pw.grad)


# ════════════════════════════════════════════════════════════════════════════
#  C. CONVTRANSPOSE2D — 20 CASES
# ════════════════════════════════════════════════════════════════════════════

class TestConvTranspose2D(ConvTestMixin, unittest.TestCase):

    # ── stride-1 (== regular conv of rotated kernel) ─────────────────────
    def test_t01_3x3_s1_p0(self):  self._convT("t01", 2, 4,  4,  4,  8,  3, 3)
    def test_t02_3x3_s1_p1(self):  self._convT("t02", 2, 4,  4,  4,  8,  3, 3, pad=1)
    def test_t03_5x5_s1_p2(self):  self._convT("t03", 2, 4,  4,  4,  8,  5, 5, pad=2)
    def test_t04_1x1_s1_p0(self):  self._convT("t04", 4, 16, 8,  8,  8,  1, 1)

    # ── stride-2 (classic decoder upsampling) ────────────────────────────
    def test_t05_3x3_s2_p0(self):  self._convT("t05", 2, 4,  4,  4,  8,  3, 3, stride=2)
    def test_t06_4x4_s2_p1(self):  self._convT("t06", 2, 4,  4,  4,  8,  4, 4, stride=2, pad=1)
    def test_t07_3x3_s2_p1(self):  self._convT("t07", 2, 8,  4,  4,  4,  3, 3, stride=2, pad=1)
    def test_t08_2x2_s2_p0(self):  self._convT("t08", 2, 4,  4,  4,  4,  2, 2, stride=2)

    # ── stride-3 ──────────────────────────────────────────────────────────
    def test_t09_3x3_s3_p0(self):  self._convT("t09", 2, 4,  3,  3,  4,  3, 3, stride=3)

    # ── batch sizes ───────────────────────────────────────────────────────
    def test_t10_batch1(self):     self._convT("t10", 1, 4,  4,  4,  8,  3, 3, stride=2, pad=1)
    def test_t11_batch16(self):    self._convT("t11", 16, 4, 4,  4,  8,  3, 3, stride=2, pad=1)

    # ── channel combinations ──────────────────────────────────────────────
    def test_t12_deep_in(self):    self._convT("t12", 2, 64, 4,  4,  32, 3, 3, stride=2, pad=1)
    def test_t13_1ch_in(self):     self._convT("t13", 2, 1,  8,  8,  8,  3, 3, stride=2, pad=1)
    def test_t14_1ch_out(self):    self._convT("t14", 2, 8,  4,  4,  1,  3, 3, stride=2, pad=1)

    # ── non-square spatial ────────────────────────────────────────────────
    def test_t15_rect(self):       self._convT("t15", 2, 4,  2,  8,  8,  3, 3, stride=2, pad=1)
    def test_t16_tall(self):       self._convT("t16", 2, 4,  8,  2,  8,  3, 3, stride=2, pad=1)

    # ── non-square kernels ────────────────────────────────────────────────
    def test_t17_1x3_kernel(self): self._convT("t17", 2, 4,  4,  4,  8,  1, 3)
    def test_t18_3x1_kernel(self): self._convT("t18", 2, 4,  4,  4,  8,  3, 1)

    # ── random upstream (non-trivial VJP) ────────────────────────────────
    def test_t19_random_upstream(self):
        N, Cin, H, W, Cout, K, S, P = 2, 4, 4, 4, 8, 3, 2, 1
        x_np = rand(N, Cin, H, W)
        w_np = rand(Cin, Cout, K, K)
        b_np = rand(1, Cout, 1, 1)
        Hout = (H - 1) * S - 2 * P + K
        Wout = (W - 1) * S - 2 * P + K
        up_np = rand(N, Cout, Hout, Wout)
        sx = Tensor(x_np, is_leaf=True, device="cuda")
        sw = Tensor(w_np, is_leaf=True, device="cuda")
        sb = Tensor(b_np, is_leaf=True, device="cuda")
        s_out = sx.conv_transpose2d(sw, stride=(S,S), padding=(P,P)) + sb
        seera_backward((s_out * Tensor(up_np, device="cuda")).sum())
        px = pt(x_np, requires_grad=True)
        pw = pt(w_np, requires_grad=True)
        pb = pt(b_np, requires_grad=True)
        p_out = F.conv_transpose2d(px, pw, bias=pb.view(Cout), stride=S, padding=P)
        (p_out * pt(up_np)).sum().backward()
        _check("t19_fwd", s2np(s_out.value), p_out, atol=ATOL_FWD)
        _check("t19_dX",  grad_np(sx), px.grad, atol=ATOL_GRAD)
        _check("t19_dW",  grad_np(sw), pw.grad, atol=ATOL_GRAD)
        _check("t19_dB",  grad_np(sb), pb.grad, atol=ATOL_GRAD)

    # ── convT → relu → sum, gradient flows through activation ────────────
    def test_t20_with_relu(self):
        N, Cin, H, W, Cout, K, S, P = 2, 4, 4, 4, 8, 3, 2, 1
        x_np = rand(N, Cin, H, W)
        w_np = rand(Cin, Cout, K, K)
        sx = Tensor(x_np, is_leaf=True, device="cuda")
        sw = Tensor(w_np, is_leaf=True, device="cuda")
        s_out = sx.conv_transpose2d(sw, stride=(S,S), padding=(P,P)).relu()
        seera_backward(s_out.sum())
        px = pt(x_np, requires_grad=True)
        pw = pt(w_np, requires_grad=True)
        F.relu(F.conv_transpose2d(px, pw, stride=S, padding=P)).sum().backward()
        _check("t20_dX", grad_np(sx), px.grad, atol=ATOL_GRAD)
        _check("t20_dW", grad_np(sw), pw.grad, atol=ATOL_GRAD)


# ════════════════════════════════════════════════════════════════════════════
#  D. CHAINED SEQUENCES
# ════════════════════════════════════════════════════════════════════════════

class TestChainedLayers(ConvTestMixin, unittest.TestCase):
    """Multi-op chains — verifies the backward graph is stitched correctly."""

    def test_conv_relu_pool(self):
        x_np = rand(2, 3, 16, 16)
        w_np = rand(8, 3, 3, 3)
        sx = Tensor(x_np, is_leaf=True, device="cuda")
        sw = Tensor(w_np, is_leaf=True, device="cuda")
        s_out = sx.conv2d(sw, stride=(1,1), padding=(1,1)).relu()
        s_p   = s_out.maxpool2d(kernelsize=(2,2), stride=(2,2))
        seera_backward(s_p.sum())
        px = pt(x_np, requires_grad=True)
        pw = pt(w_np, requires_grad=True)
        F.max_pool2d(F.relu(F.conv2d(px, pw, padding=1)), 2).sum().backward()
        _check("crp_dX", grad_np(sx), px.grad)
        _check("crp_dW", grad_np(sw), pw.grad)

    def test_conv_pool_convT_relu(self):
        """Encoder then decoder — the most common pattern in segmentation."""
        N, C, H, W = 2, 3, 8, 8
        x_np = rand(N, C, H, W)
        w1_np = rand(4, C, 3, 3)           # conv
        w2_np = rand(4, 8, 4, 4)           # convT  (Cin=4, Cout=8)

        sx  = Tensor(x_np, is_leaf=True, device="cuda")
        sw1 = Tensor(w1_np, is_leaf=True, device="cuda")
        sw2 = Tensor(w2_np, is_leaf=True, device="cuda")
        s_c = sx.conv2d(sw1, stride=(1,1), padding=(1,1)).relu()   # (N,4,8,8)
        s_p = s_c.maxpool2d(kernelsize=(2,2), stride=(2,2))         # (N,4,4,4)
        s_t = s_p.conv_transpose2d(sw2, stride=(2,2), padding=(1,1)).relu()  # (N,8,8,8)
        seera_backward(s_t.sum())

        px  = pt(x_np, requires_grad=True)
        pw1 = pt(w1_np, requires_grad=True)
        pw2 = pt(w2_np, requires_grad=True)
        p_c = F.relu(F.conv2d(px, pw1, padding=1))
        p_p = F.max_pool2d(p_c, 2, stride=2)
        p_t = F.relu(F.conv_transpose2d(p_p, pw2, stride=2, padding=1))
        p_t.sum().backward()

        _check("chain_fwd", s2np(s_t.value), p_t,   atol=ATOL_FWD)
        _check("chain_dX",  grad_np(sx),     px.grad)
        _check("chain_dW1", grad_np(sw1),    pw1.grad)
        _check("chain_dW2", grad_np(sw2),    pw2.grad)

    def test_three_conv_stack(self):
        x_np  = rand(2, 3, 16, 16)
        w1_np = rand(8,  3, 3, 3)
        w2_np = rand(16, 8, 3, 3)
        w3_np = rand(8, 16, 3, 3)
        sx  = Tensor(x_np,  is_leaf=True, device="cuda")
        sw1 = Tensor(w1_np, is_leaf=True, device="cuda")
        sw2 = Tensor(w2_np, is_leaf=True, device="cuda")
        sw3 = Tensor(w3_np, is_leaf=True, device="cuda")
        s = sx.conv2d(sw1, padding=(1,1)).relu()
        s = s.conv2d(sw2, padding=(1,1)).relu()
        s = s.conv2d(sw3, padding=(1,1))
        seera_backward(s.sum())

        px  = pt(x_np,  requires_grad=True)
        pw1 = pt(w1_np, requires_grad=True)
        pw2 = pt(w2_np, requires_grad=True)
        pw3 = pt(w3_np, requires_grad=True)
        p = F.relu(F.conv2d(px,  pw1, padding=1))
        p = F.relu(F.conv2d(p,   pw2, padding=1))
        p = F.conv2d(p, pw3, padding=1)
        p.sum().backward()

        _check("3conv_dX",  grad_np(sx),  px.grad)
        _check("3conv_dW1", grad_np(sw1), pw1.grad)
        _check("3conv_dW2", grad_np(sw2), pw2.grad)
        _check("3conv_dW3", grad_np(sw3), pw3.grad)

    def test_pool_between_convTs(self):
        """Two ConvTranspose layers with activation between them."""
        x_np  = rand(2, 8, 2, 2)
        w1_np = rand(8, 16, 4, 4)   # stride-2 upsample
        w2_np = rand(16, 8, 4, 4)   # stride-2 upsample again
        sx  = Tensor(x_np,  is_leaf=True, device="cuda")
        sw1 = Tensor(w1_np, is_leaf=True, device="cuda")
        sw2 = Tensor(w2_np, is_leaf=True, device="cuda")
        s = sx.conv_transpose2d(sw1, stride=(2,2), padding=(1,1)).relu()  # (2,16,4,4)
        s = s.conv_transpose2d(sw2, stride=(2,2), padding=(1,1))          # (2,8,8,8)
        seera_backward(s.sum())

        px  = pt(x_np,  requires_grad=True)
        pw1 = pt(w1_np, requires_grad=True)
        pw2 = pt(w2_np, requires_grad=True)
        p = F.relu(F.conv_transpose2d(px,  pw1, stride=2, padding=1))
        p = F.conv_transpose2d(p, pw2, stride=2, padding=1)
        p.sum().backward()

        _check("2cT_dX",  grad_np(sx),  px.grad,  atol=ATOL_GRAD)
        _check("2cT_dW1", grad_np(sw1), pw1.grad, atol=ATOL_GRAD)
        _check("2cT_dW2", grad_np(sw2), pw2.grad, atol=ATOL_GRAD)


# ════════════════════════════════════════════════════════════════════════════
#  E. FULL ARCHITECTURES
#  For each: build identical Seera & PyTorch models with exactly the same
#  weights, run forward + backward, compare every layer's weight gradient.
# ════════════════════════════════════════════════════════════════════════════

class ArchTestBase(unittest.TestCase):
    """Helpers for architecture-level tests."""

    def _run_arch(self, seera_model, torch_model,
                  input_np, target_np,
                  torch_device=TORCH_DEVICE):
        """
        Forward + backward in both frameworks.
        Returns (seera_layers_with_weights, torch_named_params_dict).
        """
        # ── Seera ──
        ypred_s = seera_model.forward(input_np)
        tgt_s   = Tensor(target_np, device="cuda")
        loss_s  = ((ypred_s - tgt_s) ** 2).mean()
        seera_model.zero_grad()
        seera_backward(loss_s)

        # ── PyTorch ──
        x_t   = pt(input_np)
        y_t   = pt(target_np)
        torch_model.zero_grad()
        p_out = torch_model(x_t)
        loss_p = F.mse_loss(p_out, y_t)
        loss_p.backward()

        return loss_s, loss_p

    def _loss_close(self, tag, loss_s, loss_p):
        sv = float(s2np(loss_s.value).ravel()[0])
        pv = float(loss_p.item())
        self.assertAlmostEqual(sv, pv, places=3,
                               msg=f"[{tag}] loss mismatch  seera={sv:.6f}  pytorch={pv:.6f}")

    def _check_grad(self, tag, seera_layer, torch_param_grad,
                    atol=ATOL_GRAD):
        W, B = seera_layer.get_weights()
        _check(f"{tag}_dW", grad_np(W), torch_param_grad[0], atol=atol)
        _check(f"{tag}_dB", grad_np(B), torch_param_grad[1], atol=atol)

    @staticmethod
    def _sync_conv2d(seera_layer, torch_layer):
        """Copy PyTorch weight/bias into Seera layer so they start identical."""
        w = torch_layer.weight.detach().cpu().numpy().astype(np.float32)
        b = torch_layer.bias.detach().cpu().numpy().astype(np.float32).reshape(1, -1, 1, 1)
        seera_layer.set_weights(w, b)
        # move to GPU
        from Seera import Sequential as _S
        _S._tensor_to_gpu(seera_layer.weights)
        _S._tensor_to_gpu(seera_layer.bais)

    @staticmethod
    def _sync_convT(seera_layer, torch_layer):
        """ConvTranspose: torch (Cin,Cout,K,K) == seera (Cin,Cout,K,K)."""
        w = torch_layer.weight.detach().cpu().numpy().astype(np.float32)
        b = torch_layer.bias.detach().cpu().numpy().astype(np.float32).reshape(1, -1, 1, 1)
        seera_layer.set_weights(w, b)
        from Seera import Sequential as _S
        _S._tensor_to_gpu(seera_layer.weights)
        _S._tensor_to_gpu(seera_layer.bais)

    @staticmethod
    def _sync_dense(seera_layer, torch_layer):
        """Dense: seera (in, out), PyTorch linear weight (out, in)."""
        w = torch_layer.weight.detach().cpu().numpy().T.astype(np.float32)
        b = torch_layer.bias.detach().cpu().numpy().reshape(1, -1).astype(np.float32)
        seera_layer.set_weights(w, b)
        from Seera import Sequential as _S
        _S._tensor_to_gpu(seera_layer.weights)
        _S._tensor_to_gpu(seera_layer.bais)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture 1: LeNet-5
# ─────────────────────────────────────────────────────────────────────────────
class TestLeNet5(ArchTestBase):
    """
    Classic LeNet-5 (adapted for 1×28×28 → 10-class):
      Conv(1→6, 5×5, pad=2) → relu → Pool(2×2) →
      Conv(6→16, 5×5)        → relu → Pool(2×2) →
      Flatten → Dense(256→120) → relu →
      Dense(120→84) → relu → Dense(84→10) → sigmoid
    """

    def _build_seera(self, layers):
        """layers = list of (conv/dense, torch_ref) — used only for weight sync."""
        return Sequential([
            Input((1, 28, 28)),
            layers[0],          # conv1
            MaxPool2D((2, 2), stride=2),
            layers[1],          # conv2
            MaxPool2D((2, 2), stride=2),
            Flatten(),
            layers[2],          # fc1
            layers[3],          # fc2
            layers[4],          # fc3
        ], device="cuda")

    def test_lenet5_grad_match(self):
        N = 4
        rng = np.random.RandomState(0)
        x_np = rng.randn(N, 1, 28, 28).astype(np.float32) * 0.1
        y_np = rng.randn(N, 10).astype(np.float32)

        # ── PyTorch reference model ──
        class PTLeNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(1, 6, 5, padding=2)
                self.c2 = nn.Conv2d(6, 16, 5)
                self.f1 = nn.Linear(16 * 5 * 5, 120)
                self.f2 = nn.Linear(120, 84)
                self.f3 = nn.Linear(84, 10)
            def forward(self, x):
                x = F.max_pool2d(F.relu(self.c1(x)), 2)
                x = F.max_pool2d(F.relu(self.c2(x)), 2)
                x = x.view(x.size(0), -1)
                x = F.relu(self.f1(x))
                x = F.relu(self.f2(x))
                return torch.sigmoid(self.f3(x))

        pt_model = PTLeNet().to(TORCH_DEVICE)

        # ── Seera model with synced weights ──
        c1 = Conv2D(6,  1,  (5, 5), activation="relu", zero_padding=2)
        c2 = Conv2D(16, 6,  (5, 5), activation="relu")
        f1 = Dense(16 * 5 * 5, 120, activation="relu")
        f2 = Dense(120, 84, activation="relu")
        f3 = Dense(84, 10, activation="sigmoid")

        self._sync_conv2d(c1, pt_model.c1)
        self._sync_conv2d(c2, pt_model.c2)
        self._sync_dense(f1, pt_model.f1)
        self._sync_dense(f2, pt_model.f2)
        self._sync_dense(f3, pt_model.f3)

        s_model = Sequential([
            Input((1, 28, 28)), c1,
            MaxPool2D((2, 2), stride=2),
            c2,
            MaxPool2D((2, 2), stride=2),
            Flatten(), f1, f2, f3,
        ], device="cuda")

        loss_s, loss_p = self._run_arch(s_model, pt_model, x_np, y_np)
        self._loss_close("lenet5", loss_s, loss_p)

        # Compare every layer's weight gradient
        layers_s  = [c1, c2, f1, f2, f3]
        layers_pt = [
            (pt_model.c1.weight.grad, pt_model.c1.bias.grad.view(1,-1,1,1)),
            (pt_model.c2.weight.grad, pt_model.c2.bias.grad.view(1,-1,1,1)),
            (pt_model.f1.weight.grad.T, pt_model.f1.bias.grad.view(1,-1)),
            (pt_model.f2.weight.grad.T, pt_model.f2.bias.grad.view(1,-1)),
            (pt_model.f3.weight.grad.T, pt_model.f3.bias.grad.view(1,-1)),
        ]
        names = ["c1", "c2", "f1", "f2", "f3"]
        for name, sl, (wg, bg) in zip(names, layers_s, layers_pt):
            W, B = sl.get_weights()
            _check(f"lenet_{name}_dW", grad_np(W), wg)
            _check(f"lenet_{name}_dB", grad_np(B), bg)

    def test_lenet5_loss_decreases(self):
        N = 8
        rng = np.random.RandomState(1)
        x_np = rng.randn(N, 1, 28, 28).astype(np.float32) * 0.1
        y_np = (rng.randint(0, 2, (N, 10))).astype(np.float32)

        s_model = Sequential([
            Input((1, 28, 28)),
            Conv2D(6,  1,  (5, 5), activation="relu", zero_padding=2),
            MaxPool2D((2, 2), stride=2),
            Conv2D(16, 6,  (5, 5), activation="relu"),
            MaxPool2D((2, 2), stride=2),
            Flatten(),
            Dense(16 * 5 * 5, 120, activation="relu"),
            Dense(120, 84, activation="relu"),
            Dense(84, 10, activation="sigmoid"),
        ], device="cuda")
        opt = Adam(s_model, lr=0.01)
        losses = []
        for _ in range(6):
            ypred = s_model.forward(x_np)
            tgt   = Tensor(y_np, device="cuda")
            loss  = ((ypred - tgt) ** 2).mean()
            losses.append(float(s2np(loss.value).ravel()[0]))
            s_model.zero_grad()
            seera_backward(loss)
            opt.step()
        self.assertLess(losses[-1], losses[0],
                        f"LeNet-5 loss not decreasing: {losses}")


# ─────────────────────────────────────────────────────────────────────────────
# Architecture 2: VGG-mini (3 conv blocks, each with 2 convs + pool)
# ─────────────────────────────────────────────────────────────────────────────
class TestVGGMini(ArchTestBase):
    """
    VGG-mini on tiny 32×32 input:
      [Conv(3→16,3×3,p1) → relu → Conv(16→16,3×3,p1) → relu → Pool(2×2)] ×2
      → Flatten → Dense(1024, 64) → relu → Dense(64, 2) → sigmoid
    """

    def test_vgg_mini_grad_match(self):
        N = 2
        rng = np.random.RandomState(5)
        x_np = rng.randn(N, 3, 32, 32).astype(np.float32) * 0.1
        y_np = rng.randn(N, 2).astype(np.float32)

        class PTVGGMini(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(3,  16, 3, padding=1)
                self.c2 = nn.Conv2d(16, 16, 3, padding=1)
                self.c3 = nn.Conv2d(16, 32, 3, padding=1)
                self.c4 = nn.Conv2d(32, 32, 3, padding=1)
                self.f1 = nn.Linear(32 * 8 * 8, 64)
                self.f2 = nn.Linear(64, 2)
            def forward(self, x):
                x = F.max_pool2d(F.relu(self.c2(F.relu(self.c1(x)))), 2)
                x = F.max_pool2d(F.relu(self.c4(F.relu(self.c3(x)))), 2)
                x = x.view(x.size(0), -1)
                return torch.sigmoid(self.f2(F.relu(self.f1(x))))

        pt_model = PTVGGMini().to(TORCH_DEVICE)

        c1 = Conv2D(16, 3,  (3,3), activation="relu", zero_padding=1)
        c2 = Conv2D(16, 16, (3,3), activation="relu", zero_padding=1)
        c3 = Conv2D(32, 16, (3,3), activation="relu", zero_padding=1)
        c4 = Conv2D(32, 32, (3,3), activation="relu", zero_padding=1)
        f1 = Dense(32 * 8 * 8, 64, activation="relu")
        f2 = Dense(64, 2, activation="sigmoid")

        for sl, pl in [(c1,pt_model.c1),(c2,pt_model.c2),
                       (c3,pt_model.c3),(c4,pt_model.c4)]:
            self._sync_conv2d(sl, pl)
        for sl, pl in [(f1,pt_model.f1),(f2,pt_model.f2)]:
            self._sync_dense(sl, pl)

        s_model = Sequential([
            Input((3, 32, 32)),
            c1, c2, MaxPool2D((2,2), stride=2),
            c3, c4, MaxPool2D((2,2), stride=2),
            Flatten(), f1, f2,
        ], device="cuda")

        loss_s, loss_p = self._run_arch(s_model, pt_model, x_np, y_np)
        self._loss_close("vgg_mini", loss_s, loss_p)

        for name, sl, pl in [("c1",c1,pt_model.c1),("c2",c2,pt_model.c2),
                              ("c3",c3,pt_model.c3),("c4",c4,pt_model.c4)]:
            W, B = sl.get_weights()
            _check(f"vgg_{name}_dW", grad_np(W), pl.weight.grad)
            _check(f"vgg_{name}_dB", grad_np(B), pl.bias.grad.view(1,-1,1,1))

    def test_vgg_mini_loss_decreases(self):
        N = 4
        rng = np.random.RandomState(6)
        x_np = rng.randn(N, 3, 32, 32).astype(np.float32) * 0.1
        y_np = rng.randn(N, 2).astype(np.float32)
        s_model = Sequential([
            Input((3, 32, 32)),
            Conv2D(16, 3,  (3,3), activation="relu", zero_padding=1),
            Conv2D(16, 16, (3,3), activation="relu", zero_padding=1),
            MaxPool2D((2,2), stride=2),
            Conv2D(32, 16, (3,3), activation="relu", zero_padding=1),
            Conv2D(32, 32, (3,3), activation="relu", zero_padding=1),
            MaxPool2D((2,2), stride=2),
            Flatten(),
            Dense(32*8*8, 64, activation="relu"),
            Dense(64, 2, activation="sigmoid"),
        ], device="cuda")
        opt = Adam(s_model, lr=0.01)
        losses = []
        for _ in range(6):
            ypred = s_model.forward(x_np)
            tgt   = Tensor(y_np, device="cuda")
            loss  = ((ypred - tgt)**2).mean()
            losses.append(float(s2np(loss.value).ravel()[0]))
            s_model.zero_grad(); seera_backward(loss); opt.step()
        self.assertLess(losses[-1], losses[0],
                        f"VGG-mini loss not decreasing: {losses}")


# ─────────────────────────────────────────────────────────────────────────────
# Architecture 3: ResNet-style block (skip connection via addition)
# ─────────────────────────────────────────────────────────────────────────────
class TestResNetBlock(ArchTestBase):
    """
    Manual residual block (no Seera Skip layer — we wire it with raw Tensors):
      conv1(3×3,p1) → relu → conv2(3×3,p1) → add input → relu
    Weight gradients compared to PyTorch equivalent.
    """

    def test_resnet_block_grads(self):
        N, C, H, W = 2, 8, 8, 8
        rng = np.random.RandomState(10)
        x_np  = rng.randn(N, C, H, W).astype(np.float32) * 0.1
        w1_np = rng.randn(C, C, 3, 3).astype(np.float32) * 0.1
        w2_np = rng.randn(C, C, 3, 3).astype(np.float32) * 0.1
        b1_np = np.zeros((1, C, 1, 1), dtype=np.float32)
        b2_np = np.zeros((1, C, 1, 1), dtype=np.float32)

        # Seera — manual residual
        sx  = Tensor(x_np,  is_leaf=True, device="cuda")
        sw1 = Tensor(w1_np, is_leaf=True, device="cuda")
        sw2 = Tensor(w2_np, is_leaf=True, device="cuda")
        sb1 = Tensor(b1_np, is_leaf=True, device="cuda")
        sb2 = Tensor(b2_np, is_leaf=True, device="cuda")
        s1  = (sx.conv2d(sw1, padding=(1,1)) + sb1).relu()
        s2  = (s1.conv2d(sw2, padding=(1,1)) + sb2)
        s_out = (s2 + sx).relu()          # residual add
        seera_backward(s_out.sum())

        # PyTorch
        px  = pt(x_np,  requires_grad=True)
        pw1 = pt(w1_np, requires_grad=True)
        pw2 = pt(w2_np, requires_grad=True)
        pb1 = pt(b1_np, requires_grad=True)
        pb2 = pt(b2_np, requires_grad=True)
        p1  = F.relu(F.conv2d(px, pw1, padding=1) + pb1)
        p2  = F.conv2d(p1, pw2, padding=1) + pb2
        p_out = F.relu(p2 + px)
        p_out.sum().backward()

        _check("res_fwd",  s2np(s_out.value), p_out,   atol=ATOL_FWD)
        _check("res_dX",   grad_np(sx),  px.grad)
        _check("res_dW1",  grad_np(sw1), pw1.grad)
        _check("res_dW2",  grad_np(sw2), pw2.grad)
        _check("res_dB1",  grad_np(sb1), pb1.grad.view(1,-1,1,1))
        _check("res_dB2",  grad_np(sb2), pb2.grad.view(1,-1,1,1))

    def test_resnet_block_loss_decreases(self):
        N, C, H, W = 4, 8, 8, 8
        rng = np.random.RandomState(11)
        x_np  = rng.randn(N, C, H, W).astype(np.float32) * 0.1
        y_np  = rng.randn(N, C, H, W).astype(np.float32)
        w1_np = rng.randn(C, C, 3, 3).astype(np.float32) * 0.05
        w2_np = rng.randn(C, C, 3, 3).astype(np.float32) * 0.05

        sw1 = Tensor(w1_np, is_leaf=True, device="cuda")
        sw2 = Tensor(w2_np, is_leaf=True, device="cuda")
        b1  = Tensor(np.zeros((1,C,1,1), np.float32), is_leaf=True, device="cuda")
        b2  = Tensor(np.zeros((1,C,1,1), np.float32), is_leaf=True, device="cuda")

        lr = 0.01
        losses = []
        for _ in range(8):
            sx   = Tensor(x_np, is_leaf=True, device="cuda")
            s1   = (sx.conv2d(sw1, padding=(1,1)) + b1).relu()
            s2   = (s1.conv2d(sw2, padding=(1,1)) + b2)
            s_out = (s2 + sx).relu()
            tgt  = Tensor(y_np, device="cuda")
            loss = ((s_out - tgt)**2).mean()
            losses.append(float(s2np(loss.value).ravel()[0]))
            # reset grads
            for t in [sw1, sw2, b1, b2]:
                t.node.cp = s2np(t.node.cp) * 0  # zero
                if isinstance(t.value, cuten):
                    t.node.cp = cuten(np.zeros_like(s2np(t.value)))
                else:
                    t.node.cp = np.zeros_like(t.value)
            seera_backward(loss)
            # SGD step
            for t in [sw1, sw2, b1, b2]:
                g = s2np(t.node.cp)
                if isinstance(t.value, cuten):
                    t.value = cuten(s2np(t.value) - lr * g)
                else:
                    t.value -= lr * g
        self.assertLess(losses[-1], losses[0],
                        f"ResNet block loss not decreasing: {losses}")


# ─────────────────────────────────────────────────────────────────────────────
# Architecture 4: U-Net encoder–decoder (with skip via Concatenate)
# ─────────────────────────────────────────────────────────────────────────────
class TestUNetEncoderDecoder(ArchTestBase):
    """
    Tiny U-Net on 1×16×16:
      Encoder: Conv(1→8, 3×3, p1)→relu→Pool(2×2) → (8×8×8)
      Bottleneck: Conv(8→16, 3×3, p1)→relu          → (16×8×8)
      Decoder: ConvT(16→8, 4×4, s2, p1)→relu         → (8×16×16)

    We compare weight gradients only for the three conv layers.
    (Skip connection via Seera Concatenate is tested separately below.)
    """

    def test_unet_enc_dec_grads(self):
        N = 2
        rng = np.random.RandomState(20)
        x_np = rng.randn(N, 1, 16, 16).astype(np.float32) * 0.1
        y_np = rng.randn(N, 8, 16, 16).astype(np.float32)

        # PyTorch reference
        class PTUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Conv2d(1,  8,  3, padding=1)
                self.bot = nn.Conv2d(8,  16, 3, padding=1)
                self.dec = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)
            def forward(self, x):
                e = F.max_pool2d(F.relu(self.enc(x)), 2)
                b = F.relu(self.bot(e))
                return F.relu(self.dec(b))

        pt_model = PTUNet().to(TORCH_DEVICE)

        enc = Conv2D(8,  1,  (3,3), activation="relu", zero_padding=1)
        bot = Conv2D(16, 8,  (3,3), activation="relu", zero_padding=1)
        dec = ConvTranspose2D(8, 16, (4,4), activation="relu",
                              stride=2, zero_padding=1)

        self._sync_conv2d(enc, pt_model.enc)
        self._sync_conv2d(bot, pt_model.bot)
        self._sync_convT(dec, pt_model.dec)

        s_model = Sequential([
            Input((1, 16, 16)),
            enc,
            MaxPool2D((2,2), stride=2),
            bot,
            dec,
        ], device="cuda")

        loss_s, loss_p = self._run_arch(s_model, pt_model, x_np, y_np)
        self._loss_close("unet", loss_s, loss_p)

        for name, sl, pl in [("enc", enc, pt_model.enc),
                              ("bot", bot, pt_model.bot)]:
            W, B = sl.get_weights()
            _check(f"unet_{name}_dW", grad_np(W), pl.weight.grad)
            _check(f"unet_{name}_dB", grad_np(B), pl.bias.grad.view(1,-1,1,1))

        # ConvTranspose
        Wd, Bd = dec.get_weights()
        _check("unet_dec_dW", grad_np(Wd), pt_model.dec.weight.grad, atol=ATOL_GRAD)
        _check("unet_dec_dB", grad_np(Bd), pt_model.dec.bias.grad.view(1,-1,1,1), atol=ATOL_GRAD)

    def test_unet_loss_decreases(self):
        N = 2
        rng = np.random.RandomState(21)
        x_np = rng.randn(N, 1, 16, 16).astype(np.float32) * 0.1
        y_np = rng.randn(N, 8, 16, 16).astype(np.float32)
        s_model = Sequential([
            Input((1, 16, 16)),
            Conv2D(8,  1,  (3,3), activation="relu", zero_padding=1),
            MaxPool2D((2,2), stride=2),
            Conv2D(16, 8,  (3,3), activation="relu", zero_padding=1),
            ConvTranspose2D(8, 16, (4,4), activation="relu", stride=2, zero_padding=1),
        ], device="cuda")
        opt = Adam(s_model, lr=0.01)
        losses = []
        for _ in range(8):
            ypred = s_model.forward(x_np)
            tgt   = Tensor(y_np, device="cuda")
            loss  = ((ypred - tgt)**2).mean()
            losses.append(float(s2np(loss.value).ravel()[0]))
            s_model.zero_grad(); seera_backward(loss); opt.step()
        self.assertLess(losses[-1], losses[0],
                        f"U-Net enc/dec loss not decreasing: {losses}")


# ─────────────────────────────────────────────────────────────────────────────
# Architecture 5: AlexNet-mini
# ─────────────────────────────────────────────────────────────────────────────
class TestAlexNetMini(ArchTestBase):
    """
    AlexNet-mini on 3×32×32:
      Conv(3→16, 5×5, s1, p2) → relu → Pool(3×3, s2) →
      Conv(16→32, 3×3, p1)    → relu → Pool(3×3, s2) →
      Flatten → Dense(800→128) → relu → Dense(128→4) → sigmoid
    """

    def test_alexnet_mini_grads(self):
        N = 4
        rng = np.random.RandomState(30)
        x_np = rng.randn(N, 3, 32, 32).astype(np.float32) * 0.1
        y_np = rng.randn(N, 4).astype(np.float32)

        class PTAlex(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(3,  16, 5, padding=2)
                self.c2 = nn.Conv2d(16, 32, 3, padding=1)
                self.f1 = nn.Linear(32 * 5 * 5, 128)
                self.f2 = nn.Linear(128, 4)
            def forward(self, x):
                x = F.max_pool2d(F.relu(self.c1(x)), 3, stride=2)   # (N,16,15,15)→(N,16,7,7)
                x = F.max_pool2d(F.relu(self.c2(x)), 3, stride=2)   # (N,32,7,7)→(N,32,3,3) ??
                # Clamp to make it deterministic
                x = x.view(N, -1)
                x = F.relu(self.f1(x))
                return torch.sigmoid(self.f2(x))

        pt_model = PTAlex().to(TORCH_DEVICE)

        # figure out flat dim after two pools
        dummy = torch.zeros(1, 3, 32, 32, device=TORCH_DEVICE)
        with torch.no_grad():
            tmp = F.max_pool2d(F.relu(pt_model.c1(dummy)), 3, stride=2)
            tmp = F.max_pool2d(F.relu(pt_model.c2(tmp)),   3, stride=2)
            flat_dim = tmp.view(1, -1).shape[1]

        # rebuild f1 with correct dim
        pt_model.f1 = nn.Linear(flat_dim, 128).to(TORCH_DEVICE)
        pt_model.f2 = nn.Linear(128, 4).to(TORCH_DEVICE)

        c1 = Conv2D(16, 3,  (5,5), activation="relu", zero_padding=2)
        c2 = Conv2D(32, 16, (3,3), activation="relu", zero_padding=1)
        f1 = Dense(flat_dim, 128, activation="relu")
        f2 = Dense(128, 4, activation="sigmoid")

        self._sync_conv2d(c1, pt_model.c1)
        self._sync_conv2d(c2, pt_model.c2)
        self._sync_dense(f1, pt_model.f1)
        self._sync_dense(f2, pt_model.f2)

        s_model = Sequential([
            Input((3, 32, 32)),
            c1, MaxPool2D((3,3), stride=3),
            c2, MaxPool2D((3,3), stride=3),
            Flatten(), f1, f2,
        ], device="cuda")

        loss_s, loss_p = self._run_arch(s_model, pt_model, x_np, y_np)
        # Forward parity
        _check("alex_fwd",
               s2np(s_model.forward(x_np).value),
               pt_model(pt(x_np)), atol=ATOL_FWD)

    def test_alexnet_mini_loss_decreases(self):
        N = 4
        rng = np.random.RandomState(31)
        x_np = rng.randn(N, 3, 32, 32).astype(np.float32) * 0.1
        y_np = rng.randn(N, 4).astype(np.float32)
        s_model = Sequential([
            Input((3, 32, 32)),
            Conv2D(16, 3,  (5,5), activation="relu", zero_padding=2),
            MaxPool2D((3,3), stride=3),
            Conv2D(32, 16, (3,3), activation="relu", zero_padding=1),
            MaxPool2D((3,3), stride=3),
            Flatten(),
            Dense(32*3*3, 128, activation="relu"),
            Dense(128, 4, activation="sigmoid"),
        ], device="cuda")
        opt = Adam(s_model, lr=0.01)
        losses = []
        for _ in range(6):
            ypred = s_model.forward(x_np)
            tgt   = Tensor(y_np, device="cuda")
            loss  = ((ypred - tgt)**2).mean()
            losses.append(float(s2np(loss.value).ravel()[0]))
            s_model.zero_grad(); seera_backward(loss); opt.step()
        self.assertLess(losses[-1], losses[0],
                        f"AlexNet-mini loss not decreasing: {losses}")


# ─────────────────────────────────────────────────────────────────────────────
# Architecture 6: Autoencoder (conv encoder + convT decoder)
# ─────────────────────────────────────────────────────────────────────────────
class TestAutoencoder(ArchTestBase):
    """
    Conv AE on 1×16×16:
      Enc: Conv(1→8,3×3,p1)→relu→Pool(2×2) → (8×8×8)
      Dec: ConvT(8→8,4×4,s2,p1)→relu → Conv(8→1,3×3,p1)→sigmoid
    Loss = MSE vs original input.
    """

    def test_ae_grads(self):
        N = 2
        rng = np.random.RandomState(40)
        x_np = rng.randn(N, 1, 16, 16).astype(np.float32) * 0.5
        y_np = x_np  # reconstruction target

        class PTAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc  = nn.Conv2d(1, 8, 3, padding=1)
                self.dec  = nn.ConvTranspose2d(8, 8, 4, stride=2, padding=1)
                self.out  = nn.Conv2d(8, 1, 3, padding=1)
            def forward(self, x):
                z = F.max_pool2d(F.relu(self.enc(x)), 2)
                d = F.relu(self.dec(z))
                return torch.sigmoid(self.out(d))

        pt_model = PTAE().to(TORCH_DEVICE)

        enc = Conv2D(8, 1,  (3,3), activation="relu", zero_padding=1)
        dec = ConvTranspose2D(8, 8, (4,4), activation="relu", stride=2, zero_padding=1)
        out = Conv2D(1, 8,  (3,3), activation="sigmoid", zero_padding=1)

        self._sync_conv2d(enc, pt_model.enc)
        self._sync_convT(dec, pt_model.dec)
        self._sync_conv2d(out, pt_model.out)

        s_model = Sequential([
            Input((1, 16, 16)),
            enc, MaxPool2D((2,2), stride=2),
            dec, out,
        ], device="cuda")

        loss_s, loss_p = self._run_arch(s_model, pt_model, x_np, y_np)
        self._loss_close("ae", loss_s, loss_p)

        for name, sl, pl in [("enc", enc, pt_model.enc),
                              ("out", out, pt_model.out)]:
            W, B = sl.get_weights()
            _check(f"ae_{name}_dW", grad_np(W), pl.weight.grad)
            _check(f"ae_{name}_dB", grad_np(B), pl.bias.grad.view(1,-1,1,1))
        Wd, Bd = dec.get_weights()
        _check("ae_dec_dW", grad_np(Wd), pt_model.dec.weight.grad, atol=ATOL_GRAD)
        _check("ae_dec_dB", grad_np(Bd), pt_model.dec.bias.grad.view(1,-1,1,1), atol=ATOL_GRAD)

    def test_ae_loss_decreases(self):
        N = 4
        rng = np.random.RandomState(41)
        x_np = rng.randn(N, 1, 16, 16).astype(np.float32) * 0.5
        s_model = Sequential([
            Input((1, 16, 16)),
            Conv2D(8, 1, (3,3), activation="relu", zero_padding=1),
            MaxPool2D((2,2), stride=2),
            ConvTranspose2D(8, 8, (4,4), activation="relu", stride=2, zero_padding=1),
            Conv2D(1, 8, (3,3), activation="sigmoid", zero_padding=1),
        ], device="cuda")
        opt = Adam(s_model, lr=0.01)
        losses = []
        for _ in range(8):
            ypred = s_model.forward(x_np)
            tgt   = Tensor(x_np, device="cuda")
            loss  = ((ypred - tgt)**2).mean()
            losses.append(float(s2np(loss.value).ravel()[0]))
            s_model.zero_grad(); seera_backward(loss); opt.step()
        self.assertLess(losses[-1], losses[0],
                        f"Autoencoder loss not decreasing: {losses}")


# ─────────────────────────────────────────────────────────────────────────────
# Architecture 7: FCN-mini (Fully Convolutional — no dense layers)
# ─────────────────────────────────────────────────────────────────────────────
class TestFCNMini(ArchTestBase):
    """
    Fully convolutional network: input stays spatial throughout.
    In: (N,3,16,16) → Out: (N,2,16,16)  (pixel-wise 2-class)
    Conv(3→8,3,p1)→relu → Conv(8→16,3,p1)→relu → Conv(16→2,1)→sigmoid
    Loss = MSE vs spatial target.
    """

    def test_fcn_grads(self):
        N = 2
        rng = np.random.RandomState(50)
        x_np = rng.randn(N, 3, 16, 16).astype(np.float32) * 0.1
        y_np = rng.randn(N, 2, 16, 16).astype(np.float32)

        class PTFCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(3,  8,  3, padding=1)
                self.c2 = nn.Conv2d(8,  16, 3, padding=1)
                self.c3 = nn.Conv2d(16, 2,  1)
            def forward(self, x):
                x = F.relu(self.c1(x))
                x = F.relu(self.c2(x))
                return torch.sigmoid(self.c3(x))

        pt_model = PTFCN().to(TORCH_DEVICE)

        c1 = Conv2D(8,  3,  (3,3), activation="relu",    zero_padding=1)
        c2 = Conv2D(16, 8,  (3,3), activation="relu",    zero_padding=1)
        c3 = Conv2D(2,  16, (1,1), activation="sigmoid")

        for sl, pl in [(c1,pt_model.c1),(c2,pt_model.c2),(c3,pt_model.c3)]:
            self._sync_conv2d(sl, pl)

        s_model = Sequential([
            Input((3, 16, 16)), c1, c2, c3,
        ], device="cuda")

        loss_s, loss_p = self._run_arch(s_model, pt_model, x_np, y_np)
        self._loss_close("fcn", loss_s, loss_p)

        for name, sl, pl in [("c1",c1,pt_model.c1),("c2",c2,pt_model.c2),
                              ("c3",c3,pt_model.c3)]:
            W, B = sl.get_weights()
            _check(f"fcn_{name}_dW", grad_np(W), pl.weight.grad)
            _check(f"fcn_{name}_dB", grad_np(B), pl.bias.grad.view(1,-1,1,1))

    def test_fcn_loss_decreases(self):
        N = 4
        rng = np.random.RandomState(51)
        x_np = rng.randn(N, 3, 16, 16).astype(np.float32) * 0.1
        y_np = rng.randn(N, 2, 16, 16).astype(np.float32)
        s_model = Sequential([
            Input((3, 16, 16)),
            Conv2D(8,  3,  (3,3), activation="relu",    zero_padding=1),
            Conv2D(16, 8,  (3,3), activation="relu",    zero_padding=1),
            Conv2D(2,  16, (1,1), activation="sigmoid"),
        ], device="cuda")
        opt = Adam(s_model, lr=0.01)
        losses = []
        for _ in range(8):
            ypred = s_model.forward(x_np)
            tgt   = Tensor(y_np, device="cuda")
            loss  = ((ypred - tgt)**2).mean()
            losses.append(float(s2np(loss.value).ravel()[0]))
            s_model.zero_grad(); seera_backward(loss); opt.step()
        self.assertLess(losses[-1], losses[0],
                        f"FCN-mini loss not decreasing: {losses}")


# ════════════════════════════════════════════════════════════════════════════
#  F. NUMERICAL FINITE-DIFFERENCE CHECKS
# ════════════════════════════════════════════════════════════════════════════

class TestFiniteDiff(unittest.TestCase):
    """
    For a tiny input, perturb each element by ±eps and verify
    Seera's analytic gradient matches (f(x+e)-f(x-e))/(2*eps).
    """
    EPS  = 1e-3
    ATOL = 2e-2

    def _fd_conv(self, tag, N, C, H, W, F_out, KH, KW, stride, pad, wrt="x"):
        rng = np.random.RandomState(99)
        x_np = rng.randn(N, C, H, W).astype(np.float32) * 0.1
        w_np = rng.randn(F_out, C, KH, KW).astype(np.float32) * 0.1

        def fwd(xv, wv):
            sx = Tensor(xv.copy(), is_leaf=True, device="cuda")
            sw = Tensor(wv.copy(), is_leaf=True, device="cuda")
            return float(s2np(sx.conv2d(sw,
                stride=(stride, stride),
                padding=(pad, pad)).value).sum())

        if wrt == "x":
            target = x_np
        else:
            target = w_np

        # analytic gradient
        sx  = Tensor(x_np.copy(), is_leaf=True, device="cuda")
        sw  = Tensor(w_np.copy(), is_leaf=True, device="cuda")
        seera_backward(sx.conv2d(sw, stride=(stride,stride), padding=(pad,pad)).sum())
        analytic = grad_np(sx if wrt=="x" else sw).ravel()

        # numerical gradient
        flat = target.ravel().copy()
        num  = np.zeros_like(flat)
        for i in range(flat.size):
            fp = flat.copy(); fp[i] += self.EPS
            fm = flat.copy(); fm[i] -= self.EPS
            xp = fp.reshape(target.shape) if wrt=="x" else x_np
            xm = fm.reshape(target.shape) if wrt=="x" else x_np
            wp = w_np if wrt=="x" else fp.reshape(target.shape)
            wm = w_np if wrt=="x" else fm.reshape(target.shape)
            num[i] = (fwd(xp, wp) - fwd(xm, wm)) / (2 * self.EPS)

        max_diff = float(np.abs(analytic - num).max())
        self.assertLess(max_diff, self.ATOL,
                        f"[{tag}_{wrt}] FD max_diff={max_diff:.5f}")

    def test_fd_conv_dX_s1p0(self): self._fd_conv("s1p0", 1, 1, 4, 4, 2, 3, 3, 1, 0, "x")
    def test_fd_conv_dW_s1p0(self): self._fd_conv("s1p0", 1, 1, 4, 4, 2, 3, 3, 1, 0, "w")
    def test_fd_conv_dX_s1p1(self): self._fd_conv("s1p1", 1, 2, 5, 5, 2, 3, 3, 1, 1, "x")
    def test_fd_conv_dW_s1p1(self): self._fd_conv("s1p1", 1, 2, 5, 5, 2, 3, 3, 1, 1, "w")
    def test_fd_conv_dX_s2p1(self): self._fd_conv("s2p1", 1, 2, 6, 6, 2, 3, 3, 2, 1, "x")
    def test_fd_conv_dW_s2p1(self): self._fd_conv("s2p1", 1, 2, 6, 6, 2, 3, 3, 2, 1, "w")

    def _fd_pool(self, tag, N, C, H, W, KH, KW, stride):
        rng = np.random.RandomState(88)
        x_np = rng.randn(N, C, H, W).astype(np.float32)
        # avoid ties
        x_np += np.arange(x_np.size).reshape(x_np.shape).astype(np.float32) * 0.01

        sx = Tensor(x_np.copy(), is_leaf=True, device="cuda")
        seera_backward(sx.maxpool2d(kernelsize=(KH,KW),
                                    stride=(stride,stride)).sum())
        analytic = grad_np(sx).ravel()

        def fwd(xv):
            s = Tensor(xv.copy(), is_leaf=True, device="cuda")
            return float(s2np(s.maxpool2d(kernelsize=(KH,KW),
                                          stride=(stride,stride)).value).sum())

        flat = x_np.ravel()
        num  = np.zeros_like(flat)
        for i in range(flat.size):
            fp = flat.copy(); fp[i] += self.EPS
            fm = flat.copy(); fm[i] -= self.EPS
            num[i] = (fwd(fp.reshape(x_np.shape)) -
                      fwd(fm.reshape(x_np.shape))) / (2 * self.EPS)

        max_diff = float(np.abs(analytic - num).max())
        self.assertLess(max_diff, self.ATOL,
                        f"[{tag}_pool_dX] FD max_diff={max_diff:.5f}")

    def test_fd_pool_2x2(self):  self._fd_pool("2x2", 1, 2, 6, 6, 2, 2, 2)
    def test_fd_pool_3x3(self):  self._fd_pool("3x3", 1, 2, 9, 9, 3, 3, 3)


# ════════════════════════════════════════════════════════════════════════════
#  G. PATHOLOGICAL / EDGE CASES
# ════════════════════════════════════════════════════════════════════════════

class TestPathological(ConvTestMixin, unittest.TestCase):

    def test_single_pixel_output(self):
        """Conv that maps spatial to 1×1 — all spatial info collapses."""
        self._conv2d("single_px", 2, 3, 5, 5, 4, 5, 5)

    def test_batch_1_conv(self):
        self._conv2d("b1_conv", 1, 4, 8, 8, 8, 3, 3, pad=1)

    def test_batch_1_pool(self):
        self._maxpool("b1_pool", 1, 4, 8, 8, 2, 2, 2)

    def test_batch_1_convT(self):
        self._convT("b1_cT", 1, 4, 4, 4, 8, 3, 3, stride=2, pad=1)

    def test_very_deep_channels(self):
        """128 in → 128 out — tests large channel matmul in backward."""
        self._conv2d("deep128", 2, 128, 4, 4, 128, 3, 3, pad=1)

    def test_single_channel(self):
        self._conv2d("1ch", 4, 1, 8, 8, 1, 3, 3, pad=1)

    def test_kernel_size_1(self):
        """1×1 conv is essentially a channel-wise linear projection."""
        self._conv2d("k1x1", 4, 16, 8, 8, 32, 1, 1)

    def test_all_zeros_input(self):
        """Zero input: activations zero, bias grad must still flow."""
        N, C, H, W, F = 2, 3, 8, 8, 4
        x_np = np.zeros((N, C, H, W), dtype=np.float32)
        w_np = rand(F, C, 3, 3)
        b_np = rand(1, F, 1, 1)
        sx = Tensor(x_np, is_leaf=True, device="cuda")
        sw = Tensor(w_np, is_leaf=True, device="cuda")
        sb = Tensor(b_np, is_leaf=True, device="cuda")
        s_out = sx.conv2d(sw, stride=(1,1), padding=(1,1)) + sb
        seera_backward(s_out.sum())
        # bias grad should == N * OH * OW (all ones from the sum loss)
        bg = grad_np(sb)
        self.assertFalse(np.allclose(bg, 0),
                         "Bias gradient all-zero with zero input!")

    def test_large_stride_produces_1x1(self):
        """stride > spatial that produces 1×1 output."""
        self._conv2d("large_stride", 2, 4, 8, 8, 8, 3, 3, stride=8)

    def test_gradient_nonzero_all_layers(self):
        """End-to-end: no gradient should vanish in a 4-layer conv stack."""
        x_np  = rand(2, 3, 16, 16)
        w_nps = [rand(8,3,3,3), rand(16,8,3,3), rand(8,16,3,3), rand(4,8,1,1)]
        sws   = [Tensor(w, is_leaf=True, device="cuda") for w in w_nps]
        sx    = Tensor(x_np, is_leaf=True, device="cuda")
        s = sx
        for sw in sws[:3]:
            s = s.conv2d(sw, stride=(1,1), padding=(1,1)).relu()
        s = s.conv2d(sws[3]).sigmoid()
        seera_backward(s.sum())
        for i, sw in enumerate(sws):
            g = grad_np(sw)
            self.assertFalse(np.allclose(g, 0, atol=1e-7),
                             f"Gradient vanished in conv layer {i}!")


# ════════════════════════════════════════════════════════════════════════════
#  H. ZERO_GRAD HYGIENE
# ════════════════════════════════════════════════════════════════════════════

class TestZeroGradHygiene(unittest.TestCase):

    def test_conv_zero_grad_between_steps(self):
        """Grads must be exactly zero after zero_grad()."""
        s_model = Sequential([
            Input((3, 8, 8)),
            Conv2D(8, 3, (3,3), activation="relu", zero_padding=1),
            MaxPool2D((2,2), stride=2),
            Flatten(),
            Dense(8*4*4, 4, activation="sigmoid"),
        ], device="cuda")
        rng = np.random.RandomState(0)
        x_np = rng.randn(2, 3, 8, 8).astype(np.float32)
        y_np = rng.randn(2, 4).astype(np.float32)

        ypred = s_model.forward(x_np)
        loss  = ((ypred - Tensor(y_np, device="cuda"))**2).mean()
        seera_backward(loss)
        s_model.zero_grad()

        for layer in s_model.model:
            if hasattr(layer, "get_weights"):
                W, B = layer.get_weights()
                np.testing.assert_allclose(grad_np(W), 0, atol=1e-7,
                    err_msg=f"Weight grad not zero after zero_grad() in {layer}")
                np.testing.assert_allclose(grad_np(B), 0, atol=1e-7,
                    err_msg=f"Bias grad not zero after zero_grad() in {layer}")

    def test_grad_doubles_without_zero_grad(self):
        """Two backward passes without zeroing → grads double."""
        x_np = rand(2, 4, 8, 8)
        w_np = rand(8, 4, 3, 3)
        sw = Tensor(w_np, is_leaf=True, device="cuda")

        sx1 = Tensor(x_np, is_leaf=True, device="cuda")
        seera_backward(sx1.conv2d(sw, stride=(1,1), padding=(1,1)).sum())
        g1 = grad_np(sw).copy()

        sx2 = Tensor(x_np, is_leaf=True, device="cuda")
        seera_backward(sx2.conv2d(sw, stride=(1,1), padding=(1,1)).sum())
        g2 = grad_np(sw)

        np.testing.assert_allclose(g2, 2 * g1, atol=1e-4,
                                   err_msg="Conv grad accumulation failed (expected 2×)")

    def test_convT_zero_grad(self):
        s_model = Sequential([
            Input((4, 4, 4)),
            ConvTranspose2D(8, 4, (4,4), activation="relu", stride=2, zero_padding=1),
            Conv2D(2, 8, (1,1), activation="sigmoid"),
        ], device="cuda")
        rng = np.random.RandomState(2)
        x_np = rng.randn(2, 4, 4, 4).astype(np.float32)
        y_np = rng.randn(2, 2, 8, 8).astype(np.float32)

        ypred = s_model.forward(x_np)
        loss  = ((ypred - Tensor(y_np, device="cuda"))**2).mean()
        seera_backward(loss)
        s_model.zero_grad()

        for layer in s_model.model:
            if hasattr(layer, "get_weights"):
                W, B = layer.get_weights()
                np.testing.assert_allclose(grad_np(W), 0, atol=1e-7,
                    err_msg=f"ConvT weight grad not zero after zero_grad in {layer}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bar = "═" * 72
    print(f"╔{bar}╗")
    print(f"║{'SEERA — EXTREME CONV / POOL / CONVTRANSPOSE2D TEST SUITE':^72}║")
    print(f"╚{bar}╝")
    print(f"  PyTorch : {torch.__version__}  CUDA: {torch.version.cuda}")
    print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    print()
    unittest.main(verbosity=2)