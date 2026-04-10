#!/usr/bin/env python3
"""
RUTHLESS GPU DIAGNOSTIC SUITE
=============================
Tests every operation in the GPU pipeline against NumPy/CPU reference.
Designed to catch:
  - Matrix ordering bugs (row-major vs col-major, transposed args)
  - Silent zeros in forward/backward
  - Shape mismatches between cuTen and numpy
  - Gradient flow issues (zero-grad, stuck weights)
  - Broadcasting bugs
  - Reduction backward bugs
  - Full end-to-end training convergence
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sys, os
import numpy as np

# ── Make framework importable ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cuTen import cuten
from Seera_init import tensor as Tensor, _where, _is_gpu
from Seera_Engine import autograd4nn
from Seera import (
    Dense, Input, Sequential, Flatten, Conv2D, MaxPool2D, Loss, SGD, Adam,
)

np.random.seed(42)
ATOL = 1e-3   # tensor-core precision tolerance (fp16 accumulation)
RTOL = 1e-2

PASS = 0
FAIL = 0
ERRORS = []


def check(name, condition, detail=""):
    global PASS, FAIL, ERRORS
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        msg = f"  ❌ {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        ERRORS.append(f"{name}: {detail}")


def allclose(a, b, atol=ATOL, rtol=RTOL):
    return np.allclose(a, b, atol=atol, rtol=rtol)


def to_np(val):
    """Bring cuten or Tensor value to host numpy."""
    if isinstance(val, cuten):
        return val.to_host_f32()
    if hasattr(val, 'value'):
        v = val.value
        if isinstance(v, cuten):
            return v.to_host_f32()
        return np.array(v)
    return np.array(val)


# ═══════════════════════════════════════════════════════════════
#  SECTION 1: cuTen LOW-LEVEL OPS
# ═══════════════════════════════════════════════════════════════

def test_cuten_roundtrip():
    """Data integrity: numpy → GPU → numpy must be exact."""
    print("\n[1.1] cuTen round-trip transfer")
    for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)]:
        a = np.random.randn(*shape).astype(np.float32)
        g = cuten(a)
        b = g.to_host_f32()
        check(f"roundtrip shape={shape}", np.array_equal(a, b),
              f"max diff={np.max(np.abs(a-b))}" if not np.array_equal(a, b) else "")


def test_cuten_elementwise():
    """Element-wise add, mul, sub, div, pow."""
    print("\n[1.2] cuTen element-wise ops")
    a_np = np.random.randn(4, 5).astype(np.float32)
    b_np = np.random.randn(4, 5).astype(np.float32)
    a_g = cuten(a_np)
    b_g = cuten(b_np)

    # Add
    c_g = a_g + b_g
    check("add", allclose(to_np(c_g), a_np + b_np),
          f"max diff={np.max(np.abs(to_np(c_g) - (a_np + b_np)))}")

    # Mul
    c_g = a_g * b_g
    check("mul", allclose(to_np(c_g), a_np * b_np))

    # Sub
    c_g = a_g - b_g
    check("sub", allclose(to_np(c_g), a_np - b_np))

    # Scalar mul
    c_g = a_g * 3.0
    check("scalar mul", allclose(to_np(c_g), a_np * 3.0))

    # Scalar add
    c_g = a_g + 2.0
    check("scalar add", allclose(to_np(c_g), a_np + 2.0))

    # Pow
    a_pos = np.abs(a_np) + 0.1
    a_pos_g = cuten(a_pos)
    c_g = a_pos_g ** 2.0
    check("pow", allclose(to_np(c_g), a_pos ** 2.0))


def test_cuten_broadcast():
    """Broadcasting: (N,C,H,W) + (1,C,1,1)."""
    print("\n[1.3] cuTen broadcasting")
    a_np = np.random.randn(2, 3, 4, 4).astype(np.float32)
    b_np = np.random.randn(1, 3, 1, 1).astype(np.float32)
    a_g = cuten(a_np)
    b_g = cuten(b_np)

    # broadcast add
    c_g = a_g + b_g
    expected = a_np + b_np
    check("broadcast add (N,C,H,W)+(1,C,1,1)", allclose(to_np(c_g), expected),
          f"max diff={np.max(np.abs(to_np(c_g) - expected))}")

    # broadcast mul
    c_g = a_g * b_g
    expected = a_np * b_np
    check("broadcast mul (N,C,H,W)*(1,C,1,1)", allclose(to_np(c_g), expected))

    # (N, D) + (1, D) — Dense bias broadcast
    a2 = np.random.randn(8, 10).astype(np.float32)
    b2 = np.random.randn(1, 10).astype(np.float32)
    check("broadcast add (N,D)+(1,D)", allclose(
        to_np(cuten(a2) + cuten(b2)), a2 + b2))


def test_cuten_matmul():
    """Matmul: (M,K) @ (K,N) and batched (M,K) @ (Nbatch,K,N)."""
    print("\n[1.4] cuTen matmul")

    # Non-batched
    M, K, N = 8, 16, 6
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C_ref = A @ B
    C_gpu = to_np(cuten(A).matmul(cuten(B)))
    check(f"matmul ({M},{K})@({K},{N})", allclose(C_gpu, C_ref, atol=5e-3),
          f"max diff={np.max(np.abs(C_gpu - C_ref))}")
    check(f"matmul result shape", C_gpu.shape == C_ref.shape,
          f"got {C_gpu.shape}, expected {C_ref.shape}")

    # Check NOT all zeros
    check("matmul result not all zero", np.any(C_gpu != 0))

    # Non-batched larger (stress tensor-core alignment)
    M, K, N = 33, 65, 17
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C_ref = A @ B
    C_gpu = to_np(cuten(A).matmul(cuten(B)))
    check(f"matmul odd ({M},{K})@({K},{N})", allclose(C_gpu, C_ref, atol=0.05),
          f"max diff={np.max(np.abs(C_gpu - C_ref))}")

    # Dense-like: X(N,in) @ W(in,out) — the core Dense layer operation
    N_batch, In, Out = 4, 10, 5
    X = np.random.randn(N_batch, In).astype(np.float32)
    W = np.random.randn(In, Out).astype(np.float32)
    ref = X @ W
    gpu_result = to_np(cuten(X).matmul(cuten(W)))
    check(f"dense-like matmul ({N_batch},{In})@({In},{Out})", allclose(gpu_result, ref, atol=5e-3),
          f"max diff={np.max(np.abs(gpu_result - ref))}")
    check("dense matmul not zero", np.any(gpu_result != 0))


def test_cuten_transpose():
    """Transpose: 2D and 3D."""
    print("\n[1.5] cuTen transpose")
    A = np.random.randn(7, 11).astype(np.float32)
    AT = to_np(cuten(A).T)
    check("2D transpose shape", AT.shape == (11, 7))
    check("2D transpose values", allclose(AT, A.T))

    B = np.random.randn(3, 5, 7).astype(np.float32)
    BT = to_np(cuten(B).T)
    expected = B.transpose(0, 2, 1)
    check("3D transpose shape", BT.shape == (3, 7, 5))
    check("3D transpose values", allclose(BT, expected))


def test_cuten_activations():
    """All unary activations vs numpy reference."""
    print("\n[1.6] cuTen activations")
    x = np.random.randn(4, 8).astype(np.float32)
    xg = cuten(x)

    check("relu", allclose(to_np(xg.relu()), np.maximum(x, 0)))
    check("sigmoid", allclose(to_np(xg.sigmoid()), 1 / (1 + np.exp(-x)), atol=5e-4))
    check("tanh", allclose(to_np(xg.tanh()), np.tanh(x)))

    x_pos = (np.abs(x) + 0.01).astype(np.float32)
    xpg = cuten(x_pos)
    check("log", allclose(to_np(xpg.log()), np.log(x_pos), atol=5e-4))
    check("exp", allclose(to_np(xg.exp()), np.exp(x), atol=5e-3))
    check("sqrt", allclose(to_np(xpg.sqrt()), np.sqrt(x_pos), atol=5e-4))
    check("abs", allclose(to_np(xg.abs()), np.abs(x)))


def test_cuten_softmax():
    """Softmax along last dim."""
    print("\n[1.7] cuTen softmax")
    x = np.random.randn(4, 10).astype(np.float32)
    ref = np.exp(x - np.max(x, axis=-1, keepdims=True))
    ref = ref / ref.sum(axis=-1, keepdims=True)
    gpu = to_np(cuten(x).softmax())
    check("softmax values", allclose(gpu, ref, atol=5e-4),
          f"max diff={np.max(np.abs(gpu - ref))}")
    check("softmax sums to 1", allclose(gpu.sum(axis=-1), np.ones(4)))


def test_cuten_reductions():
    """Sum, mean, max, min along dims."""
    print("\n[1.8] cuTen reductions")
    x = np.random.randn(3, 4, 5).astype(np.float32)
    xg = cuten(x)

    for dim in range(3):
        ref_sum = x.sum(axis=dim)
        gpu_sum = to_np(xg.sum(dim=dim))
        check(f"sum dim={dim} shape", gpu_sum.shape == ref_sum.shape,
              f"got {gpu_sum.shape}, expected {ref_sum.shape}")
        check(f"sum dim={dim} values", allclose(gpu_sum, ref_sum, atol=5e-3))

        ref_mean = x.mean(axis=dim)
        gpu_mean = to_np(xg.mean(dim=dim))
        check(f"mean dim={dim}", allclose(gpu_mean, ref_mean, atol=5e-3))


def test_cuten_conv2d():
    """Conv2D forward vs numpy."""
    print("\n[1.9] cuTen conv2d")
    from scipy.signal import correlate2d

    N, C, H, W = 2, 3, 8, 8
    F, KH, KW = 4, 3, 3
    x = np.random.randn(N, C, H, W).astype(np.float32)
    w = np.random.randn(F, C, KH, KW).astype(np.float32)

    gpu_out = to_np(cuten(x).conv2d(cuten(w), strideh=1, stridew=1, padh=0, padw=0))
    # Reference using scipy
    OH = H - KH + 1
    OW = W - KW + 1
    ref = np.zeros((N, F, OH, OW), dtype=np.float32)
    for n in range(N):
        for f in range(F):
            for c in range(C):
                ref[n, f] += correlate2d(x[n, c], w[f, c], mode='valid')
    check("conv2d shape", gpu_out.shape == ref.shape)
    check("conv2d values", allclose(gpu_out, ref, atol=0.05),
          f"max diff={np.max(np.abs(gpu_out - ref))}")
    check("conv2d not zero", np.any(gpu_out != 0))


def test_cuten_flatten():
    """Flatten preserves batch dim."""
    print("\n[1.10] cuTen flatten")
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    xg = cuten(x)
    flat = xg.flatten()
    check("flatten shape", flat.shape == (2, 60),
          f"got {flat.shape}")
    check("flatten values match", allclose(to_np(flat), x.reshape(2, -1)))


# ═══════════════════════════════════════════════════════════════
#  SECTION 2: TENSOR AUTOGRAD — FORWARD + BACKWARD
# ═══════════════════════════════════════════════════════════════

def test_tensor_matmul_grad():
    """GPU Tensor matmul forward + backward gradient check."""
    print("\n[2.1] Tensor matmul fwd+bwd (GPU)")
    M, K, N = 4, 8, 5
    A_np = np.random.randn(M, K).astype(np.float32) * 0.1
    B_np = np.random.randn(K, N).astype(np.float32) * 0.1

    # GPU path
    A_gpu = Tensor(A_np, is_leaf=True, device="cuda")
    B_gpu = Tensor(B_np, is_leaf=True, device="cuda")
    C_gpu = A_gpu.matmul(B_gpu)
    loss_gpu = C_gpu.sum()
    autograd4nn(loss_gpu)

    # CPU reference
    A_cpu = Tensor(A_np.copy(), is_leaf=True)
    B_cpu = Tensor(B_np.copy(), is_leaf=True)
    C_cpu = A_cpu.matmul(B_cpu)
    loss_cpu = C_cpu.sum()
    autograd4nn(loss_cpu)

    # Compare forward
    fwd_gpu = to_np(C_gpu)
    fwd_cpu = C_cpu.value
    check("matmul fwd GPU≈CPU", allclose(fwd_gpu, fwd_cpu, atol=0.01),
          f"max diff={np.max(np.abs(fwd_gpu - fwd_cpu))}")

    # Compare gradients
    dA_gpu = to_np(A_gpu.node.cp)
    dA_cpu = A_cpu.node.cp
    dB_gpu = to_np(B_gpu.node.cp)
    dB_cpu = B_cpu.node.cp

    check("dA GPU≈CPU", allclose(dA_gpu, dA_cpu, atol=0.01),
          f"max diff={np.max(np.abs(dA_gpu - dA_cpu))}")
    check("dB GPU≈CPU", allclose(dB_gpu, dB_cpu, atol=0.01),
          f"max diff={np.max(np.abs(dB_gpu - dB_cpu))}")
    check("dA not zero", np.any(np.abs(dA_gpu) > 1e-6))
    check("dB not zero", np.any(np.abs(dB_gpu) > 1e-6))


def test_tensor_dense_like_grad():
    """Simulate Dense: X@W + b — gradients on GPU vs CPU."""
    print("\n[2.2] Dense-like X@W+b fwd+bwd")
    N_batch, In, Out = 4, 10, 5
    X_np = np.random.randn(N_batch, In).astype(np.float32) * 0.1
    W_np = np.random.randn(In, Out).astype(np.float32) * 0.1
    b_np = np.random.randn(1, Out).astype(np.float32) * 0.01

    # GPU
    X_g = Tensor(X_np, is_leaf=True, device="cuda")
    W_g = Tensor(W_np, is_leaf=True, device="cuda")
    b_g = Tensor(b_np, is_leaf=True, device="cuda")
    z_g = X_g.matmul(W_g) + b_g
    loss_g = z_g.sum()
    autograd4nn(loss_g)

    # CPU
    X_c = Tensor(X_np.copy(), is_leaf=True)
    W_c = Tensor(W_np.copy(), is_leaf=True)
    b_c = Tensor(b_np.copy(), is_leaf=True)
    z_c = X_c.matmul(W_c) + b_c
    loss_c = z_c.sum()
    autograd4nn(loss_c)

    check("dense fwd GPU≈CPU", allclose(to_np(z_g), z_c.value, atol=0.01))

    dW_g = to_np(W_g.node.cp)
    dW_c = W_c.node.cp
    db_g = to_np(b_g.node.cp)
    db_c = b_c.node.cp

    check("dW GPU≈CPU", allclose(dW_g, dW_c, atol=0.02),
          f"max diff={np.max(np.abs(dW_g - dW_c))}")
    check("db GPU≈CPU", allclose(db_g, db_c, atol=0.02),
          f"max diff={np.max(np.abs(db_g - db_c))}")
    check("dW not zero", np.any(np.abs(dW_g) > 1e-6))
    check("db not zero", np.any(np.abs(db_g) > 1e-6))


def test_tensor_relu_grad():
    """ReLU backward on GPU."""
    print("\n[2.3] Tensor ReLU fwd+bwd")
    x_np = np.random.randn(4, 8).astype(np.float32)

    # GPU
    x_g = Tensor(x_np, is_leaf=True, device="cuda")
    y_g = x_g.relu()
    loss_g = y_g.sum()
    autograd4nn(loss_g)

    # CPU
    x_c = Tensor(x_np.copy(), is_leaf=True)
    y_c = x_c.relu()
    loss_c = y_c.sum()
    autograd4nn(loss_c)

    check("relu fwd GPU≈CPU", allclose(to_np(y_g), y_c.value))
    dx_g = to_np(x_g.node.cp)
    dx_c = x_c.node.cp
    check("relu grad GPU≈CPU", allclose(dx_g, dx_c),
          f"max diff={np.max(np.abs(dx_g - dx_c))}")
    check("relu grad not zero", np.any(np.abs(dx_g) > 0))


def test_tensor_sigmoid_grad():
    """Sigmoid backward on GPU."""
    print("\n[2.4] Tensor Sigmoid fwd+bwd")
    x_np = np.random.randn(4, 8).astype(np.float32) * 0.5

    x_g = Tensor(x_np, is_leaf=True, device="cuda")
    y_g = x_g.sigmoid()
    loss_g = y_g.sum()
    autograd4nn(loss_g)

    x_c = Tensor(x_np.copy(), is_leaf=True)
    y_c = x_c.sigmoid()
    loss_c = y_c.sum()
    autograd4nn(loss_c)

    check("sigmoid fwd GPU≈CPU", allclose(to_np(y_g), y_c.value, atol=5e-4))
    check("sigmoid grad GPU≈CPU", allclose(to_np(x_g.node.cp), x_c.node.cp, atol=5e-3))
    check("sigmoid grad not zero", np.any(np.abs(to_np(x_g.node.cp)) > 1e-6))


def test_tensor_softmax_grad():
    """Softmax VJP backward on GPU."""
    print("\n[2.5] Tensor Softmax fwd+bwd")
    x_np = np.random.randn(4, 10).astype(np.float32)

    x_g = Tensor(x_np, is_leaf=True, device="cuda")
    y_g = x_g.softmax()
    loss_g = y_g.sum()
    autograd4nn(loss_g)

    x_c = Tensor(x_np.copy(), is_leaf=True)
    y_c = x_c.softmax()
    loss_c = y_c.sum()
    autograd4nn(loss_c)

    check("softmax fwd GPU≈CPU", allclose(to_np(y_g), y_c.value, atol=5e-3))
    dx_g = to_np(x_g.node.cp)
    dx_c = x_c.node.cp
    check("softmax grad GPU≈CPU", allclose(dx_g, dx_c, atol=5e-3),
          f"max diff={np.max(np.abs(dx_g - dx_c))}")

    # Softmax grad for sum-of-outputs should be ~0 (softmax outputs sum=1 always)
    # So d/dx_i (sum softmax) ≈ 0 — this is mathematically correct
    # But we still want to check it's close to CPU


def test_tensor_reduction_grad():
    """Sum, mean backward on GPU."""
    print("\n[2.6] Tensor reduction backward")
    x_np = np.random.randn(4, 6).astype(np.float32)

    # sum(axis=1) backward
    x_g = Tensor(x_np, is_leaf=True, device="cuda")
    s_g = x_g.sum(axis=1)
    total = s_g.sum()
    autograd4nn(total)

    x_c = Tensor(x_np.copy(), is_leaf=True)
    s_c = x_c.sum(axis=1)
    total_c = s_c.sum()
    autograd4nn(total_c)

    check("sum grad GPU≈CPU", allclose(to_np(x_g.node.cp), x_c.node.cp, atol=5e-3),
          f"GPU grad:\n{to_np(x_g.node.cp)}\nCPU grad:\n{x_c.node.cp}")
    check("sum grad not zero", np.any(np.abs(to_np(x_g.node.cp)) > 1e-6))

    # mean() backward (full reduce)
    x_g2 = Tensor(x_np, is_leaf=True, device="cuda")
    m_g = x_g2.mean()
    autograd4nn(m_g)

    x_c2 = Tensor(x_np.copy(), is_leaf=True)
    m_c = x_c2.mean()
    autograd4nn(m_c)

    check("mean grad GPU≈CPU", allclose(to_np(x_g2.node.cp), x_c2.node.cp, atol=5e-3),
          f"max diff={np.max(np.abs(to_np(x_g2.node.cp) - x_c2.node.cp))}")


def test_tensor_pow_grad():
    """Power backward on GPU."""
    print("\n[2.7] Tensor pow fwd+bwd")
    x_np = np.abs(np.random.randn(4, 5).astype(np.float32)) + 0.1

    x_g = Tensor(x_np, is_leaf=True, device="cuda")
    y_g = x_g ** 2.0
    loss_g = y_g.sum()
    autograd4nn(loss_g)

    x_c = Tensor(x_np.copy(), is_leaf=True)
    y_c = x_c ** 2.0
    loss_c = y_c.sum()
    autograd4nn(loss_c)

    check("pow fwd GPU≈CPU", allclose(to_np(y_g), y_c.value))
    check("pow grad GPU≈CPU", allclose(to_np(x_g.node.cp), x_c.node.cp, atol=0.01),
          f"max diff={np.max(np.abs(to_np(x_g.node.cp) - x_c.node.cp))}")
    check("pow grad not zero", np.any(np.abs(to_np(x_g.node.cp)) > 1e-4))


# ═══════════════════════════════════════════════════════════════
#  SECTION 3: LOSS FUNCTIONS — END-TO-END GRADIENT FLOW
# ═══════════════════════════════════════════════════════════════

def test_mse_loss_grad():
    """MSE loss: (pred - target)^2 gradient flow through all ops."""
    print("\n[3.1] MSE loss gradient flow")
    y_np = np.random.randn(4, 3).astype(np.float32) * 0.5
    t_np = np.random.randn(4, 3).astype(np.float32) * 0.5

    # GPU
    y_g = Tensor(y_np, is_leaf=True, device="cuda")
    t_g = Tensor(t_np, device="cuda")
    diff = y_g - t_g
    sq = diff ** 2.0
    loss_g = sq.mean()
    autograd4nn(loss_g)

    # CPU
    y_c = Tensor(y_np.copy(), is_leaf=True)
    t_c = Tensor(t_np.copy())
    diff_c = y_c - t_c
    sq_c = diff_c ** 2.0
    loss_c = sq_c.mean()
    autograd4nn(loss_c)

    loss_val_g = float(to_np(loss_g).ravel()[0])
    loss_val_c = float(loss_c.value.ravel()[0])
    check("MSE loss value GPU≈CPU", abs(loss_val_g - loss_val_c) < 0.01,
          f"GPU={loss_val_g}, CPU={loss_val_c}")
    check("MSE loss not zero", loss_val_g > 1e-6, f"loss={loss_val_g}")

    dy_g = to_np(y_g.node.cp)
    dy_c = y_c.node.cp
    check("MSE grad GPU≈CPU", allclose(dy_g, dy_c, atol=0.02),
          f"max diff={np.max(np.abs(dy_g - dy_c))}")
    check("MSE grad not zero", np.any(np.abs(dy_g) > 1e-6))


def test_cce_loss_grad():
    """Categorical Cross-Entropy: softmax → log → loss."""
    print("\n[3.2] CCE loss gradient flow")
    logits_np = np.random.randn(4, 5).astype(np.float32)
    labels_np = np.zeros((4, 5), dtype=np.float32)
    labels_np[np.arange(4), np.random.randint(0, 5, 4)] = 1.0

    eps = 1e-15

    # GPU
    logits_g = Tensor(logits_np, is_leaf=True, device="cuda")
    labels_g = Tensor(labels_np, device="cuda")
    sm_g = logits_g.softmax()
    per_sample = (-labels_g * (sm_g + eps).log()).sum(axis=-1)
    loss_g = per_sample.mean()
    autograd4nn(loss_g)

    # CPU
    logits_c = Tensor(logits_np.copy(), is_leaf=True)
    labels_c = Tensor(labels_np.copy())
    sm_c = logits_c.softmax()
    per_sample_c = (-labels_c * (sm_c + eps).log()).sum(axis=-1)
    loss_c = per_sample_c.mean()
    autograd4nn(loss_c)

    loss_val_g = float(to_np(loss_g).ravel()[0])
    loss_val_c = float(loss_c.value.ravel()[0])
    check("CCE loss value GPU≈CPU", abs(loss_val_g - loss_val_c) < 0.1,
          f"GPU={loss_val_g}, CPU={loss_val_c}")
    check("CCE loss not zero", loss_val_g > 0.01, f"loss={loss_val_g}")

    dlogits_g = to_np(logits_g.node.cp)
    dlogits_c = logits_c.node.cp
    check("CCE grad not zero", np.any(np.abs(dlogits_g) > 1e-6),
          f"grad max={np.max(np.abs(dlogits_g))}")
    check("CCE grad GPU≈CPU", allclose(dlogits_g, dlogits_c, atol=0.05),
          f"max diff={np.max(np.abs(dlogits_g - dlogits_c))}")


# ═══════════════════════════════════════════════════════════════
#  SECTION 4: FULL DENSE NETWORK — WEIGHT UPDATES
# ═══════════════════════════════════════════════════════════════

def test_dense_single_step():
    """Single Dense forward → loss → backward → check weight grad nonzero."""
    print("\n[4.1] Dense single step weight grad check")
    model_g = Sequential([
        Input((4,)),
        Dense(4, 8, activation="relu"),
        Dense(8, 3, activation="sigmoid"),
    ], device="cuda")

    X = np.random.randn(2, 4).astype(np.float32) * 0.5
    Y = np.abs(np.random.randn(2, 3).astype(np.float32)) * 0.5

    X_t = Tensor(X, is_leaf=True, device="cuda")
    Y_t = Tensor(Y, device="cuda")

    out = model_g.forward(X_t)
    loss = Loss()
    l = loss.mse(out, Y_t)

    # Check forward not zero
    out_np = to_np(out)
    check("Dense fwd not zero", np.any(out_np != 0), f"out={out_np}")

    loss_val = float(to_np(l).ravel()[0])
    check("Dense loss not zero", loss_val > 1e-8, f"loss={loss_val}")

    model_g.zero_grad()
    autograd4nn(l)

    # Check gradients on leaf weights
    for i, layer in enumerate(model_g.model):
        if hasattr(layer, 'weights') and isinstance(layer.weights, Tensor):
            w_grad = to_np(layer.weights.node.cp)
            b_grad = to_np(layer.bais.node.cp)
            w_grad_max = np.max(np.abs(w_grad))
            b_grad_max = np.max(np.abs(b_grad))
            check(f" layer {i} weight grad not zero", w_grad_max > 1e-10,
                  f"max |grad|={w_grad_max}")
            check(f" layer {i} bias grad not zero", b_grad_max > 1e-10,
                  f"max |grad|={b_grad_max}")


def test_dense_weight_actually_changes():
    """Verify SGD actually modifies weights (not stuck)."""
    print("\n[4.2] Dense weight update check (SGD)")
    model = Sequential([
        Input((4,)),
        Dense(4, 3, activation="sigmoid"),
    ], device="cuda")

    # Snapshot weights before
    w_before = to_np(model.model[1].weights.value).copy()
    b_before = to_np(model.model[1].bais.value).copy()

    X = np.random.randn(2, 4).astype(np.float32)
    Y = np.random.randn(2, 3).astype(np.float32) * 0.5

    X_t = Tensor(X, is_leaf=True, device="cuda")
    Y_t = Tensor(Y, device="cuda")

    opt = SGD(model, lr=0.1)
    out = model.forward(X_t)
    loss = Loss().mse(out, Y_t)
    model.zero_grad()
    autograd4nn(loss)
    opt.step()

    w_after = to_np(model.model[1].weights.value).copy()
    b_after = to_np(model.model[1].bais.value).copy()

    w_change = np.max(np.abs(w_after - w_before))
    b_change = np.max(np.abs(b_after - b_before))

    check("weights changed after SGD step", w_change > 1e-8,
          f"max weight change={w_change}")
    check("bias changed after SGD step", b_change > 1e-8,
          f"max bias change={b_change}")


def test_dense_training_convergence():
    """Train a tiny Dense net on XOR-like — loss must decrease."""
    print("\n[4.3] Dense training convergence (XOR-like)")
    np.random.seed(0)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    model = Sequential([
        Input((2,)),
        Dense(2, 16, activation="relu"),
        Dense(16, 1, activation="sigmoid"),
    ], device="cuda")

    opt = SGD(model, lr=0.5)
    loss_fn = Loss()

    losses = []
    for epoch in range(200):
        X_t = Tensor(X, is_leaf=True, device="cuda")
        Y_t = Tensor(Y, device="cuda")
        out = model.forward(X_t)
        l = loss_fn.mse(out, Y_t)
        loss_val = float(to_np(l).ravel()[0])
        losses.append(loss_val)
        model.zero_grad()
        autograd4nn(l)
        opt.step()

    check("loss decreased over training", losses[-1] < losses[0],
          f"start={losses[0]:.6f}, end={losses[-1]:.6f}")
    check("loss not stuck at zero", losses[0] > 1e-6,
          f"initial loss={losses[0]}")
    check("final loss reasonable", losses[-1] < 0.3,
          f"final loss={losses[-1]:.6f}")

    # Print loss trajectory
    print(f"    Loss trajectory: start={losses[0]:.6f}, "
          f"mid={losses[100]:.6f}, end={losses[-1]:.6f}")


def test_dense_gpu_vs_cpu_training():
    """1 epoch on GPU vs CPU — losses should be close."""
    print("\n[4.4] GPU vs CPU Dense training comparison")
    np.random.seed(42)
    X = np.random.randn(8, 4).astype(np.float32) * 0.3
    Y = np.random.randn(8, 2).astype(np.float32) * 0.3

    # Fixed init
    W1 = np.random.randn(4, 8).astype(np.float32) * 0.1
    b1 = np.zeros((1, 8), dtype=np.float32)
    W2 = np.random.randn(8, 2).astype(np.float32) * 0.1
    b2 = np.zeros((1, 2), dtype=np.float32)

    # GPU model
    model_g = Sequential([
        Input((4,)),
        Dense(4, 8, activation="relu"),
        Dense(8, 2, activation="sigmoid"),
    ], device="cuda")
    model_g.model[1].set_weights(
        Tensor(W1.copy(), is_leaf=True, device="cuda"),
        Tensor(b1.copy(), is_leaf=True, device="cuda"),
    )
    model_g.model[2].set_weights(
        Tensor(W2.copy(), is_leaf=True, device="cuda"),
        Tensor(b2.copy(), is_leaf=True, device="cuda"),
    )

    # CPU model
    model_c = Sequential([
        Input((4,)),
        Dense(4, 8, activation="relu"),
        Dense(8, 2, activation="sigmoid"),
    ])
    model_c.model[1].set_weights(W1.copy(), b1.copy())
    model_c.model[2].set_weights(W2.copy(), b2.copy())

    loss_fn = Loss()
    opt_g = SGD(model_g, lr=0.01)
    opt_c = SGD(model_c, lr=0.01)

    # 1 forward+backward
    X_g = Tensor(X.copy(), is_leaf=True, device="cuda")
    Y_g = Tensor(Y.copy(), device="cuda")
    out_g = model_g.forward(X_g)
    l_g = loss_fn.mse(out_g, Y_g)

    X_c = Tensor(X.copy(), is_leaf=True)
    Y_c = Tensor(Y.copy())
    out_c = model_c.forward(X_c)
    l_c = loss_fn.mse(out_c, Y_c)

    loss_g_val = float(to_np(l_g).ravel()[0])
    loss_c_val = float(l_c.value.ravel()[0])

    check("GPU vs CPU loss match", abs(loss_g_val - loss_c_val) < 0.05,
          f"GPU={loss_g_val:.6f}, CPU={loss_c_val:.6f}")

    # Backward
    model_g.zero_grad()
    autograd4nn(l_g)
    model_c.zero_grad()
    autograd4nn(l_c)

    # Compare weight gradients
    dW1_g = to_np(model_g.model[1].weights.node.cp)
    dW1_c = model_c.model[1].weights.node.cp
    check("layer1 dW GPU≈CPU", allclose(dW1_g, dW1_c, atol=0.05),
          f"max diff={np.max(np.abs(dW1_g - dW1_c))}")

    dW2_g = to_np(model_g.model[2].weights.node.cp)
    dW2_c = model_c.model[2].weights.node.cp
    check("layer2 dW GPU≈CPU", allclose(dW2_g, dW2_c, atol=0.05),
          f"max diff={np.max(np.abs(dW2_g - dW2_c))}")


# ═══════════════════════════════════════════════════════════════
#  SECTION 5: CLASSIFICATION CONVERGENCE TEST
# ═══════════════════════════════════════════════════════════════

def test_classification_convergence():
    """Train softmax classifier on easy linear data — must converge."""
    print("\n[5.1] Classification convergence (softmax + CCE)")
    np.random.seed(42)
    N_samples = 100
    # 3-class data
    X = np.random.randn(N_samples, 4).astype(np.float32) * 0.5
    labels = np.random.randint(0, 3, N_samples)
    Y = np.zeros((N_samples, 3), dtype=np.float32)
    Y[np.arange(N_samples), labels] = 1.0
    # Make linearly separable
    for i in range(N_samples):
        X[i, labels[i]] += 2.0

    model = Sequential([
        Input((4,)),
        Dense(4, 8, activation="relu"),
        Dense(8, 3, activation="softmax"),
    ], device="cuda")

    opt = Adam(model, lr=0.01)
    loss_fn = Loss()
    eps = 1e-15

    losses = []
    for epoch in range(100):
        X_t = Tensor(X, is_leaf=True, device="cuda")
        Y_t = Tensor(Y, device="cuda")
        out = model.forward(X_t)
        l = loss_fn.categorical_cross_entropy(out, Y_t, epsilon=eps)
        lv = float(to_np(l).ravel()[0])
        losses.append(lv)
        model.zero_grad()
        autograd4nn(l)
        opt.step()

    check("classification loss decreased", losses[-1] < losses[0],
          f"start={losses[0]:.4f}, end={losses[-1]:.4f}")
    check("classification loss < 1.0", losses[-1] < 1.0,
          f"final={losses[-1]:.4f}")
    print(f"    Loss: start={losses[0]:.4f}, mid={losses[50]:.4f}, end={losses[-1]:.4f}")


# ═══════════════════════════════════════════════════════════════
#  SECTION 6: CONV2D GRADIENT FLOW
# ═══════════════════════════════════════════════════════════════

def test_conv2d_grad():
    """Conv2D forward+backward GPU vs CPU."""
    print("\n[6.1] Conv2D gradient flow")
    N, C, H, W = 2, 1, 8, 8
    F, KH, KW = 2, 3, 3
    x_np = np.random.randn(N, C, H, W).astype(np.float32) * 0.3
    w_np = np.random.randn(F, C, KH, KW).astype(np.float32) * 0.3

    # GPU
    x_g = Tensor(x_np, is_leaf=True, device="cuda")
    w_g = Tensor(w_np, is_leaf=True, device="cuda")
    y_g = x_g.conv2d(w_g)
    loss_g = y_g.sum()
    autograd4nn(loss_g)

    # CPU
    x_c = Tensor(x_np.copy(), is_leaf=True)
    w_c = Tensor(w_np.copy(), is_leaf=True)
    y_c = x_c.conv2d(w_c)
    loss_c = y_c.sum()
    autograd4nn(loss_c)

    check("conv2d fwd GPU≈CPU", allclose(to_np(y_g), y_c.value, atol=0.05))
    dw_g = to_np(w_g.node.cp)
    dw_c = w_c.node.cp
    check("conv2d dW GPU≈CPU", allclose(dw_g, dw_c, atol=0.1),
          f"max diff={np.max(np.abs(dw_g - dw_c))}")
    check("conv2d dW not zero", np.any(np.abs(dw_g) > 1e-6))

    dx_g = to_np(x_g.node.cp)
    dx_c = x_c.node.cp
    check("conv2d dX GPU≈CPU", allclose(dx_g, dx_c, atol=0.1),
          f"max diff={np.max(np.abs(dx_g - dx_c))}")


# ═══════════════════════════════════════════════════════════════
#  SECTION 7: FLATTEN GRADIENT FLOW
# ═══════════════════════════════════════════════════════════════

def test_flatten_grad():
    """Flatten → Dense backward gradient flow on GPU."""
    print("\n[7.1] Flatten gradient flow")
    x_np = np.random.randn(2, 3, 4, 4).astype(np.float32) * 0.3

    # GPU
    x_g = Tensor(x_np, is_leaf=True, device="cuda")
    flat_g = x_g.flatten()
    # Matmul with a weight to create grad flow
    W_np = np.random.randn(48, 5).astype(np.float32) * 0.1
    W_g = Tensor(W_np, is_leaf=True, device="cuda")
    out_g = flat_g.matmul(W_g)
    loss_g = out_g.sum()
    autograd4nn(loss_g)

    # CPU
    x_c = Tensor(x_np.copy(), is_leaf=True)
    flat_c = x_c.flatten()
    W_c = Tensor(W_np.copy(), is_leaf=True)
    out_c = flat_c.matmul(W_c)
    loss_c = out_c.sum()
    autograd4nn(loss_c)

    dx_g = to_np(x_g.node.cp)
    dx_c = x_c.node.cp
    check("flatten grad shape", dx_g.shape == x_np.shape,
          f"got {dx_g.shape}")
    check("flatten grad GPU≈CPU", allclose(dx_g, dx_c, atol=0.05),
          f"max diff={np.max(np.abs(dx_g - dx_c))}")
    check("flatten grad not zero", np.any(np.abs(dx_g) > 1e-6))


# ═══════════════════════════════════════════════════════════════
#  SECTION 8: ADAM OPTIMIZER GPU 
# ═══════════════════════════════════════════════════════════════

def test_adam_weight_update():
    """Adam optimizer: weights must change after step."""
    print("\n[8.1] Adam weight update check")
    model = Sequential([
        Input((4,)),
        Dense(4, 3, activation="sigmoid"),
    ], device="cuda")

    w_before = to_np(model.model[1].weights.value).copy()

    X = np.random.randn(4, 4).astype(np.float32)
    Y = np.random.randn(4, 3).astype(np.float32) * 0.5

    opt = Adam(model, lr=0.01)
    X_t = Tensor(X, is_leaf=True, device="cuda")
    Y_t = Tensor(Y, device="cuda")
    out = model.forward(X_t)
    loss = Loss().mse(out, Y_t)
    model.zero_grad()
    autograd4nn(loss)
    opt.step()

    w_after = to_np(model.model[1].weights.value).copy()
    change = np.max(np.abs(w_after - w_before))
    check("Adam updated weights", change > 1e-6, f"max change={change}")


# ═══════════════════════════════════════════════════════════════
#  SECTION 9: MAXPOOL GRADIENT FLOW
# ═══════════════════════════════════════════════════════════════

def test_maxpool_grad():
    """MaxPool2D gradient flow."""
    print("\n[9.1] MaxPool2D gradient flow")
    x_np = np.random.randn(1, 1, 4, 4).astype(np.float32)

    # GPU
    x_g = Tensor(x_np, is_leaf=True, device="cuda")
    p_g = x_g.maxpool2d(kernelsize=(2, 2), stride=2, padding=0)
    loss_g = p_g.sum()
    autograd4nn(loss_g)

    # CPU
    x_c = Tensor(x_np.copy(), is_leaf=True)
    p_c = x_c.maxpool2d(kernelsize=(2, 2), stride=2, padding=0)
    loss_c = p_c.sum()
    autograd4nn(loss_c)

    check("maxpool fwd GPU≈CPU", allclose(to_np(p_g), p_c.value, atol=1e-3))
    dx_g = to_np(x_g.node.cp)
    dx_c = x_c.node.cp
    check("maxpool grad GPU≈CPU", allclose(dx_g, dx_c, atol=1e-3),
          f"GPU:\n{dx_g}\nCPU:\n{dx_c}")
    check("maxpool grad not zero", np.any(np.abs(dx_g) > 0))


# ═══════════════════════════════════════════════════════════════
#  SECTION 10: EDGE CASES & TRAPS
# ═══════════════════════════════════════════════════════════════

def test_scalar_times_tensor():
    """scalar * tensor and tensor * scalar must give same result."""
    print("\n[10.1] Scalar × Tensor commutativity")
    x_np = np.random.randn(3, 4).astype(np.float32)
    x_g = cuten(x_np)

    a = x_g * 2.5
    b = 2.5 * x_g
    check("scalar*cuten == cuten*scalar (not zero)", np.any(to_np(a) != 0))
    check("scalar*cuten == cuten*scalar", allclose(to_np(a), to_np(b)))


def test_zero_input_nonzero_loss():
    """Zero input should still produce nonzero loss if bias exists."""
    print("\n[10.2] Zero input → nonzero output (bias test)")
    model = Sequential([
        Input((4,)),
        Dense(4, 3, activation="sigmoid"),
    ], device="cuda")

    X = np.zeros((2, 4), dtype=np.float32)
    out = model.forward(Tensor(X, is_leaf=True, device="cuda"))
    out_np = to_np(out)
    check("zero-input output ≈ sigmoid(bias) [not zero]", np.any(out_np != 0),
          f"output={out_np}")


def test_identity_matmul():
    """Multiply by identity matrix — result should be the original."""
    print("\n[10.3] Identity matmul")
    A = np.random.randn(5, 5).astype(np.float32)
    I = np.eye(5, dtype=np.float32)
    result = to_np(cuten(A).matmul(cuten(I)))
    check("A @ I == A", allclose(result, A, atol=5e-3),
          f"max diff={np.max(np.abs(result - A))}")


def test_known_matmul():
    """Known multiplication to verify row/col ordering."""
    print("\n[10.4] Known matmul values")
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)
    expected = A @ B  # [[19,22],[43,50]]
    result = (cuten(A).matmul(cuten(B))).to_host_f32()
    print("="*30)
    print(result)
    print(expected)
    print("="*30)

    check("2x2 known matmul", allclose(result, expected, atol=0.1),
          f"expected:\n{expected}\ngot:\n{result}")
    check("result[0,0] ≈ 19", abs(result[0, 0] - 19) < 0.5,
          f"got {result[0, 0]}")
    check("result[1,1] ≈ 50", abs(result[1, 1] - 50) < 0.5,
          f"got {result[1, 1]}")


def test_gradient_accumulation_not_stale():
    """After zero_grad, old grads must NOT persist."""
    print("\n[10.5] Gradient reset (zero_grad)")
    model = Sequential([
        Input((4,)),
        Dense(4, 3, activation="sigmoid"),
    ], device="cuda")

    X = np.random.randn(2, 4).astype(np.float32)
    Y = np.random.randn(2, 3).astype(np.float32) * 0.5

    # First pass
    out1 = model.forward(Tensor(X, is_leaf=True, device="cuda"))
    loss1 = Loss().mse(out1, Tensor(Y, device="cuda"))
    model.zero_grad()
    autograd4nn(loss1)
    g1 = to_np(model.model[1].weights.node.cp).copy()

    # Second pass with zero_grad
    model.zero_grad()
    g_after_reset = to_np(model.model[1].weights.node.cp)
    check("grad is zero after zero_grad", np.allclose(g_after_reset, 0),
          f"max |grad|={np.max(np.abs(g_after_reset))}")


# ═══════════════════════════════════════════════════════════════
#  SECTION 11: CNN PIPELINE (Conv → Pool → Flatten → Dense)
# ═══════════════════════════════════════════════════════════════

def test_cnn_pipeline_grad():
    """Conv2D → MaxPool → Flatten → Dense backward — full pipeline."""
    print("\n[11.1] CNN pipeline gradient flow")
    x_np = np.random.randn(2, 1, 8, 8).astype(np.float32) * 0.3
    y_np = np.random.randn(2, 2).astype(np.float32) * 0.3

    model = Sequential([
        Input((1, 8, 8)),
        Conv2D(4, 1, (3, 3), activation="relu"),
        MaxPool2D(pool_size=(2, 2), stride=2),
        Flatten(),
        Dense(4 * 3 * 3, 2, activation="sigmoid"),
    ], device="cuda")

    opt = SGD(model, lr=0.01)
    loss_fn = Loss()

    X_t = Tensor(x_np, is_leaf=True, device="cuda")
    Y_t = Tensor(y_np, device="cuda")

    out = model.forward(X_t)
    l = loss_fn.mse(out, Y_t)
    loss_val = float(to_np(l).ravel()[0])
    check("CNN pipeline loss not zero", loss_val > 1e-8, f"loss={loss_val}")

    model.zero_grad()
    autograd4nn(l)

    # Check conv layer grad
    conv_grad = to_np(model.model[1].weights.node.cp)
    check("conv weight grad not zero", np.any(np.abs(conv_grad) > 1e-8),
          f"max |grad|={np.max(np.abs(conv_grad))}")

    # Check dense layer grad
    dense_grad = to_np(model.model[4].weights.node.cp)
    check("dense weight grad not zero", np.any(np.abs(dense_grad) > 1e-8),
          f"max |grad|={np.max(np.abs(dense_grad))}")

    # Do optimizer step and check weights changed
    w_before = to_np(model.model[1].weights.value).copy()
    opt.step()
    w_after = to_np(model.model[1].weights.value).copy()
    check("conv weights updated by SGD", np.max(np.abs(w_after - w_before)) > 1e-10,
          f"max change={np.max(np.abs(w_after - w_before))}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  RUTHLESS GPU DIAGNOSTIC SUITE")
    print("=" * 70)

    # Section 1: cuTen low-level
    test_cuten_roundtrip()
    test_cuten_elementwise()
    test_cuten_broadcast()
    test_cuten_matmul()
    test_cuten_transpose()
    test_cuten_activations()
    test_cuten_softmax()
    test_cuten_reductions()
    test_cuten_conv2d()
    test_cuten_flatten()

    # Section 2: Tensor autograd
    test_tensor_matmul_grad()
    test_tensor_dense_like_grad()
    test_tensor_relu_grad()
    test_tensor_sigmoid_grad()
    test_tensor_softmax_grad()
    test_tensor_reduction_grad()
    test_tensor_pow_grad()

    # Section 3: Loss functions
    test_mse_loss_grad()
    test_cce_loss_grad()

    # Section 4: Dense network
    test_dense_single_step()
    test_dense_weight_actually_changes()
    test_dense_training_convergence()
    test_dense_gpu_vs_cpu_training()

    # Section 5: Classification
    test_classification_convergence()

    # Section 6: Conv2D
    test_conv2d_grad()

    # Section 7: Flatten
    test_flatten_grad()

    # Section 8: Adam
    test_adam_weight_update()

    # Section 9: MaxPool
    test_maxpool_grad()

    # Section 10: Edge cases
    test_scalar_times_tensor()
    test_zero_input_nonzero_loss()
    test_identity_matmul()
    test_known_matmul()
    test_gradient_accumulation_not_stale()

    # Section 11: CNN pipeline
    test_cnn_pipeline_grad()

    # ── Summary ──
    print("\n" + "=" * 70)
    print(f"  RESULTS: {PASS} PASSED, {FAIL} FAILED")
    print("=" * 70)
    if ERRORS:
        print("\n  FAILURES:")
        for e in ERRORS:
            print(f"    ❌ {e}")
    else:
        print("\n  🎉 ALL TESTS PASSED!")

    sys.exit(1 if FAIL > 0 else 0)
