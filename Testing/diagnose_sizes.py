"""
Targeted diagnostic: Test which matrix sizes and reduction shapes work/fail.
Includes batch=1 tests per user request.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from cuTen import cuten
import seera_cuda

print("=" * 60)
print("  Targeted Size Diagnostic")
print("=" * 60)

# ── 1. Matmul at various sizes ──
print("\n--- Matmul size sweep ---")
test_sizes = [
    (1, 3, 2),    # batch=1
    (1, 8, 4),    # batch=1
    (1, 256, 128),# batch=1 large
    (4, 3, 2),    # tiny
    (8, 8, 8),    # small square
    (16, 32, 16), # Test D like
    (16, 64, 32), # Test D large
    (8, 256, 128),# Test H first layer
    (32, 8, 4),   # Test J like
    (32, 16, 4),  # Test J like
    (8, 128, 64), # Test H second layer
    (16, 400, 256),# Test F first layer
    (8, 64, 128), # Test H layer 3 (expand)
    (8, 128, 5),  # Test H last layer
]

for M, K, N in test_sizes:
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32) * 0.1
    B = np.random.randn(K, N).astype(np.float32) * 0.1

    A_gpu = cuten(A)
    B_gpu = cuten(B)
    C_gpu = A_gpu.matmul(B_gpu)
    C_np = C_gpu.to_host_f32()
    C_exp = A @ B
    diff = np.max(np.abs(C_np - C_exp))
    status = "✅" if diff < 0.5 else "❌"
    print(f"  {status} ({M},{K})@({K},{N})  max_diff={diff:.6f}  "
          f"out[:2,:2]={C_np.ravel()[:4]}  "
          f"exp[:2,:2]={C_exp.ravel()[:4]}")

# ── 2. Reduction sum at various shapes ──
print("\n--- Sum reduction size sweep ---")
reduce_tests = [
    ((3, 4), 0),
    ((3, 4), 1),
    ((8, 16), 0),  # Test I failing
    ((8, 16), 1),
    ((16, 32), 0),
    ((16, 64), 0),
    ((8, 256), 0),
    ((8, 256), 1),
    ((32, 8), 0),
    ((32, 8), 1),
    ((1, 8), 0),   # batch=1
    ((1, 128), 0),
]

for shape, dim in reduce_tests:
    np.random.seed(42)
    data = np.random.randn(*shape).astype(np.float32)
    data_gpu = cuten(data)
    result_gpu = data_gpu.sum(dim=dim)
    result_np = result_gpu.to_host_f32()
    expected = data.sum(axis=dim)
    diff = np.max(np.abs(result_np.ravel() - expected.ravel()))
    status = "✅" if diff < 0.01 else "❌"
    print(f"  {status} shape={shape} sum(dim={dim})  "
          f"out_shape={result_gpu.shape} max_diff={diff:.6f}  "
          f"out[:4]={result_np.ravel()[:4]}  exp[:4]={expected.ravel()[:4]}")

# ── 3. Forward pass through Dense layers of various sizes ──
print("\n--- Dense forward: matmul + broadcast add ---")
dense_tests = [
    (1, 3, 2),     # batch=1
    (1, 256, 128), # batch=1 large
    (8, 3, 2),     # small
    (8, 8, 4),     # Test J
    (16, 32, 64),  # Test D layer 1
    (8, 256, 128), # Test H layer 1
    (16, 400, 256),# Test F layer 1
]

for batch, inp, out in dense_tests:
    np.random.seed(42)
    X = np.random.randn(batch, inp).astype(np.float32) * 0.1
    W = np.random.randn(inp, out).astype(np.float32) * 0.05
    b = np.zeros((1, out), dtype=np.float32)

    X_gpu = cuten(X)
    W_gpu = cuten(W)
    b_gpu = cuten(b)

    # Step by step
    z_matmul = X_gpu.matmul(W_gpu)
    z_matmul_np = z_matmul.to_host_f32()
    z_matmul_exp = X @ W

    z_full = z_matmul + b_gpu
    z_np = z_full.to_host_f32()
    z_exp = X @ W + b

    diff_mm = np.max(np.abs(z_matmul_np - z_matmul_exp))
    diff_add = np.max(np.abs(z_np - z_exp))
    status = "✅" if diff_add < 1.0 else "❌"
    print(f"  {status} ({batch},{inp})@({inp},{out})+b  "
          f"mm_diff={diff_mm:.6f}  add_diff={diff_add:.6f}  "
          f"mm[:2]={z_matmul_np.ravel()[:2]}  exp_mm[:2]={z_matmul_exp.ravel()[:2]}")

# ── 4. Multi-layer forward with intermediate checks ──
print("\n--- Multi-layer forward (Test H equivalent) ---")
from Seera_init import tensor as Tensor
from Seera import Dense, Input, Sequential, Loss

np.random.seed(42)
W1 = np.random.randn(256, 128).astype(np.float32) * 0.02
b1 = np.zeros((1, 128), dtype=np.float32)
W2 = np.random.randn(128, 64).astype(np.float32) * 0.02
b2 = np.zeros((1, 64), dtype=np.float32)
W3 = np.random.randn(64, 128).astype(np.float32) * 0.02
b3 = np.zeros((1, 128), dtype=np.float32)
W4 = np.random.randn(128, 5).astype(np.float32) * 0.02
b4 = np.zeros((1, 5), dtype=np.float32)

x_np = np.random.randn(8, 256).astype(np.float32) * 0.1

# ── NumPy reference ──
z1_np = np.maximum(x_np @ W1 + b1, 0)  # relu
z2_np = np.maximum(z1_np @ W2 + b2, 0)
z3_np = np.maximum(z2_np @ W3 + b3, 0)
z4_np = z3_np @ W4 + b4
e = np.exp(z4_np - z4_np.max(axis=-1, keepdims=True))
softmax_np = e / e.sum(axis=-1, keepdims=True)

print(f"  NumPy z1 max={np.max(np.abs(z1_np)):.4f}  z2 max={np.max(np.abs(z2_np)):.4f}")
print(f"  NumPy z3 max={np.max(np.abs(z3_np)):.4f}  z4 max={np.max(np.abs(z4_np)):.4f}")
print(f"  NumPy softmax[:2] = {softmax_np[:2]}")

# ── GPU step-by-step ──
x_gpu = cuten(x_np)
W1_gpu, b1_gpu = cuten(W1), cuten(b1)
W2_gpu, b2_gpu = cuten(W2), cuten(b2)
W3_gpu, b3_gpu = cuten(W3), cuten(b3)
W4_gpu, b4_gpu = cuten(W4), cuten(b4)

# Layer 1
mm1 = x_gpu.matmul(W1_gpu)
mm1_np = mm1.to_host_f32()
print(f"  GPU mm1 max={np.max(np.abs(mm1_np)):.4f}  diff={np.max(np.abs(mm1_np - (x_np @ W1))):.6f}")

z1_gpu = mm1 + b1_gpu
z1_host = z1_gpu.to_host_f32()
print(f"  GPU z1+b max={np.max(np.abs(z1_host)):.4f}  diff={np.max(np.abs(z1_host - (x_np@W1+b1))):.6f}")

# ReLU
relu1_out_ptr = seera_cuda.cuda_malloc_f32(z1_gpu.size)
relu1_grad_ptr = seera_cuda.cuda_malloc_f32(z1_gpu.size)
seera_cuda.cuda_relu_fwd(z1_gpu.main_ptr, relu1_out_ptr, relu1_grad_ptr, z1_gpu.size)
r1 = cuten(data=None, dtype="float32")
r1.main_ptr = relu1_out_ptr
r1.shape = z1_gpu.shape
r1.size = z1_gpu.size
r1_np = r1.to_host_f32()
print(f"  GPU relu1 max={np.max(np.abs(r1_np)):.4f}  diff={np.max(np.abs(r1_np - z1_np)):.6f}")

# Layer 2
mm2 = r1.matmul(W2_gpu)
mm2_np = mm2.to_host_f32()
print(f"  GPU mm2 max={np.max(np.abs(mm2_np)):.4f}  diff={np.max(np.abs(mm2_np - (z1_np@W2))):.6f}")

# Skip to layer 4 output
z2_gpu_plus_b = mm2 + b2_gpu
relu2_ptr = seera_cuda.cuda_malloc_f32(z2_gpu_plus_b.size)
relu2_grad = seera_cuda.cuda_malloc_f32(z2_gpu_plus_b.size)
seera_cuda.cuda_relu_fwd(z2_gpu_plus_b.main_ptr, relu2_ptr, relu2_grad, z2_gpu_plus_b.size)
r2 = cuten(data=None, dtype="float32")
r2.main_ptr = relu2_ptr
r2.shape = z2_gpu_plus_b.shape
r2.size = z2_gpu_plus_b.size
r2_np = r2.to_host_f32()
print(f"  GPU relu2 max={np.max(np.abs(r2_np)):.4f}  diff={np.max(np.abs(r2_np - z2_np)):.6f}")

mm3 = r2.matmul(W3_gpu)
z3_gpu_plus_b = mm3 + b3_gpu
relu3_ptr = seera_cuda.cuda_malloc_f32(z3_gpu_plus_b.size)
relu3_grad = seera_cuda.cuda_malloc_f32(z3_gpu_plus_b.size)
seera_cuda.cuda_relu_fwd(z3_gpu_plus_b.main_ptr, relu3_ptr, relu3_grad, z3_gpu_plus_b.size)
r3 = cuten(data=None, dtype="float32")
r3.main_ptr = relu3_ptr
r3.shape = z3_gpu_plus_b.shape
r3.size = z3_gpu_plus_b.size
r3_np = r3.to_host_f32()
print(f"  GPU relu3 max={np.max(np.abs(r3_np)):.4f}  diff={np.max(np.abs(r3_np - z3_np)):.6f}")

mm4 = r3.matmul(W4_gpu)
z4_gpu_plus_b = mm4 + b4_gpu
z4_host = z4_gpu_plus_b.to_host_f32()
print(f"  GPU z4 (pre-softmax) max={np.max(np.abs(z4_host)):.4f}  diff={np.max(np.abs(z4_host - z4_np)):.6f}")
print(f"  GPU z4[:2] = {z4_host[:2]}")
print(f"  NP  z4[:2] = {z4_np[:2]}")

# Softmax
sm_ptr = seera_cuda.cuda_malloc_f32(z4_gpu_plus_b.size)
seera_cuda.cuda_softmax_fwd(z4_gpu_plus_b.main_ptr, sm_ptr, z4_gpu_plus_b.shape[0], z4_gpu_plus_b.shape[1])
sm_host = seera_cuda.to_host_f32(sm_ptr, z4_gpu_plus_b.shape)
print(f"  GPU softmax[:2] = {sm_host[:2]}")
sm_diff = np.max(np.abs(sm_host - softmax_np))
print(f"  softmax diff = {sm_diff:.6f}")

# ── 5. Test using model.fit with batch=1 ──
print("\n--- model.fit with batch_size=1 ---")
np.random.seed(42)
x_fit = np.random.randn(4, 3).astype(np.float32)
y_fit = np.zeros((4, 2), dtype=np.float32)
for i in range(4):
    y_fit[i, i % 2] = 1.0

model = Sequential([
    Input((3,)),
    Dense(3, 2, activation="softmax"),
], "cuda")

loss_fn = Loss()
from Seera import SGD
optimizer = SGD(model, lr=0.01)

history = model.fit(
    x_fit, y_fit,
    Optimizer=optimizer,
    Loss=loss_fn.categorical_cross_entropy,
    Epochs=5,
    batch_size=1,
    Loss_interval=1,
)
print(f"  history = {history}")

print("\n" + "=" * 60)
print("  Diagnostic complete!")
print("=" * 60)
