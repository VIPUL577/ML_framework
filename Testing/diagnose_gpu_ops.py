"""
Minimal GPU operation diagnostic — isolate where values go wrong.
Each test checks ONE operation and prints the raw GPU output.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from cuTen import cuten
import seera_cuda

print("=" * 60)
print("  GPU Operation-Level Diagnostic")
print("=" * 60)

# ── 1. Basic transfer: CPU → GPU → CPU roundtrip ──
print("\n--- 1. Transfer roundtrip ---")
a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
a_gpu = cuten(a_np)
a_back = a_gpu.to_host_f32()
print(f"  Input:  {a_np}")
print(f"  Output: {a_back}")
print(f"  Match: {np.allclose(a_np, a_back)}")

# ── 2. Scalar multiply ──
print("\n--- 2. Scalar multiply ---")
b_gpu = a_gpu * 3.0
b_back = b_gpu.to_host_f32()
print(f"  Input * 3.0 = {b_back}")
print(f"  Expected:     {a_np * 3.0}")
print(f"  Match: {np.allclose(a_np * 3.0, b_back)}")

# ── 3. Element-wise add (same shape) ──
print("\n--- 3. Elemadd (same shape) ---")
x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
y_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
x_gpu = cuten(x_np)
y_gpu = cuten(y_np)
z_gpu = x_gpu + y_gpu
z_back = z_gpu.to_host_f32()
print(f"  x + y = {z_back}")
print(f"  Expected: {x_np + y_np}")
print(f"  Match: {np.allclose(x_np + y_np, z_back)}")

# ── 4. Element-wise mul (same shape) ──
print("\n--- 4. Elemmul (same shape) ---")
m_gpu = x_gpu * y_gpu
m_back = m_gpu.to_host_f32()
print(f"  x * y = {m_back}")
print(f"  Expected: {x_np * y_np}")
print(f"  Match: {np.allclose(x_np * y_np, m_back)}")

# ── 5. Broadcast add: (4, 3) + (1, 3) ──
print("\n--- 5. Broadcast add ---")
p_np = np.arange(12, dtype=np.float32).reshape(4, 3)
q_np = np.array([[100, 200, 300]], dtype=np.float32)
p_gpu = cuten(p_np)
q_gpu = cuten(q_np)
r_gpu = p_gpu + q_gpu
r_back = r_gpu.to_host_f32()
print(f"  p + q = \n{r_back}")
print(f"  Expected: \n{p_np + q_np}")
print(f"  Match: {np.allclose(p_np + q_np, r_back)}")

# ── 6. Matmul forward ──
print("\n--- 6. Matmul forward ---")
A_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
B_np = np.array([[5, 6], [7, 8]], dtype=np.float32)
A_gpu = cuten(A_np)
B_gpu = cuten(B_np)
C_gpu = A_gpu.matmul(B_gpu)
C_back = C_gpu.to_host_f32()
C_expected = A_np @ B_np
print(f"  A @ B = \n{C_back}")
print(f"  Expected: \n{C_expected}")
print(f"  Match: {np.allclose(C_expected, C_back, atol=1.0)}")

# ── 7. Matmul backward (manual) ──
print("\n--- 7. Matmul backward (manual) ---")
A_np2 = np.random.randn(4, 3).astype(np.float32) * 0.1
B_np2 = np.random.randn(3, 2).astype(np.float32) * 0.1
dC_np2 = np.ones((4, 2), dtype=np.float32)

A2 = cuten(A_np2)
B2 = cuten(B_np2)
dC2 = cuten(dC_np2)

M, K, N = 4, 3, 2
dA_ptr = seera_cuda.cuda_malloc_f32(M * K)
dB_ptr = seera_cuda.cuda_malloc_f32(K * N)
seera_cuda.cuda_matmul_bwd(A2.main_ptr, B2.main_ptr, dC2.main_ptr,
                            dA_ptr, dB_ptr, M, N, K, 1)

dA_back = seera_cuda.to_host_f32(dA_ptr, (M, K))
dB_back = seera_cuda.to_host_f32(dB_ptr, (K, N))

dA_expected = dC_np2 @ B_np2.T
dB_expected = A_np2.T @ dC_np2

print(f"  dA (GPU) = \n{dA_back}")
print(f"  dA (exp) = \n{dA_expected}")
print(f"  dA diff  = {np.max(np.abs(dA_back - dA_expected)):.6f}")
print(f"  dB (GPU) = \n{dB_back}")
print(f"  dB (exp) = \n{dB_expected}")
print(f"  dB diff  = {np.max(np.abs(dB_back - dB_expected)):.6f}")

seera_cuda.cuda_free(dA_ptr)
seera_cuda.cuda_free(dB_ptr)

# ── 8. Sum reduction ──
print("\n--- 8. Sum reduction ---")
s_np = np.arange(12, dtype=np.float32).reshape(3, 4)
s_gpu = cuten(s_np)
s_sum0 = s_gpu.sum(dim=0)  # shape (4,)
s_sum1 = s_gpu.sum(dim=1)  # shape (3,)
print(f"  Input shape: {s_gpu.shape}")
print(f"  sum(dim=0) shape={s_sum0.shape}: {s_sum0.to_host_f32()}")
print(f"  Expected:                        {s_np.sum(axis=0)}")
print(f"  sum(dim=1) shape={s_sum1.shape}: {s_sum1.to_host_f32()}")
print(f"  Expected:                        {s_np.sum(axis=1)}")

# ── 9. Sum backward (broadcast) ──
print("\n--- 9. Sum backward ---")
grad_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
grad_gpu = cuten(grad_np)
dimarr = np.array([3, 4], dtype=np.int32)
out_ptr = seera_cuda.cuda_malloc_f32(12)
seera_cuda.cuda_sum_bwd(grad_gpu.main_ptr, out_ptr, 2, 0, dimarr)
out_back = seera_cuda.to_host_f32(out_ptr, (3, 4))
print(f"  Sum bwd (broadcast [4] → [3,4]):")
print(f"  Result:\n{out_back}")
print(f"  Expected:\n{np.broadcast_to(grad_np, (3, 4))}")
seera_cuda.cuda_free(out_ptr)

# ── 10. Softmax ──
print("\n--- 10. Softmax ---")
logits_np = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
logits_gpu = cuten(logits_np)
sm_out_ptr = seera_cuda.cuda_malloc_f32(6)
seera_cuda.cuda_softmax_fwd(logits_gpu.main_ptr, sm_out_ptr, 2, 3)
sm_back = seera_cuda.to_host_f32(sm_out_ptr, (2, 3))
# NumPy softmax
def np_softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
sm_expected = np_softmax(logits_np)
print(f"  Softmax out:\n{sm_back}")
print(f"  Expected:\n{sm_expected}")
print(f"  Max diff: {np.max(np.abs(sm_back - sm_expected)):.6f}")
seera_cuda.cuda_free(sm_out_ptr)

# ── 11. ReLU activation ──
print("\n--- 11. ReLU ---")
relu_in_np = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
relu_in_gpu = cuten(relu_in_np)
relu_out_ptr = seera_cuda.cuda_malloc_f32(5)
relu_grad_ptr = seera_cuda.cuda_malloc_f32(5)
seera_cuda.cuda_relu_fwd(relu_in_gpu.main_ptr, relu_out_ptr, relu_grad_ptr, 5)
relu_out = seera_cuda.to_host_f32(relu_out_ptr, (5,))
relu_grad = seera_cuda.to_host_f32(relu_grad_ptr, (5,))
print(f"  Input:    {relu_in_np}")
print(f"  ReLU out: {relu_out}")
print(f"  Expected: {np.maximum(relu_in_np, 0)}")
print(f"  Grad:     {relu_grad}")
print(f"  Expected: {(relu_in_np > 0).astype(np.float32)}")
seera_cuda.cuda_free(relu_out_ptr)
seera_cuda.cuda_free(relu_grad_ptr)

# ── 12. Log activation ──
print("\n--- 12. Log ---")
log_in_np = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float32)
log_in_gpu = cuten(log_in_np)
log_out_ptr = seera_cuda.cuda_malloc_f32(4)
log_grad_ptr = seera_cuda.cuda_malloc_f32(4)
seera_cuda.cuda_log_fwd(log_in_gpu.main_ptr, log_out_ptr, log_grad_ptr, 4)
log_out = seera_cuda.to_host_f32(log_out_ptr, (4,))
log_grad = seera_cuda.to_host_f32(log_grad_ptr, (4,))
print(f"  Input:    {log_in_np}")
print(f"  Log out:  {log_out}")
print(f"  Expected: {np.log(log_in_np)}")
print(f"  Grad:     {log_grad}")
print(f"  Expected: {1.0 / log_in_np}")
seera_cuda.cuda_free(log_out_ptr)
seera_cuda.cuda_free(log_grad_ptr)

# ── 13. Full Tensor forward test: Dense layer ──
print("\n--- 13. Tensor-level: X @ W + b ---")
from Seera_init import tensor as Tensor
X_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
W_np = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
b_np = np.array([[1, 2, 3]], dtype=np.float32)

X = Tensor(X_np, is_leaf=True, device="cuda")
W = Tensor(W_np, is_leaf=True, device="cuda")
b = Tensor(b_np, device="cuda")

z = X.matmul(W) + b
z_np = z.value.to_host_f32()
z_expected = X_np @ W_np + b_np
print(f"  z (GPU)  = \n{z_np}")
print(f"  z (exp)  = \n{z_expected}")
print(f"  Max diff = {np.max(np.abs(z_np - z_expected)):.6f}")

# ── 14. Full CCE loss computation ──
print("\n--- 14. CCE loss ---")
from Seera import Loss
pred_np = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)
target_np = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
pred = Tensor(pred_np, is_leaf=True, device="cuda")
target = Tensor(target_np, device="cuda")
loss = Loss().categorical_cross_entropy(pred, target)
loss_val = loss.value.to_host_f32()
loss_expected = (-target_np * np.log(pred_np + 1e-15)).sum(axis=-1).mean()
print(f"  CCE loss (GPU):  {loss_val}")
print(f"  CCE loss (exp):  {loss_expected}")

# Check intermediate nodes 
print("\n  --- Tracing CCE graph nodes ---")
def trace_tensor(t, depth=0, visited=None):
    if visited is None:
        visited = set()
    if id(t) in visited:
        return
    visited.add(id(t))
    prefix = "  " * (depth + 1)
    val = t.value
    vshape = val.shape if hasattr(val, 'shape') else 'scalar'
    if isinstance(val, cuten):
        host_val = val.to_host_f32()
        print(f"{prefix}shape={vshape} val[:4]={host_val.ravel()[:4]} "
              f"matm={t.matm} softmax={t.isoftmax} "
              f"reduction={t.ireduction is not None}")
    else:
        print(f"{prefix}shape={vshape} val[:4]={np.asarray(val).ravel()[:4]} "
              f"matm={t.matm}")
    for child in t.node.child:
        trace_tensor(child, depth+1, visited)

trace_tensor(loss)

print("\n" + "=" * 60)
print("  Diagnostic complete!")
print("=" * 60)
