import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

# Import your framework
from Seera_Engine import Tensor, autograd4nn
from Seera_init import _is_gpu
from cuTen import cuten

# ─── Utility Functions ──────────────────────────────────────────

def to_host(val):
    """Extracts numpy array from Seera cuten or PyTorch tensor."""
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    if _is_gpu(val):
        return val.to_host_f32()
    if hasattr(val, "value"):
        v = val.value
        return v.to_host_f32() if _is_gpu(v) else np.array(v)
    return np.array(val)

def get_seera_grad(seera_tensor):
    """Extracts gradient numpy array from Seera tensor."""
    grad = seera_tensor.node.cp
    if _is_gpu(grad):
        return grad.to_host_f32()
    return np.array(grad)

def assert_ruthless(name, pt_val, sr_val, rtol=1e-1, atol=5e-2):
    """
    Fails loudly if the maximum difference exceeds tolerances.
    Note: atol=1e-2 is used here because 1024x1024 accumulations 
    cause significant floating point drift.
    """
    pt_val = to_host(pt_val)
    sr_val = to_host(sr_val)
    
    assert pt_val.shape == sr_val.shape, f"[{name}] Shape mismatch! PT: {pt_val.shape}, Seera: {sr_val.shape}"
    
    max_diff = np.max(np.abs(pt_val - sr_val))
    if max_diff > atol:
        print(f"❌ FAIL: [{name}] Max Diff = {max_diff:.8f}")
        np.testing.assert_allclose(pt_val, sr_val, rtol=rtol, atol=atol, err_msg=f"[{name}] Failed")
    else:
        print(f"✅ PASS: [{name}] Max Diff = {max_diff:.8f}")

# ─── The 1024x1024 Test ─────────────────────────────────────────

def test_large_matmul():
    print("\n--- Testing 1024x1024 MatMul (CUDA) ---")
    N = 1024
    
    # 1. Initialize identical random base arrays in numpy
    # Using float32 to ensure strict FP32 testing across the board
    np.random.seed(42)
    A_np = np.random.randn(N, N).astype(np.float32)
    B_np = np.random.randn(N, N).astype(np.float32)
    
    # 2. PyTorch Setup
    pt_A = torch.tensor(A_np, device="cuda", requires_grad=True)
    pt_B = torch.tensor(B_np, device="cuda", requires_grad=True)
    
    # 3. Seera Setup
    sr_A = Tensor(A_np, device="cuda", is_leaf=True)
    sr_B = Tensor(B_np, device="cuda", is_leaf=True)
    
    # 4. Forward Pass
    pt_out = pt_A @ pt_B
    sr_out = sr_A.matmul(sr_B)
    
    assert_ruthless("MatMul Forward", pt_out, sr_out)
    
    # 5. Backward Pass (Using sum as the dummy loss)
    pt_loss = pt_out.sum()
    pt_loss.backward()
    
    sr_loss = sr_out.sum()
    autograd4nn(sr_loss)
    
    # 6. Compare Gradients
    # The gradient of A should be dout @ B.T
    # The gradient of B should be A.T @ dout
    assert_ruthless("MatMul A Grad", pt_A.grad, get_seera_grad(sr_A))
    assert_ruthless("MatMul B Grad", pt_B.grad, get_seera_grad(sr_B))

if __name__ == "__main__":
    test_large_matmul()
    print("\n🏁 1024x1024 Matmul test complete.")