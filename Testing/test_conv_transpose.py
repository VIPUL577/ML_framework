import numpy as np
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Import your framework
from Seera_Engine import Tensor, autograd4nn
from Seera import Conv2D, ConvTranspose2D
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

def assert_ruthless(name, pt_val, sr_val, rtol=1e-3, atol=5e-3):
    """
    Fails loudly if the maximum difference exceeds tolerances.
    atol=5e-3 accounts for FP32 accumulation drift in large convolutions.
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

# ─── Convolution Stress Tests ───────────────────────────────────

def test_large_conv2d():
    print("\n--- Testing Large Conv2D (CUDA) ---")
    # Large spatial dimensions to stress-test memory strides
    N, Cin, H, W = 8, 32, 64, 64
    Cout, KH, KW = 64, 3, 3
    stride, padding = 2, 1
    
    # 1. PyTorch Setup
    torch.manual_seed(42)
    pt_X = torch.randn(N, Cin, H, W, device="cuda", requires_grad=True)
    pt_conv = nn.Conv2d(Cin, Cout, (KH, KW), stride=stride, padding=padding).cuda()
    
    # 2. Extract weights (PyTorch Conv2D weights are [Cout, Cin, KH, KW])
    W_np = pt_conv.weight.detach().cpu().numpy()
    # PyTorch bias is [Cout]. Seera broadcasts bias as [1, Cout, 1, 1]
    B_np = pt_conv.bias.detach().cpu().numpy().reshape(1, Cout, 1, 1)
    X_np = pt_X.detach().cpu().numpy()
    
    # 3. Seera Setup
    sr_X = Tensor(X_np, device="cuda", is_leaf=True)
    sr_conv = Conv2D(Cout, Cin, (KH, KW), activation="relu", stride=stride, zero_padding=padding)
    
    sr_W = Tensor(W_np, device="cuda", is_leaf=True)
    sr_B = Tensor(B_np, device="cuda", is_leaf=True)
    sr_conv.set_weights(sr_W, sr_B)
    
    # Link input manually to bypass Sequential container requirement
    sr_conv.inpobj = type('obj', (object,), {'outputs': sr_X})
    
    # 4. Forward
    pt_out = torch.relu(pt_conv(pt_X))
    sr_out = sr_conv.forward()
    
    assert_ruthless("Conv2D Forward", pt_out, sr_out)
    
    # 5. Backward (using sum as dummy loss)
    pt_out.sum().backward()
    autograd4nn(sr_out.sum())
    
    # 6. Compare Gradients
    assert_ruthless("Conv2D X Grad", pt_X.grad, get_seera_grad(sr_X))
    assert_ruthless("Conv2D W Grad", pt_conv.weight.grad, get_seera_grad(sr_W))
    assert_ruthless("Conv2D B Grad", pt_conv.bias.grad.reshape(1, -1, 1, 1), get_seera_grad(sr_B))


def test_large_conv_transpose2d():
    print("\n--- Testing Large ConvTranspose2D (CUDA) ---")
    # Upsampling from 32x32 back to 64x64
    N, Cin, H, Win = 8, 64, 32, 32
    Cout, KH, KW = 32, 4, 4
    stride, padding = 2, 1
    
    # 1. PyTorch Setup
    torch.manual_seed(42)
    pt_X = torch.randn(N, Cin, H, Win, device="cuda", requires_grad=True)
    # PyTorch weight shape for ConvTranspose2d is [Cin, Cout, KH, KW]
    pt_convT = nn.ConvTranspose2d(Cin, Cout, (KH, KW), stride=stride, padding=padding).cuda()
    
    # 2. Extract weights
    W_np = pt_convT.weight.detach().cpu().numpy()
    B_np = pt_convT.bias.detach().cpu().numpy().reshape(1, Cout, 1, 1)
    X_np = pt_X.detach().cpu().numpy()
    
    # 3. Seera Setup
    sr_X = Tensor(X_np, device="cuda", is_leaf=True)
    sr_convT = ConvTranspose2D(Cout, Cin, (KH, KW), activation="relu", stride=stride, zero_padding=padding)
    
    sr_W = Tensor(W_np, device="cuda", is_leaf=True)
    sr_B = Tensor(B_np, device="cuda", is_leaf=True)
    sr_convT.set_weights(sr_W, sr_B)
    sr_convT.inpobj = type('obj', (object,), {'outputs': sr_X})
    
    # 4. Forward
    pt_out = torch.relu(pt_convT(pt_X))
    sr_out = sr_convT.forward()
    
    assert_ruthless("ConvTranspose2D Forward", pt_out, sr_out)
    
    # 5. Backward
    pt_out.sum().backward()
    autograd4nn(sr_out.sum())
    
    # 6. Compare Gradients
    assert_ruthless("ConvTranspose2D X Grad", pt_X.grad, get_seera_grad(sr_X))
    assert_ruthless("ConvTranspose2D W Grad", pt_convT.weight.grad, get_seera_grad(sr_W))
    assert_ruthless("ConvTranspose2D B Grad", pt_convT.bias.grad.reshape(1, -1, 1, 1), get_seera_grad(sr_B))

if __name__ == "__main__":
    test_large_conv2d()
    test_large_conv_transpose2d()
    print("\n🏁 Conv2D and ConvTranspose2D tests complete.")