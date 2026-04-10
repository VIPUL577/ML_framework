import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your framework
from Seera import Sequential, Input, Dense, Conv2D, ConvTranspose2D, MaxPool2D, Flatten, Concatenate, BatchNorm1d, BatchNorm2d, Loss, SGD, Adam
from Seera_init import tensor as Tensor
from Seera_Engine import autograd4nn
from cuTen import cuten

# ─── UTILITIES FOR SYNCHRONIZATION ──────────────────────────────────
def to_numpy(t):
    """Safely extract numpy array from PyTorch or Seera tensor."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    if hasattr(t, "to_host_f32"): # cuTen
        return t.to_host_f32()
    if hasattr(t, "value"): # Seera Tensor
        val = t.value
        return val.to_host_f32() if hasattr(val, "to_host_f32") else val
    return t

def check_match(pt_tensor, seera_tensor, name="Tensor", atol=1e-4):
    pt_np = to_numpy(pt_tensor)
    sr_np = to_numpy(seera_tensor)
    
    if pt_np.shape != sr_np.shape:
        print(f"❌ {name} Shape Mismatch: PT {pt_np.shape} vs Seera {sr_np.shape}")
        return False
        
    diff = np.abs(pt_np - sr_np).max()
    if diff > atol:
        print(f"❌ {name} Value Mismatch! Max diff: {diff:.6f}")
        return False
    print(f"✅ {name} Matches. (Max diff: {diff:.6e})")
    return True

def sync_dense(pt_layer, seera_layer):
    """PyTorch is (out, in), Seera is (in, out). PyTorch bias is (out,), Seera is (1, out)"""
    w = pt_layer.weight.detach().cpu().numpy().T
    b = pt_layer.bias.detach().cpu().numpy().reshape(1, -1)
    seera_layer.set_weights(w, b)

def sync_conv2d(pt_layer, seera_layer):
    """Both use (out, in, h, w). Bias needs reshaping to (1, out, 1, 1)"""
    w = pt_layer.weight.detach().cpu().numpy()
    b = pt_layer.bias.detach().cpu().numpy().reshape(1, -1, 1, 1)
    seera_layer.set_weights(w, b)

# ─── THE 15 EXTENSIVE TESTS ─────────────────────────────────────────

def test_01_large_dense_fp_bp():
    print("\n--- Test 1: Large Dense Layer Match ---")
    N, IN_F, OUT_F = 64, 1024, 512
    x_np = np.random.randn(N, IN_F).astype(np.float32)
    
    # PyTorch
    pt_x = torch.tensor(x_np, requires_grad=True)
    pt_dense = nn.Linear(IN_F, OUT_F)
    pt_out = pt_dense(pt_x)
    pt_out.sum().backward()
    
    # Seera
    sr_x = Tensor(x_np, is_leaf=True, device="cuda")
    sr_dense = Dense(IN_F, OUT_F, activation="relu")
    sync_dense(pt_dense, sr_dense)
    sr_dense.inpobj = type('obj', (object,), {'outputs': sr_x}) # Mock previous layer
    sr_out = sr_dense.forward()
    autograd4nn(sr_out.sum())
    
    # We compare pre-activation (z) or remove relu for pure linear test. 
    # Let's compare weight gradients.
    check_match(pt_dense.weight.grad.T, sr_dense.weights.node.cp, "Dense Weight Grad")
    check_match(pt_dense.bias.grad.reshape(1, -1), sr_dense.bais.node.cp, "Dense Bias Grad")

def test_02_large_conv2d_fp_bp():
    print("\n--- Test 2: Large Conv2D Match ---")
    N, C, H, W = 16, 64, 32, 32
    OUT_C, K, S, P = 128, 3, 2, 1
    x_np = np.random.randn(N, C, H, W).astype(np.float32)
    
    pt_x = torch.tensor(x_np, requires_grad=True)
    pt_conv = nn.Conv2d(C, OUT_C, K, stride=S, padding=P)
    pt_out = pt_conv(pt_x)
    pt_out.sum().backward()
    
    sr_x = Tensor(x_np, is_leaf=True, device="cuda")
    sr_conv = Conv2D(OUT_C, C, (K, K), activation="relu", stride=S, zero_padding=P)
    sync_conv2d(pt_conv, sr_conv)
    sr_conv.inpobj = type('obj', (object,), {'outputs': sr_x})
    sr_out = sr_conv.forward()
    # Mocking backward from the pre-activation for pure Conv match
    autograd4nn(sr_conv.z.sum())
    
    check_match(pt_out, sr_conv.z, "Conv2D Forward")
    check_match(pt_conv.weight.grad, sr_conv.weights.node.cp, "Conv2D Weight Grad")

def test_03_conv_transpose2d():
    print("\n--- Test 3: ConvTranspose2D (Deconv) Match ---")
    N, C, H, W = 8, 32, 16, 16
    OUT_C, K, S, P = 16, 4, 2, 1
    x_np = np.random.randn(N, C, H, W).astype(np.float32)
    
    pt_x = torch.tensor(x_np, requires_grad=True)
    pt_convT = nn.ConvTranspose2d(C, OUT_C, K, stride=S, padding=P)
    pt_out = pt_convT(pt_x)
    pt_out.sum().backward()
    
    sr_x = Tensor(x_np, is_leaf=True, device="cuda")
    sr_convT = ConvTranspose2D(OUT_C, C, (K, K), activation="relu", stride=S, zero_padding=P)
    
    # Sync weights
    w = pt_convT.weight.detach().cpu().numpy()
    b = pt_convT.bias.detach().cpu().numpy().reshape(1, -1, 1, 1)
    sr_convT.set_weights(w, b)
    
    sr_convT.inpobj = type('obj', (object,), {'outputs': sr_x})
    sr_convT.forward()
    autograd4nn(sr_convT.z.sum())
    
    check_match(pt_out, sr_convT.z, "ConvTranspose2D Forward")
    check_match(pt_convT.weight.grad, sr_convT.weights.node.cp, "ConvTranspose2D Weight Grad")

def test_04_maxpool2d():
    print("\n--- Test 4: MaxPool2D Match ---")
    N, C, H, W = 8, 16, 32, 32
    x_np = np.random.randn(N, C, H, W).astype(np.float32)
    
    pt_x = torch.tensor(x_np, requires_grad=True)
    pt_out = F.max_pool2d(pt_x, 2, stride=2)
    pt_out.sum().backward()
    
    sr_x = Tensor(x_np, is_leaf=True, device="cuda")
    sr_pool = MaxPool2D(pool_size=(2, 2), stride=2)
    sr_pool.inpobj = type('obj', (object,), {'outputs': sr_x})
    sr_out = sr_pool.forward()
    autograd4nn(sr_out.sum())
    
    check_match(pt_out, sr_out, "MaxPool2D Forward")
    check_match(pt_x.grad, sr_x.node.cp, "MaxPool2D Input Grad")

def test_05_unpool_nearest():
    print("\n--- Test 5: Unpool / Upsample Nearest Match ---")
    N, C, H, W = 4, 8, 16, 16
    x_np = np.random.randn(N, C, H, W).astype(np.float32)
    
    pt_x = torch.tensor(x_np, requires_grad=True)
    pt_out = F.interpolate(pt_x, scale_factor=2, mode='nearest')
    pt_out.sum().backward()
    
    sr_x = Tensor(x_np, is_leaf=True, device="cuda")
    sr_out = Tensor.Unpool2Dnearest(sr_x, size=(2, 2))
    autograd4nn(sr_out.sum())
    
    check_match(pt_out, sr_out, "Unpool Forward")
    check_match(pt_x.grad, sr_x.node.cp, "Unpool Input Grad")

def test_06_batchnorm1d():
    print("\n--- Test 6: BatchNorm1d Match ---")
    N, C = 32, 128
    x_np = np.random.randn(N, C).astype(np.float32)
    
    pt_x = torch.tensor(x_np, requires_grad=True)
    pt_bn = nn.BatchNorm1d(C, momentum=0.1, eps=1e-5)
    pt_out = pt_bn(pt_x)
    pt_out.sum().backward()
    
    sr_x = Tensor(x_np, is_leaf=True, device="cuda")
    sr_bn = BatchNorm1d(C, momentum=0.1, eps=1e-5)
    sr_bn.inpobj = type('obj', (object,), {'outputs': sr_x})
    sr_out = sr_bn.forward()
    autograd4nn(sr_out.sum())
    
    check_match(pt_out, sr_out, "BatchNorm1d Forward")
    check_match(pt_bn.weight.grad, sr_bn.gamma.node.cp, "BN1d Gamma Grad")
    check_match(pt_bn.bias.grad, sr_bn.beta.node.cp, "BN1d Beta Grad")

def test_07_batchnorm2d():
    print("\n--- Test 7: BatchNorm2d Match ---")
    N, C, H, W = 8, 16, 16, 16
    x_np = np.random.randn(N, C, H, W).astype(np.float32)
    
    pt_x = torch.tensor(x_np, requires_grad=True)
    pt_bn = nn.BatchNorm2d(C, momentum=0.1, eps=1e-5)
    pt_out = pt_bn(pt_x)
    pt_out.sum().backward()
    
    sr_x = Tensor(x_np, is_leaf=True, device="cuda")
    sr_bn = BatchNorm2d(C, momentum=0.1, eps=1e-5)
    sr_bn.inpobj = type('obj', (object,), {'outputs': sr_x})
    sr_out = sr_bn.forward()
    autograd4nn(sr_out.sum())
    
    check_match(pt_out, sr_out, "BatchNorm2d Forward")

def test_08_activations_vjp():
    print("\n--- Test 8: Activations (ReLU/Softmax) Match ---")
    x_np = np.random.randn(10, 10).astype(np.float32)
    
    # Softmax
    pt_x = torch.tensor(x_np, requires_grad=True)
    pt_out = F.softmax(pt_x, dim=-1)
    (pt_out * torch.ones_like(pt_out)).sum().backward()
    
    sr_x = Tensor(x_np, is_leaf=True, device="cuda")
    sr_out = sr_x.softmax()
    autograd4nn((sr_out * Tensor.ones(sr_out.shape, device="cuda")).sum())
    
    check_match(pt_out, sr_out, "Softmax Forward")
    check_match(pt_x.grad, sr_x.node.cp, "Softmax VJP Grad")

def test_09_flatten_concat():
    print("\n--- Test 9: Flatten & Concatenate Graph Routing ---")
    x1_np = np.random.randn(4, 3, 8, 8).astype(np.float32)
    x2_np = np.random.randn(4, 3, 8, 8).astype(np.float32)
    
    pt_x1 = torch.tensor(x1_np, requires_grad=True)
    pt_x2 = torch.tensor(x2_np, requires_grad=True)
    pt_cat = torch.cat([pt_x1, pt_x2], dim=1)
    pt_flat = pt_cat.view(4, -1)
    pt_flat.sum().backward()
    
    sr_x1 = Tensor(x1_np, is_leaf=True, device="cuda")
    sr_x2 = Tensor(x2_np, is_leaf=True, device="cuda")
    sr_cat = sr_x1.concatenete(sr_x2)
    sr_flat = sr_cat.flatten()
    autograd4nn(sr_flat.sum())
    
    check_match(pt_x1.grad, sr_x1.node.cp, "Concat+Flatten Input 1 Grad")
    check_match(pt_x2.grad, sr_x2.node.cp, "Concat+Flatten Input 2 Grad")

def test_10_cce_loss():
    print("\n--- Test 10: Categorical Cross Entropy Match ---")
    # PyTorch uses logits for CrossEntropyLoss, Seera uses post-softmax probs. 
    # To compare, we pass PyTorch log_softmax, and Seera explicit math.
    N, C = 32, 10
    logits_np = np.random.randn(N, C).astype(np.float32)
    y_np = np.eye(C)[np.random.choice(C, N)].astype(np.float32) # One hot
    
    pt_logits = torch.tensor(logits_np, requires_grad=True)
    pt_y = torch.tensor(y_np)
    pt_loss = -torch.sum(pt_y * F.log_softmax(pt_logits, dim=-1), dim=-1).mean()
    pt_loss.backward()
    
    sr_logits = Tensor(logits_np, is_leaf=True, device="cuda")
    sr_y = Tensor(y_np, device="cuda")
    sr_probs = sr_logits.softmax()
    sr_loss = Loss().categorical_cross_entropy(sr_probs, sr_y)
    autograd4nn(sr_loss)
    
    check_match(pt_loss, sr_loss.value, "CCE Loss Forward")
    check_match(pt_logits.grad, sr_logits.node.cp, "CCE Loss Input Grad")

def test_11_sgd_momentum():
    print("\n--- Test 11: SGD Momentum Optimizer Match ---")
    x_np = np.random.randn(10, 10).astype(np.float32)
    
    pt_x = torch.tensor(x_np, requires_grad=True)
    pt_dense = nn.Linear(10, 5)
    pt_opt = torch.optim.SGD(pt_dense.parameters(), lr=0.1, momentum=0.9)
    
    sr_dense = Dense(10, 5, activation="relu")
    sync_dense(pt_dense, sr_dense)
    sr_model = Sequential([Input((10,)), sr_dense], device="cuda")
    sr_opt = SGD(sr_model, lr=0.1, momentum=0.9)
    
    # Step 1
    pt_out = pt_dense(pt_x).sum()
    pt_out.backward()
    pt_opt.step()
    
    sr_x = Tensor(x_np, is_leaf=True, device="cuda")
    sr_out = sr_model.forward(sr_x).sum()
    autograd4nn(sr_out)
    sr_opt.step()
    
    check_match(pt_dense.weight.T, sr_dense.weights, "SGD Weights after 1 Step")

def test_12_adam_moments():
    print("\n--- Test 12: Adam Optimizer Moments Match ---")
    # Same setup as 11, but Adam
    x_np = np.random.randn(10, 10).astype(np.float32)
    pt_x = torch.tensor(x_np, requires_grad=True)
    pt_dense = nn.Linear(10, 5)
    pt_opt = torch.optim.Adam(pt_dense.parameters(), lr=0.01)
    
    sr_dense = Dense(10, 5, activation="relu")
    sync_dense(pt_dense, sr_dense)
    sr_model = Sequential([Input((10,)), sr_dense], device="cuda")
    sr_opt = Adam(sr_model, lr=0.01)
    
    pt_dense(pt_x).sum().backward()
    pt_opt.step()
    
    sr_x = Tensor(x_np, is_leaf=True, device="cuda")
    autograd4nn(sr_model.forward(sr_x).sum())
    sr_opt.step()
    
    # Adam has complex internal states, checking final weights is best proxy
    check_match(pt_dense.weight.T, sr_dense.weights, "Adam Weights after 1 Step", atol=1e-3)

def test_13_large_mlp_integration():
    print("\n--- Test 13: Large Deep MLP Integration ---")
    x_np = np.random.randn(64, 512).astype(np.float32)
    
    pt_m = nn.Sequential(
        nn.Linear(512, 1024), nn.ReLU(),
        nn.Linear(1024, 256), nn.ReLU(),
        nn.Linear(256, 10)
    )
    pt_x = torch.tensor(x_np, requires_grad=True)
    pt_out = pt_m(pt_x)
    pt_out.sum().backward()
    
    sr_m = Sequential([
        Input((512,)),
        Dense(512, 1024, "relu"),
        Dense(1024, 256, "relu"),
        Dense(256, 10, "softmax") # Using softmax to differentiate, compare pre-softmax if needed
    ], device="cuda")
    
    sync_dense(pt_m[0], sr_m.model[1])
    sync_dense(pt_m[2], sr_m.model[2])
    sync_dense(pt_m[4], sr_m.model[3])
    
    sr_x = Tensor(x_np, is_leaf=True, device="cuda")
    sr_out = sr_m.forward(sr_x) # Outputs softmax
    # For fair grad comparison, let's just backprop sum of pre-softmax (z of last layer)
    autograd4nn(sr_m.model[3].z.sum())
    
    check_match(pt_m[0].weight.grad.T, sr_m.model[1].weights.node.cp, "Deep MLP First Layer Grad")

def test_14_cnn_block_integration():
    print("\n--- Test 14: Deep CNN Block (Conv->BN->ReLU->Pool) ---")
    x_np = np.random.randn(8, 3, 64, 64).astype(np.float32)
    
    pt_x = torch.tensor(x_np, requires_grad=True)
    pt_conv = nn.Conv2d(3, 16, 3, padding=1)
    pt_bn = nn.BatchNorm2d(16)
    pt_pool = nn.MaxPool2d(2)
    
    pt_out = pt_pool(F.relu(pt_bn(pt_conv(pt_x))))
    pt_out.sum().backward()
    
    sr_x = Tensor(x_np, is_leaf=True, device="cuda")
    sr_m = Sequential([
        Input((3, 64, 64)),
        Conv2D(16, 3, (3,3), "relu", padding=1),
        BatchNorm2d(16),
        MaxPool2D((2,2))
    ], device="cuda")
    
    sync_conv2d(pt_conv, sr_m.model[1])
    # Sync BN
    sr_m.model[2].gamma.value = cuten(pt_bn.weight.detach().cpu().numpy())
    sr_m.model[2].beta.value = cuten(pt_bn.bias.detach().cpu().numpy())
    
    sr_out = sr_m.forward(sr_x)
    autograd4nn(sr_out.sum())
    
    check_match(pt_conv.weight.grad, sr_m.model[1].weights.node.cp, "CNN Block Conv Grad")

def test_15_end_to_end_loss_curve():
    print("\n--- Test 15: End-to-End Overfit Loss Curve Sync ---")
    # If this matches perfectly over 10 iterations, your framework is mathematically identical to PyTorch.
    np.random.seed(42)
    torch.manual_seed(42)
    
    X_data = np.random.randn(16, 128).astype(np.float32)
    Y_data = np.eye(10)[np.random.choice(10, 16)].astype(np.float32)
    
    pt_m = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    pt_opt = torch.optim.SGD(pt_m.parameters(), lr=0.01)
    pt_x = torch.tensor(X_data)
    pt_y = torch.tensor(Y_data)
    
    sr_m = Sequential([Input((128,)), Dense(128, 64, "relu"), Dense(64, 10, "softmax")], device="cuda")
    sync_dense(pt_m[0], sr_m.model[1])
    sync_dense(pt_m[2], sr_m.model[2])
    sr_opt = SGD(sr_m, lr=0.01)
    
    for i in range(10):
        # PyTorch
        pt_opt.zero_grad()
        pt_logits = pt_m(pt_x)
        pt_loss = -torch.sum(pt_y * F.log_softmax(pt_logits, dim=-1), dim=-1).mean()
        pt_loss.backward()
        pt_opt.step()
        
        # Seera
        sr_x = Tensor(X_data, is_leaf=True, device="cuda")
        sr_y = Tensor(Y_data, device="cuda")
        sr_m.zero_grad()
        sr_probs = sr_m.forward(sr_x)
        sr_loss = Loss().categorical_cross_entropy(sr_probs, sr_y)
        autograd4nn(sr_loss)
        sr_opt.step()
        
        diff = np.abs(pt_loss.item() - to_numpy(sr_loss.value)[0])
        print(f"Iter {i}: PT Loss {pt_loss.item():.4f} | Seera Loss {to_numpy(sr_loss.value)[0]:.4f} | Diff: {diff:.6f}")
        if diff > 1e-4:
            print("❌ CATASTROPHIC DIVERGENCE IN TRAINING LOOP.")
            return
            
    print("✅ PASSED: Loss curves match perfectly over 10 iterations.")


if __name__ == "__main__":
    print("=====================================================")
    print("🚀 SEERA VS PYTORCH: EXTENSIVE CUDA DIAGNOSTICS")
    print("=====================================================")
    test_01_large_dense_fp_bp()
    test_02_large_conv2d_fp_bp()
    test_03_conv_transpose2d()
    test_04_maxpool2d()
    test_05_unpool_nearest()
    test_06_batchnorm1d()
    test_07_batchnorm2d()
    test_08_activations_vjp()
    test_09_flatten_concat()
    test_10_cce_loss()
    test_11_sgd_momentum()
    test_12_adam_moments()
    test_13_large_mlp_integration()
    test_14_cnn_block_integration()
    test_15_end_to_end_loss_curve()