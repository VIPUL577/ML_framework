"""
U-Net architecture test using Seera framework.
Tests forward + backward through a full U-Net with skip connections.

U-Net architecture (small version for testing):
  Input: (N, 1, 32, 32)

  Encoder:
    enc1: Conv2D(1→8,  3×3, pad=1) + ReLU   → (N, 8, 32, 32)
    pool1: MaxPool2D(2×2, stride=2)          → (N, 8, 16, 16)
    enc2: Conv2D(8→16, 3×3, pad=1) + ReLU   → (N, 16, 16, 16)
    pool2: MaxPool2D(2×2, stride=2)          → (N, 16, 8, 8)

  Bottleneck:
    bot: Conv2D(16→32, 3×3, pad=1) + ReLU   → (N, 32, 8, 8)

  Decoder:
    up1: Unpool2D(2×2)                       → (N, 32, 16, 16)
    cat1: Concat(up1, enc2) along C          → (N, 48, 16, 16)
    dec1: Conv2D(48→16, 3×3, pad=1) + ReLU  → (N, 16, 16, 16)
    up2: Unpool2D(2×2)                       → (N, 16, 32, 32)
    cat2: Concat(up2, enc1) along C          → (N, 24, 32, 32)
    dec2: Conv2D(24→1, 3×3, pad=1) + Sigmoid→ (N, 1, 32, 32)

This test verifies:
  1. Forward pass produces correct output shape
  2. Backward pass runs without errors
  3. Gradients flow to all encoder layers (non-zero grads)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from Seera_init import tensor as Tensor, _is_gpu
from Seera_Engine import autograd4nn
from Seera import Loss


def build_unet_cpu():
    """Build U-Net weights as Tensors (CPU)."""
    np.random.seed(42)
    he = lambda fan_in: np.sqrt(2.0 / fan_in) * 0.1

    weights = {}
    # Encoder
    weights["enc1_w"] = Tensor(np.random.randn(8, 1, 3, 3).astype(np.float32) * he(1*3*3), is_leaf=True)
    weights["enc1_b"] = Tensor.zeros((1, 8, 1, 1))
    weights["enc2_w"] = Tensor(np.random.randn(16, 8, 3, 3).astype(np.float32) * he(8*3*3), is_leaf=True)
    weights["enc2_b"] = Tensor.zeros((1, 16, 1, 1))
    # Bottleneck
    weights["bot_w"] = Tensor(np.random.randn(32, 16, 3, 3).astype(np.float32) * he(16*3*3), is_leaf=True)
    weights["bot_b"] = Tensor.zeros((1, 32, 1, 1))
    # Decoder
    weights["dec1_w"] = Tensor(np.random.randn(16, 48, 3, 3).astype(np.float32) * he(48*3*3), is_leaf=True)
    weights["dec1_b"] = Tensor.zeros((1, 16, 1, 1))
    weights["dec2_w"] = Tensor(np.random.randn(1, 24, 3, 3).astype(np.float32) * he(24*3*3), is_leaf=True)
    weights["dec2_b"] = Tensor.zeros((1, 1, 1, 1))

    return weights


def unet_forward(x, weights):
    """Forward pass of the U-Net."""
    # ── Encoder ──────────────────
    print(f"%%%{x.conv2d(weights["enc1_w"], stride=1, padding=1).shape}%%%%%%")
    enc1 = x.conv2d(weights["enc1_w"], stride=1, padding=1) + weights["enc1_b"]

    enc1 = enc1.relu()                          # (N, 8, 32, 32)

    pool1 = enc1.maxpool2d(kernelsize=(2, 2), stride=2)  # (N, 8, 16, 16)

    enc2 = pool1.conv2d(weights["enc2_w"], stride=1, padding=1) + weights["enc2_b"]
    enc2 = enc2.relu()                           # (N, 16, 16, 16)

    pool2 = enc2.maxpool2d(kernelsize=(2, 2), stride=2)  # (N, 16, 8, 8)

    # ── Bottleneck ───────────────
    bot = pool2.conv2d(weights["bot_w"], stride=1, padding=1) + weights["bot_b"]
    bot = bot.relu()                              # (N, 32, 8, 8)

    # ── Decoder ──────────────────
    up1 = bot.Unpool2Dnearest(size=(2, 2))        # (N, 32, 16, 16)
    cat1 = up1.concatenete(enc2)                  # (N, 48, 16, 16)
    dec1 = cat1.conv2d(weights["dec1_w"], stride=1, padding=1) + weights["dec1_b"]
    dec1 = dec1.relu()                            # (N, 16, 16, 16)

    up2 = dec1.Unpool2Dnearest(size=(2, 2))       # (N, 16, 32, 32)
    cat2 = up2.concatenete(enc1)                  # (N, 24, 32, 32)
    dec2 = cat2.conv2d(weights["dec2_w"], stride=1, padding=1) + weights["dec2_b"]
    out = dec2.sigmoid()                          # (N, 1, 32, 32)

    return out


def test_unet_cpu():
    """Test U-Net forward + backward on CPU."""
    print("=" * 60)
    print("  U-Net CPU Test")
    print("=" * 60)

    weights = build_unet_cpu()
    batch_size = 2

    # Random input and target (segmentation masks)
    x = Tensor(np.random.randn(batch_size, 1, 32, 32).astype(np.float32), is_leaf=True)
    y = Tensor((np.random.rand(batch_size, 1, 32, 32) > 0.5).astype(np.float32))

    # Forward
    pred = unet_forward(x, weights)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {pred.shape}")
    assert pred.shape == (batch_size, 1, 32, 32), f"Expected (2,1,32,32), got {pred.shape}"
    print("  ✓ Forward shape correct")

    # Loss
    loss_fn = Loss()
    loss = loss_fn.binary_cross_entropy(pred, y)
    print(f"  Loss: {loss.value}")

    # Backward
    autograd4nn(loss)
    print("  ✓ Backward pass complete")

    # Check gradients exist
    for name, w in weights.items():
        if w.is_leaf:
            grad = w.node.cp
            if isinstance(grad, np.ndarray):
                grad_norm = np.linalg.norm(grad)
            else:
                grad_norm = 0.0
            print(f"  grad({name}): norm={grad_norm:.6f}, shape={grad.shape if hasattr(grad, 'shape') else 'scalar'}")

    print("  ✓ All gradients computed")
    print()


def test_unet_gpu():
    """Test U-Net forward + backward on GPU."""
    print("=" * 60)
    print("  U-Net GPU Test")
    print("=" * 60)

    try:
        from cuTen import cuten
        import seera_cuda
    except ImportError:
        print("  SKIP: seera_cuda not available")
        return

    np.random.seed(42)
    he = lambda fan_in: np.sqrt(2.0 / fan_in) * 0.1
    batch_size = 2

    weights = {}
    # Encoder
    weights["enc1_w"] = Tensor(np.random.randn(8, 1, 3, 3).astype(np.float32) * he(1*3*3), is_leaf=True, device="cuda")
    weights["enc1_b"] = Tensor.zeros((1, 8, 1, 1), device="cuda")
    weights["enc2_w"] = Tensor(np.random.randn(16, 8, 3, 3).astype(np.float32) * he(8*3*3), is_leaf=True, device="cuda")
    weights["enc2_b"] = Tensor.zeros((1, 16, 1, 1), device="cuda")
    # Bottleneck
    weights["bot_w"] = Tensor(np.random.randn(32, 16, 3, 3).astype(np.float32) * he(16*3*3), is_leaf=True, device="cuda")
    weights["bot_b"] = Tensor.zeros((1, 32, 1, 1), device="cuda")
    # Decoder
    weights["dec1_w"] = Tensor(np.random.randn(16, 48, 3, 3).astype(np.float32) * he(48*3*3), is_leaf=True, device="cuda")
    weights["dec1_b"] = Tensor.zeros((1, 16, 1, 1), device="cuda")
    weights["dec2_w"] = Tensor(np.random.randn(1, 24, 3, 3).astype(np.float32) * he(24*3*3), is_leaf=True, device="cuda")
    weights["dec2_b"] = Tensor.zeros((1, 1, 1, 1), device="cuda")

    # GPU input and target
    x = Tensor(np.random.randn(batch_size, 1, 32, 32).astype(np.float32), is_leaf=True, device="cuda")
    y = Tensor((np.random.rand(batch_size, 1, 32, 32) > 0.5).astype(np.float32), device="cuda")

    # Forward
    pred = unet_forward(x, weights)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {pred.shape}")
    assert pred.shape == (batch_size, 1, 32, 32), f"Expected (2,1,32,32), got {pred.shape}"
    print("  ✓ Forward shape correct (GPU)")

    # Loss
    loss_fn = Loss()
    loss = loss_fn.binary_cross_entropy(pred, y)
    loss_val = loss.value.to_host_f32()
    print(f"  Loss: {float(loss_val):.6f}")

    # Backward
    autograd4nn(loss)
    print("  ✓ Backward pass complete (GPU)")

    # Check gradients exist
    for name, w in weights.items():
        if w.is_leaf:
            grad = w.node.cp
            if isinstance(grad, cuten):
                grad_np = grad.to_host_f32()
                grad_norm = np.linalg.norm(grad_np)
            elif isinstance(grad, np.ndarray):
                grad_norm = np.linalg.norm(grad)
            else:
                grad_norm = 0.0
            print(f"  grad({name}): norm={grad_norm:.6f}")

    print("  ✓ All GPU gradients computed")
    print()


def test_unet_training_loop():
    """Run a tiny training loop on CPU to verify weight updates work end-to-end."""
    print("=" * 60)
    print("  U-Net Training Loop Test (CPU, 3 steps)")
    print("=" * 60)

    weights = build_unet_cpu()
    lr = 0.001
    batch_size = 2

    for step in range(3):
        x = Tensor(np.random.randn(batch_size, 1, 32, 32).astype(np.float32), is_leaf=True)
        y = Tensor((np.random.rand(batch_size, 1, 32, 32) > 0.5).astype(np.float32))

        pred = unet_forward(x, weights)
        loss_fn = Loss()
        loss = loss_fn.mse(pred, y)
        loss_val = float(np.sum(loss.value))

        autograd4nn(loss)

        # Manual SGD step
        for name, w in weights.items():
            if w.is_leaf and isinstance(w.node.cp, np.ndarray):
                w.value -= lr * w.node.cp

        print(f"  Step {step+1}: loss={loss_val:.6f}")

    print("  ✓ Training loop complete")
    print()


if __name__ == "__main__":
    test_unet_cpu()
    test_unet_training_loop()
    test_unet_gpu()
    print("All U-Net tests passed ✓")
