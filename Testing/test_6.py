
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from Seera import Sequential, Input, Dense, Conv2D, Flatten, Loss, SGD, Adam
from Seera_init import tensor as Tensor
from Seera_Engine import autograd4nn
from cuTen import cuten
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
def check_tensor_diff(t_before, t_after, name="Tensor"):
    """Helper to check if a tensor's values actually changed on the GPU."""
    diff = np.abs(t_before - t_after).sum()
    if diff == 0:
        print(f"❌ FAILED: {name} did not change. Diff: {diff}")
    else:
        print(f"✅ PASSED: {name} updated successfully. Diff: {diff:.6f}")

def test_1_cuda_autograd_flow():
    print("\n--- Test 1: Basic CUDA Autograd Flow ---")
    try:
        # Create leaf tensors on CUDA
        x = Tensor.randn(2, 3, device="cuda")
        w = Tensor.randn(3, 2, device="cuda")
        
        # Forward pass
        y = x.matmul(w)
        loss = y.sum()
        
        # Backward pass
        autograd4nn(loss)
        
        # Verify gradients exist and are cuten objects
        grad_w = w.node.cp
        if isinstance(grad_w, cuten):
            grad_sum = np.abs(grad_w.to_host_f32()).sum()
            if grad_sum > 0:
                print(f"✅ PASSED: Basic Matmul Autograd. Gradient sum: {grad_sum:.4f}")
            else:
                print("❌ FAILED: Gradients are zero.")
        else:
            print(f"❌ FAILED: Gradient is not a cuten object. Type: {type(grad_w)}")
    except Exception as e:
        print(f"❌ FAILED with exception: {e}")

def test_2_dense_layer_gradients():
    print("\n--- Test 2: Dense Layer Backward Pass ---")
    try:
        model = Sequential([
            Input((4,)),
            Dense(4, 2, activation="relu")
        ], device="cuda")
        
        x = Tensor.randn(1, 4, device="cuda")
        out = model.forward(x)
        loss = out.sum()
        
        model.zero_grad()
        autograd4nn(loss)
        
        dense_layer = model.model[1]
        w_grad = dense_layer.weights.node.cp
        
        if isinstance(w_grad, cuten) and np.abs(w_grad.to_host_f32()).sum() > 0:
             print("✅ PASSED: Dense layer received valid gradients.")
        else:
             print("❌ FAILED: Dense layer gradients are zero or not cuten.")
    except Exception as e:
        print(f"❌ FAILED with exception: {e}")

def test_3_optimizer_updates():
    print("\n--- Test 3: Optimizer Weight Updates (SGD & Adam) ---")
    try:
        model = Sequential([
            Input((4,)),
            Dense(4, 2, activation="relu")
        ], device="cuda")
        
        opt = SGD(model, lr=0.1)
        dense_layer = model.model[1]
        
        # Save pre-update weights
        w_initial = dense_layer.weights.to_numpy()
        
        x = Tensor.randn(1, 4, device="cuda")
        out = model.forward(x)
        loss = out.sum()
        
        model.zero_grad()
        autograd4nn(loss)
        opt.step()
        
        w_updated = dense_layer.weights.to_numpy()
        check_tensor_diff(w_initial, w_updated, "SGD Weight Update")
        
    except Exception as e:
        print(f"❌ FAILED with exception: {e}")

def test_4_single_batch_overfit():
    print("\n--- Test 4: Single Batch Overfit (Sanity Check) ---")
    # If the model cannot overfit a single batch of 2 samples, 
    # there is a fundamental math error in the framework.
    try:
        model = Sequential([
            Input((4,)),
            Dense(4, 8, activation="relu"),
            Dense(8, 2, activation="softmax")
        ], device="cuda")
        
        opt = Adam(model, lr=0.05)
        loss_fn = Loss().categorical_cross_entropy
        
        # Create a static dataset of 2 samples
        X_data = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
        # One-hot encoded targets
        Y_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        
        X_tensor = Tensor(X_data, is_leaf=True, device="cuda")
        Y_tensor = Tensor(Y_data, device="cuda")
        
        initial_loss = None
        final_loss = None
        
        print("Training for 50 epochs on a single batch...")
        for epoch in range(50):
            preds = model.forward(X_tensor)
            loss = loss_fn(preds, Y_tensor)
            
            # Extract scalar loss
            loss_val = float(loss.value.to_host_f32().ravel()[0])
            if epoch == 0:
                initial_loss = loss_val
            final_loss = loss_val
            model.zero_grad()
            autograd4nn(loss)
            opt.step()
            
        print(f"Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
        if final_loss < initial_loss * 0.1:
            print("✅ PASSED: Model successfully overfit the batch.")
        else:
            print("❌ FAILED: Model failed to overfit. Loss did not decrease sufficiently.")
            
    except Exception as e:
        print(f"❌ FAILED with exception: {e}")

if __name__ == "__main__":
    print("Starting CUDA Deep Learning Framework Diagnostics...")
    test_1_cuda_autograd_flow()
    test_2_dense_layer_gradients()
    test_3_optimizer_updates()
    test_4_single_batch_overfit()