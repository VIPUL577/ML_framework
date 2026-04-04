import numpy as np

try:
    from numba import njit, jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ─────────────────────────────────────────────────────────────
# Node: stores gradient info and children in the computation graph
# ─────────────────────────────────────────────────────────────
class node:
    def __init__(self, child_grad, out=0, node_no=0):
        self.out = out
        self.child_grad = np.array(child_grad, dtype=object)
        if isinstance(child_grad, list) and len(child_grad) > 0 and len(child_grad[0]) > 0:
            self.cp = np.zeros_like(child_grad[0][0])
        else:
            self.cp = 0
        self.child = []

    @property
    def grad(self):
        return self.cp


# ─────────────────────────────────────────────────────────────
# Tensor: core data structure with autograd support
# Convention: batch dimension is axis 0 for all operations
#   Dense:  (N, features)
#   Conv:   (N, C, H, W)
# ─────────────────────────────────────────────────────────────
class tensor(node):
    def __init__(self, value, dtype="float32", is_leaf=False):
        self.value = np.array(value).astype(dtype)
        child_grad = (
            [np.zeros_like(self.value), np.zeros_like(self.value)],
            [np.zeros_like(self.value), np.zeros_like(self.value)],
        )
        self.node = node(child_grad)
        self.is_leaf = is_leaf
        self.dtype = dtype

        # ── special backward context flags ──
        self.matm = False
        self.isoftmax = False
        self.iconv2d = (False, 1, 0)
        self.upctx = (False, 0, 0)
        self.flctx = 0          # flatten context (original shape)
        self.mpctx = (False, 0, 0, 0, 1, 0)
        self.iconcatenete = 0
        self.ibatchnorm = None  # BatchNorm context (set during forward)
        self.ireduction = None  # Reduction context: {"axis", "keepdims", "scale"}
        self.convTrans = False

        if is_leaf:
            self.node.out = self.value

    # ─── Helpers ─────────────────────────────────────────────
    @staticmethod
    def _reduce_broadcast(grad, target_shape):
        """Sum-reduce *grad* along any axes that were broadcast to match
        *target_shape*.  E.g. grad (N, out) → bias (1, out) sums axis 0."""
        if grad.shape == target_shape:
            return grad
        # Pad target_shape on the left with 1s so ndims match
        ndim_diff = grad.ndim - len(target_shape)
        padded = (1,) * ndim_diff + tuple(target_shape)
        # Sum along every axis where padded has size 1
        reduce_axes = tuple(i for i, s in enumerate(padded) if s == 1)
        out = grad.sum(axis=reduce_axes, keepdims=True)
        # Remove any extra leading dims we added
        if ndim_diff > 0:
            out = out.reshape(target_shape)
        return out

    # ─── Element-wise ops ────────────────────────────────────
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = tensor(other * np.ones_like(self.value))
        elif not isinstance(other, tensor):
            other = tensor(other)
        out = tensor(self.value + other.value)
        child_grad = [
            [self.value, np.ones_like(self.value)],
            [other.value, np.ones_like(other.value)],
        ]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        out.node.child = [self, other]
        return out

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = tensor(other * np.ones_like(self.value))
        elif not isinstance(other, tensor):
            other = tensor(other)
        out = tensor(self.value * other.value)
        child_grad = [
            [self.value, other.value],
            [other.value, self.value],
        ]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        out.node.child = [self, other]
        return out

    def __pow__(self, other):
        if not isinstance(other, float):
            other = float(other)
        out = tensor(self.value ** other)
        gradient = other * (self.value ** (other - 1))
        child_grad = [
            [self.value, gradient],
            [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)],
        ]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        out.node.child = [self]
        return out

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (self * -1)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    # ─── Unary math ops ──────────────────────────────────────
    def _unary(self, fwd_val, gradient):
        """Helper to build a unary op node."""
        out = tensor(fwd_val)
        child_grad = [
            [self.value, gradient],
            [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)],
        ]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        out.node.child = [self]
        return out

    def sin(self):
        return self._unary(np.sin(self.value), np.cos(self.value))

    def cos(self):
        return self._unary(np.cos(self.value), -np.sin(self.value))

    def tan(self):
        return self._unary(np.tan(self.value), 1 / (np.cos(self.value) ** 2))

    def tanh(self):
        return self._unary(np.tanh(self.value), 1 - np.tanh(self.value) ** 2)

    def relu(self):
        return self._unary(
            np.where(self.value > 0, self.value, 0),
            np.where(self.value > 0, 1.0, 0.0),
        )

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.value))
        return self._unary(s, s * (1 - s))

    def log(self):
        return self._unary(np.log(self.value), 1 / self.value)

    def exp(self):
        return self._unary(np.exp(self.value), np.exp(self.value))

    def abs(self):
        return self._unary(np.abs(self.value), np.sign(self.value))

    def sqrt(self):
        return self._unary(np.sqrt(self.value), 0.5 / np.sqrt(self.value))

    def clip(self, min_val, max_val):
        gradient = np.ones_like(self.value)
        gradient[self.value < min_val] = 0
        gradient[self.value > max_val] = 0
        return self._unary(np.clip(self.value, min_val, max_val), gradient)

    # ─── Softmax (batched, VJP-based) ────────────────────────
    def softmax(self, axis=-1):
        """Batched softmax along *axis* (default: last axis).
        Stores the softmax output for efficient VJP backward."""
        shifted = self.value - np.max(self.value, axis=axis, keepdims=True)
        exps = np.exp(shifted)
        s = exps / np.sum(exps, axis=axis, keepdims=True)

        out = tensor(s)
        out.isoftmax = True
        # Store softmax output and axis in child_grad for backward
        child_grad = np.empty((1, 2), dtype=object)
        child_grad[0, 0] = self.value
        child_grad[0, 1] = s  # softmax output (not Jacobian)
        out.node.child_grad = child_grad
        out.node.out = out.value
        out.node.child = [self]
        return out

    # ─── Matmul (batched) ────────────────────────────────────
    def matmul(self, other):
        """Matrix multiply: (N, in) @ (in, out) → (N, out)
        Also handles non-batched cases via standard np matmul."""
        out = tensor(self.value @ other.value)
        out.node.child = [self, other]
        out.matm = True
        out.node.out = out.value
        child_grad_obj = np.empty((2, 2), dtype=object)
        child_grad_obj[0, 0] = self.value
        child_grad_obj[0, 1] = other.value
        child_grad_obj[1, 0] = other.value
        child_grad_obj[1, 1] = self.value
        out.node.child_grad = child_grad_obj
        return out

    # ─── Reductions ──────────────────────────────────────────
    def sum(self, axis=None, keepdims=False):
        s = np.sum(self.value, axis=axis, keepdims=keepdims)
        out = tensor(s)
        out.node.out = out.value
        out.node.child = [self]
        # Store reduction context for backward
        out.ireduction = {
            "input_shape": self.value.shape,
            "axis": axis,
            "keepdims": keepdims,
            "scale": 1.0,  # sum gradient is 1
        }
        return out

    def mean(self, axis=None, keepdims=False):
        out = tensor(np.mean(self.value, axis=axis, keepdims=keepdims))
        out.node.child = [self]
        out.node.out = out.value
        if axis is None:
            n = self.value.size
        elif isinstance(axis, (tuple, list)):
            n = 1
            for ax in axis:
                n *= self.value.shape[ax]
        else:
            n = self.value.shape[axis]
        out.ireduction = {
            "input_shape": self.value.shape,
            "axis": axis,
            "keepdims": keepdims,
            "scale": 1.0 / n,  # mean gradient is 1/N
        }
        return out

    def max(self, axis=None, keepdims=False):
        out_value = np.max(self.value, axis=axis, keepdims=keepdims)
        out = tensor(out_value)
        out.node.child = [self]
        if axis is None:
            gradient = np.where(self.value == np.max(self.value), 1.0, 0.0)
        else:
            gradient = np.zeros_like(self.value)
            max_indices = np.argmax(self.value, axis=axis)
            gradient.flat[max_indices] = 1.0
        child_grad = [
            [self.value, gradient],
            [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)],
        ]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        return out

    def min(self, axis=None, keepdims=False):
        out_value = np.min(self.value, axis=axis, keepdims=keepdims)
        out = tensor(out_value)
        out.node.child = [self]
        if axis is None:
            gradient = np.where(self.value == np.min(self.value), 1.0, 0.0)
        else:
            gradient = np.zeros_like(self.value)
            min_indices = np.argmin(self.value, axis=axis)
            gradient.flat[min_indices] = 1.0
        child_grad = [
            [self.value, gradient],
            [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)],
        ]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        return out

    # ─── Shape ops ───────────────────────────────────────────
    @property
    def shape(self):
        return self.value.shape

    def T(self):
        transposed = self.value.T
        out = tensor(transposed, dtype=self.dtype, is_leaf=self.is_leaf)
        self.node = out.node
        self.matm = out.matm
        return out

    def squeeze(self, axis=None):
        out = tensor(np.squeeze(self.value, axis=axis))
        out.node.child = [self]
        gradient = np.ones_like(self.value)
        child_grad = [
            [self.value, gradient],
            [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)],
        ]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        return out

    def unsqueeze(self, axis):
        out = tensor(np.expand_dims(self.value, axis=axis))
        out.node.child = [self]
        gradient = np.ones_like(self.value)
        child_grad = [
            [self.value, gradient],
            [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)],
        ]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        return out

    def flatten(self):
        """Flatten preserving batch dimension: (N, ...) → (N, flat_size)"""
        original_shape = self.value.shape
        N = original_shape[0]
        out = tensor(self.value.reshape(N, -1))
        out.flctx = original_shape  # saved for backward reshape
        out.node.child = [self]
        return out

    def __getitem__(self, key):
        out = tensor(self.value[key])
        out.node.child = [self]
        gradient = np.zeros_like(self.value)
        sliced_positions = np.zeros_like(self.value, dtype=bool)
        sliced_positions[key] = True
        gradient[sliced_positions] = 1.0
        child_grad = [
            [self.value, gradient],
            [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)],
        ]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        return out

    # ─── im2col / col2im  (pure NumPy, batched) ─────────────
    @staticmethod
    def im2col_batch(X, KH, KW, stride=1, pad=0):
        """Vectorized im2col for batched input.
        X: (N, C, H, W)  →  cols: (N, C*KH*KW, OH*OW)
        """
        N, C, H, W = X.shape
        OH = (H + 2 * pad - KH) // stride + 1
        OW = (W + 2 * pad - KW) // stride + 1

        if pad > 0:
            X = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

        # Use stride tricks for zero-copy view
        _, _, Hp, Wp = X.shape
        s_n, s_c, s_h, s_w = X.strides
        shape = (N, C, KH, KW, OH, OW)
        strides = (s_n, s_c, s_h * stride, s_w * stride, s_h, s_w)
        # Wait — stride tricks for im2col needs the *inner* patch strides
        # to step by 1, and the *outer* window position to step by stride.
        # shape: (N, C, OH, OW, KH, KW)
        shape = (N, C, OH, OW, KH, KW)
        strides = (s_n, s_c, s_h * stride, s_w * stride, s_h, s_w)
        col = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        # col: (N, C, OH, OW, KH, KW) → (N, C*KH*KW, OH*OW)
        col = col.reshape(N, C * KH * KW, OH * OW)
        # Make contiguous copy (stride tricks creates a view which may cause
        # issues with later operations)
        return np.ascontiguousarray(col)

    @staticmethod
    def col2im_batch(cols, X_shape, KH, KW, stride=1, pad=0):
        """Inverse of im2col_batch.  Accumulates overlapping patches.
        cols: (N, C*KH*KW, OH*OW)  →  X: (N, C, H, W)
        """
        N, C, H, W = X_shape
        OH = (H + 2 * pad - KH) // stride + 1
        OW = (W + 2 * pad - KW) // stride + 1

        Hp, Wp = H + 2 * pad, W + 2 * pad
        X_padded = np.zeros((N, C, Hp, Wp), dtype=cols.dtype)

        cols_reshaped = cols.reshape(N, C, KH, KW, OH, OW)
        for i in range(KH):
            i_end = i + stride * OH
            for j in range(KW):
                j_end = j + stride * OW
                X_padded[:, :, i:i_end:stride, j:j_end:stride] += cols_reshaped[:, :, i, j, :, :]

        if pad > 0:
            return X_padded[:, :, pad:-pad, pad:-pad]
        return X_padded

    # ─── Conv2D forward (batched) ────────────────────────────
    def conv2d(self, W, stride=1, padding=0):
        """Batched conv2d: self (N, C, H, W)  *  W (F, C, KH, KW)
        → out (N, F, OH, OW)"""
        x = self.value
        if x.ndim == 3:
            # Single sample: add batch dim
            x = x[np.newaxis]
        N, C, H, W_in = x.shape
        F, _, KH, KW = W.value.shape

        OH = (H + 2 * padding - KH) // stride + 1
        OW = (W_in + 2 * padding - KW) // stride + 1

        # im2col: (N, C*KH*KW, OH*OW)
        col = tensor.im2col_batch(x.astype(np.float32), KH, KW, stride, padding)
        W_col = W.value.reshape(F, -1)  # (F, C*KH*KW)

        # Batched matmul: (N, F, C*KH*KW) @ (N, C*KH*KW, OH*OW) → (N, F, OH*OW)
        out_val = np.einsum("fc,ncp->nfp", W_col, col)  # broadcast W across batch
        out_val = out_val.reshape(N, F, OH, OW)

        out = tensor(out_val)
        out.iconv2d = (True, stride, padding)
        out.node.out = out.value
        out.node.child = [self, W]
        return out

    # ─── MaxPool2D forward (batched) ─────────────────────────
    def maxpool2d(self, kernelsize, stride=1, padding=0):
        """Batched maxpool: self (N, C, H, W) → out (N, C, OH, OW)"""
        if not isinstance(kernelsize, tuple):
            raise ValueError("kernelsize should be a tuple (height, width)")
        KH, KW = kernelsize
        img = self.value
        if img.ndim == 3:
            img = img[np.newaxis]
        N, C, H, W = img.shape
        OH = (H + 2 * padding - KH) // stride + 1
        OW = (W + 2 * padding - KW) // stride + 1

        # im2col: (N, C*KH*KW, OH*OW)
        col = tensor.im2col_batch(img.astype(np.float32), KH, KW, stride, padding)
        # Reshape to (N, C, KH*KW, OH*OW) to pool over kernel window
        col = col.reshape(N, C, KH * KW, OH * OW)

        mask = np.argmax(col, axis=2)         # (N, C, OH*OW)
        out_val = np.max(col, axis=2)         # (N, C, OH*OW)
        out_val = out_val.reshape(N, C, OH, OW)
        mask = mask.reshape(N, C, OH, OW)

        out = tensor(out_val)
        out.mpctx = (True, mask, self.value.shape, kernelsize, stride, padding)
        out.node.out = out.value
        out.node.child = [self]
        return out

    # ─── Concatenate (batched, along channel axis) ───────────
    def concatenete(self, other):
        """Concatenate along axis=1 (channel dim) for (N, C, H, W) inputs,
        or along axis=0 for 1-D/2-D inputs."""
        if self.value.ndim >= 3 and other.value.ndim >= 3:
            # 4-D batched: concat along channels (axis=1)
            out = tensor(np.concatenate([self.value, other.value], axis=1))
            out.iconcatenete = self.value.shape[1]  # split point
        else:
            out = tensor(np.concatenate([self.value, other.value], axis=0))
            out.iconcatenete = self.value.shape[0]
        out.node.child = [self, other]
        out.node.out = out.value
        return out

    # ─── Upsample2D Nearest (batched) ────────────────────────
    def UpSample2Dnearest(self, size):
        """Nearest-neighbor upsample: (N, C, H, W) → (N, C, H*sh, W*sw)"""
        x = self.value
        if x.ndim == 3:
            x = x[np.newaxis]
        sw, sh = size
        x_up = np.repeat(np.repeat(x, sh, axis=2), sw, axis=3)
        out = tensor(x_up)
        out.node.child = [self]
        out.node.out = out.value
        out.upctx = (True, self.value.shape, size)
        return out

    # ─── BatchNorm forward ───────────────────────────────────
    def batchnorm(self, gamma, beta, running_mean, running_var,
                  training=True, momentum=0.1, eps=1e-5, mode="1d"):
        """BatchNorm forward.
        self: input tensor (N, features) for 1d or (N, C, H, W) for 2d.
        gamma, beta: learnable scale/shift tensors.
        running_mean, running_var: numpy arrays (updated in-place).
        Returns output tensor with ibatchnorm context for backward.
        """
        x = self.value

        if mode == "2d":
            # Normalize over N, H, W (per channel)
            reduce_axes = (0, 2, 3)
            broadcast_shape = (1, -1, 1, 1)
        else:
            # Normalize over N (per feature)
            reduce_axes = (0,)
            broadcast_shape = (1, -1)

        if training:
            mu = np.mean(x, axis=reduce_axes)
            var = np.var(x, axis=reduce_axes)
            # Update running stats (in-place on numpy arrays)
            running_mean[:] = (1 - momentum) * running_mean + momentum * mu
            running_var[:] = (1 - momentum) * running_var + momentum * var
        else:
            mu = running_mean
            var = running_var

        # Normalize
        mu_b = mu.reshape(broadcast_shape)
        var_b = var.reshape(broadcast_shape)
        std_inv = 1.0 / np.sqrt(var_b + eps)
        x_hat = (x - mu_b) * std_inv

        gamma_b = gamma.value.reshape(broadcast_shape)
        beta_b = beta.value.reshape(broadcast_shape)
        y = gamma_b * x_hat + beta_b

        out = tensor(y)
        out.node.child = [self, gamma, beta]
        out.node.out = out.value

        # Store context for backward
        out.ibatchnorm = {
            "x_hat": x_hat,
            "std_inv": std_inv,
            "gamma": gamma.value,
            "reduce_axes": reduce_axes,
            "broadcast_shape": broadcast_shape,
            "N_elements": x.size // gamma.value.size,  # elements per feature
        }
        return out

    # ─── Factory methods ─────────────────────────────────────
    @classmethod
    def zeros(cls, shape, dtype="float32"):
        return cls(np.zeros(shape), dtype=dtype, is_leaf=True)

    @classmethod
    def ones(cls, shape, dtype="float32"):
        return cls(np.ones(shape), dtype=dtype, is_leaf=True)

    @classmethod
    def random(cls, shape, dtype="float32"):
        return cls(np.random.random(shape), dtype=dtype, is_leaf=True)

    @classmethod
    def randn(cls, *shape, dtype="float32"):
        return cls(np.random.randn(*shape), dtype=dtype, is_leaf=True)

    @classmethod
    def eye(cls, n, dtype="float32"):
        return cls(np.eye(n), dtype=dtype, is_leaf=True)

    @classmethod
    def arange(cls, start, stop=None, step=1, dtype="float32"):
        if stop is None:
            stop = start
            start = 0
        return cls(np.arange(start, stop, step), dtype=dtype, is_leaf=True)

    @classmethod
    def linspace(cls, start, stop, num=50, dtype="float32"):
        return cls(np.linspace(start, stop, num), dtype=dtype, is_leaf=True)

    # ─── Utility ─────────────────────────────────────────────
    def detach(self):
        return tensor(self.value, dtype=self.dtype, is_leaf=True)

    def to_numpy(self):
        return self.value.copy()

    def item(self):
        if self.value.size == 1:
            return self.value.item()
        raise ValueError("Can only convert tensors with a single element to Python scalars")

    def __repr__(self):
        return f"Tensor\n({self.value},\nshape={self.value.shape})"