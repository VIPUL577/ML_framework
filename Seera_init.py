import numpy as np
from cuTen import cuten

# ── Try importing C++ engine (falls back to NumPy if unavailable) ──
try:
    import seera_cpp
    _USE_CPP = True
except ImportError:
    _USE_CPP = False
    import warnings
    warnings.warn("C++ engine not found. Using NumPy fallback. Run: python build_engine.py")

# ── Try importing CUDA engine ──
try:
    import seera_cuda
    _USE_CUDA = True
except ImportError:
    _USE_CUDA = False


def _where(value):
    if isinstance(value, cuten):
        return cuten
    return np


def _is_gpu(value):
    return isinstance(value, cuten)


# ─────────────────────────────────────────────────────────────
# Node: gradient info and children in the computation graph
# ─────────────────────────────────────────────────────────────
class node:
    def __init__(self, child_grad, out=0, dtype="float32", device="cpu"):
        self.out = out
        if device == "cuda":
            # child_grad is already a list-of-lists of cuten objects
            self.child_grad = np.array(child_grad, dtype=object)
            if isinstance(child_grad, list) and len(child_grad) > 0 and len(child_grad[0]) > 0:
                ref = child_grad[0][0]
                self.cp = cuten.zeros_like(ref) if isinstance(ref, cuten) else 0
            else:
                self.cp = 0
        else:
            self.child_grad = np.array(child_grad, dtype=dtype)
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
# Batch dim is axis 0:  Dense(N, features)  Conv(N, C, H, W)
# ─────────────────────────────────────────────────────────────
class tensor(node):
    def __init__(self, value, dtype="float32", is_leaf=False, device="cpu"):
        # If value is already a cuten, keep it as-is (GPU tensor)
        if isinstance(value, cuten):
            self.value = value
            device = "cuda"
        else:
            self.value = np.ascontiguousarray(np.array(value).astype(dtype))
            if device == "cuda":
                self.value = cuten(data=self.value, dtype=dtype)

        bk = _where(self.value)
        if bk is cuten:
            z1 = cuten.zeros_like(self.value)
            z2 = cuten.zeros_like(self.value)
            z3 = cuten.zeros_like(self.value)
            z4 = cuten.zeros_like(self.value)
            child_grad = [[z1, z2], [z3, z4]]
            self.node = node(child_grad, device="cuda")
        else:
            child_grad = (
                [np.zeros_like(self.value), np.zeros_like(self.value)],
                [np.zeros_like(self.value), np.zeros_like(self.value)],
            )
            self.node = node(child_grad, device="cpu")

        self.is_leaf = is_leaf
        self.dtype = dtype

        # ── special backward context flags ──
        self.matm = False
        self.isoftmax = False
        self.iconv2d = (False, 1, 1, 0, 0)
        self.unpoolctx = (False, 0, 0)
        self.flctx = 0
        self.mpctx = (False, 0, 0, 0, 1, 1, 0, 0)
        self.iconcatenete = 0
        self.ibatchnorm = None
        self.ireduction = None
        self.convTrans = (False, 1, 1, 0, 0)

        if is_leaf:
            self.node.out = self.value

    # ─── Element-wise ops ────────────────────────────────────
    def __add__(self, other):
        gpu = _is_gpu(self.value)

        if isinstance(other, (int, float)):
            if gpu:
                other = tensor(cuten.ones_like(self.value) * other)
            else:
                other = tensor(other * np.ones_like(self.value))
        elif not isinstance(other, tensor):
            other = tensor(other)

        if gpu:
            result = self.value + other.value
            out = tensor(result)
            child_grad = [
                [self.value, cuten.ones_like(self.value)],
                [other.value, cuten.ones_like(other.value)],
            ]
            out.node.child_grad = np.array(child_grad, dtype=object)
        else:
            if _USE_CPP and self.value.shape == other.value.shape:
                result = seera_cpp.add(self.value, other.value)
            else:
                result = self.value + other.value
            out = tensor(result)
            child_grad = [
                [self.value, np.ones_like(self.value)],
                [other.value, np.ones_like(other.value)],
            ]
            out.node.child_grad = np.array(child_grad, dtype=object)

        out.node.out = out.value
        out.node.child = [self, other]
        return out

    def __mul__(self, other):
        gpu = _is_gpu(self.value)

        if isinstance(other, (int, float)):
            if gpu:
                other = tensor(cuten.ones_like(self.value) * other)
            else:
                other = tensor(other * np.ones_like(self.value))
        elif not isinstance(other, tensor):
            other = tensor(other)

        if gpu:
            result = self.value * other.value
            out = tensor(result)
            child_grad = [
                [self.value, other.value],
                [other.value, self.value],
            ]
            out.node.child_grad = np.array(child_grad, dtype=object)
        else:
            if _USE_CPP and self.value.shape == other.value.shape:
                result = seera_cpp.mul(self.value, other.value)
            else:
                result = self.value * other.value
            out = tensor(result)
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
        gpu = _is_gpu(self.value)

        if gpu:
            out_ptr = seera_cuda.cuda_malloc_f32(self.value.size)
            grad_ptr = seera_cuda.cuda_malloc_f32(self.value.size)
            seera_cuda.cuda_pow_fwd(self.value.main_ptr, other, out_ptr, grad_ptr, self.value.size)

            out_val = cuten(data=None, dtype="float32")
            out_val.main_ptr = out_ptr
            out_val.shape = self.value.shape
            out_val.size = self.value.size

            grad_val = cuten(data=None, dtype="float32")
            grad_val.main_ptr = grad_ptr
            grad_val.shape = self.value.shape
            grad_val.size = self.value.size

            return self._unary(out_val, grad_val)
        else:
            if _USE_CPP:
                out_val, gradient = seera_cpp.pow_act(
                    np.ascontiguousarray(self.value, dtype=np.float32), other
                )
            else:
                out_val = self.value ** other
                gradient = other * (self.value ** (other - 1))
            return self._unary(out_val, gradient)

    def __neg__(self):    return self * (-1)
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rsub__(self, other): return other + (self * -1)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1

    # ─── Unary op helper ─────────────────────────────────────
    def _unary(self, fwd_val, gradient):
        gpu = _is_gpu(self.value)
        out = tensor(fwd_val)
        if gpu:
            nan_like = cuten.zeros_like(self.value)  # placeholder, won't be used
            nan_grad = cuten.zeros_like(self.value)
            child_grad = [
                [self.value, gradient],
                [nan_like, nan_grad],
            ]
            out.node.child_grad = np.array(child_grad, dtype=object)
        else:
            child_grad = [
                [self.value, gradient],
                [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)],
            ]
            out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        out.node.child = [self]
        return out

    # ─── GPU unary activation helper ─────────────────────────
    def _gpu_activation(self, kernel_fn):
        """Run a CUDA unary activation that fills (out, grad).
        Returns (out_cuten, grad_cuten)."""
        out_ptr = seera_cuda.cuda_malloc_f32(self.value.size)
        grad_ptr = seera_cuda.cuda_malloc_f32(self.value.size)
        kernel_fn(self.value.main_ptr, out_ptr, grad_ptr, self.value.size)

        out_val = cuten(data=None, dtype="float32")
        out_val.main_ptr = out_ptr
        out_val.shape = self.value.shape
        out_val.size = self.value.size

        grad_val = cuten(data=None, dtype="float32")
        grad_val.main_ptr = grad_ptr
        grad_val.shape = self.value.shape
        grad_val.size = self.value.size

        return out_val, grad_val

    # ─── Activations (C++ / CUDA accelerated) ───────────────
    def relu(self):
        if _is_gpu(self.value):
            o, g = self._gpu_activation(seera_cuda.cuda_relu_fwd)
            return self._unary(o, g)
        if _USE_CPP:
            o, g = seera_cpp.relu(np.ascontiguousarray(self.value, dtype=np.float32))
        else:
            o = np.where(self.value > 0, self.value, 0)
            g = np.where(self.value > 0, 1.0, 0.0)
        return self._unary(o, g)

    def sigmoid(self):
        if _is_gpu(self.value):
            o, g = self._gpu_activation(seera_cuda.cuda_sigmoid_fwd)
            return self._unary(o, g)
        if _USE_CPP:
            o, g = seera_cpp.sigmoid(np.ascontiguousarray(self.value, dtype=np.float32))
        else:
            s = 1 / (1 + np.exp(-self.value))
            o, g = s, s * (1 - s)
        return self._unary(o, g)

    def tanh(self):
        if _is_gpu(self.value):
            o, g = self._gpu_activation(seera_cuda.cuda_tanh_fwd)
            return self._unary(o, g)
        if _USE_CPP:
            o, g = seera_cpp.tanh_act(np.ascontiguousarray(self.value, dtype=np.float32))
        else:
            t = np.tanh(self.value)
            o, g = t, 1 - t ** 2
        return self._unary(o, g)

    def log(self):
        if _is_gpu(self.value):
            o, g = self._gpu_activation(seera_cuda.cuda_log_fwd)
            return self._unary(o, g)
        if _USE_CPP:
            o, g = seera_cpp.log_act(np.ascontiguousarray(self.value, dtype=np.float32))
        else:
            o, g = np.log(self.value), 1 / self.value
        return self._unary(o, g)

    def exp(self):
        if _is_gpu(self.value):
            o, g = self._gpu_activation(seera_cuda.cuda_exp_fwd)
            return self._unary(o, g)
        if _USE_CPP:
            o, g = seera_cpp.exp_act(np.ascontiguousarray(self.value, dtype=np.float32))
        else:
            e = np.exp(self.value)
            o, g = e, e
        return self._unary(o, g)

    def abs(self):
        if _is_gpu(self.value):
            o, g = self._gpu_activation(seera_cuda.cuda_abs_fwd)
            return self._unary(o, g)
        if _USE_CPP:
            o, g = seera_cpp.abs_act(np.ascontiguousarray(self.value, dtype=np.float32))
        else:
            o, g = np.abs(self.value), np.sign(self.value)
        return self._unary(o, g)

    def sqrt(self):
        if _is_gpu(self.value):
            o, g = self._gpu_activation(seera_cuda.cuda_sqrt_fwd)
            return self._unary(o, g)
        if _USE_CPP:
            o, g = seera_cpp.sqrt_act(np.ascontiguousarray(self.value, dtype=np.float32))
        else:
            s = np.sqrt(self.value)
            o, g = s, 0.5 / (s + 1e-12)
        return self._unary(o, g)

    def clip(self, min_val, max_val):
        if _is_gpu(self.value):
            out_ptr = seera_cuda.cuda_malloc_f32(self.value.size)
            grad_ptr = seera_cuda.cuda_malloc_f32(self.value.size)
            seera_cuda.cuda_clip_fwd(self.value.main_ptr, float(min_val), float(max_val),
                                     out_ptr, grad_ptr, self.value.size)
            out_val = cuten(data=None, dtype="float32")
            out_val.main_ptr = out_ptr
            out_val.shape = self.value.shape
            out_val.size = self.value.size

            grad_val = cuten(data=None, dtype="float32")
            grad_val.main_ptr = grad_ptr
            grad_val.shape = self.value.shape
            grad_val.size = self.value.size
            return self._unary(out_val, grad_val)
        if _USE_CPP:
            o, g = seera_cpp.clip_act(
                np.ascontiguousarray(self.value, dtype=np.float32), min_val, max_val
            )
        else:
            gradient = np.ones_like(self.value)
            gradient[self.value < min_val] = 0
            gradient[self.value > max_val] = 0
            o, g = np.clip(self.value, min_val, max_val), gradient
        return self._unary(o, g)

    def sin(self):
        return self._unary(np.sin(self.value), np.cos(self.value))

    def cos(self):
        return self._unary(np.cos(self.value), -np.sin(self.value))

    def tan(self):
        return self._unary(np.tan(self.value), 1 / (np.cos(self.value) ** 2))

    # ─── Softmax (C++ / CUDA accelerated, VJP-based) ─────────
    def softmax(self, axis=-1):
        gpu = _is_gpu(self.value)

        if gpu:
            s = self.value.softmax()  # cuten.softmax()
            out = tensor(s)
            out.isoftmax = True
            child_grad = np.empty((1, 2), dtype=object)
            child_grad[0, 0] = self.value
            child_grad[0, 1] = s  # softmax output for VJP backward
            out.node.child_grad = child_grad
            out.node.out = out.value
            out.node.child = [self]
            return out

        x = np.ascontiguousarray(self.value, dtype=np.float32)
        if _USE_CPP and x.ndim >= 2:
            s = seera_cpp.softmax(x)
        else:
            shifted = x - np.max(x, axis=axis, keepdims=True)
            exps = np.exp(shifted)
            s = exps / np.sum(exps, axis=axis, keepdims=True)

        out = tensor(s)
        out.isoftmax = True
        child_grad = np.empty((1, 2), dtype=object)
        child_grad[0, 0] = self.value
        child_grad[0, 1] = s  # softmax output for VJP backward
        out.node.child_grad = child_grad
        out.node.out = out.value
        out.node.child = [self]
        return out

    # ─── Matmul (C++ / CUDA accelerated) ─────────────────────
    def matmul(self, other):
        gpu = _is_gpu(self.value)

        if gpu:
            result = self.value.matmul(other.value)
            out = tensor(result)
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

        a = np.ascontiguousarray(self.value, dtype=np.float32)
        b = np.ascontiguousarray(other.value, dtype=np.float32)
        if _USE_CPP and a.ndim == 2 and b.ndim == 2:
            result = seera_cpp.matmul(a, b)
        else:
            result = a @ b

        out = tensor(result)
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
        gpu = _is_gpu(self.value)

        if gpu:
            if axis is None:
                # Reduce all dimensions sequentially (last to first)
                result = self
                for dim in reversed(range(len(self.value.shape))):
                    result = result.sum(axis=dim)
                return result
            s = self.value.sum(dim=axis)
            out = tensor(s)
            out.node.out = out.value
            out.node.child = [self]
            out.ireduction = {
                "input_shape": self.value.shape,
                "keepdims": keepdims,
                "scale": 1.0,
                "gpu": True,
                "dimarr": np.array(self.value.shape, dtype=np.int32),
                "ndims": len(self.value.shape),
                "dim": axis,
            }
            return out

        s = np.sum(self.value, axis=axis, keepdims=keepdims)
        out = tensor(s)
        out.node.out = out.value
        out.node.child = [self]
        out.ireduction = {
            "input_shape": self.value.shape,
            "axis": axis,
            "keepdims": keepdims,
            "scale": 1.0,
        }
        return out

    def mean(self, axis=None, keepdims=False):
        gpu = _is_gpu(self.value)

        if gpu:
            if axis is None:
                # mean over all elements = sum_all / total_size
                total = self.value.size
                s = self.sum()  # reduces to scalar
                return s * (1.0 / total)
            s = self.value.mean(dim=axis)
            out = tensor(s)
            out.node.child = [self]
            out.node.out = out.value
            n = self.value.shape[axis]
            out.ireduction = {
                "input_shape": self.value.shape,
                "axis": axis,
                "keepdims": keepdims,
                "scale": 1.0 / n,
                "gpu": True,
                "dimarr": np.array(self.value.shape, dtype=np.int32),
                "ndims": len(self.value.shape),
                "dim": axis,
            }
            return out

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
            "scale": 1.0 / n,
        }
        return out

    def max(self, axis=None, keepdims=False):
        gpu = _is_gpu(self.value)

        if gpu and axis is not None:
            out_val = self.value.max(dim=axis)
            out = tensor(out_val)
            out.node.child = [self]
            out.node.out = out.value
            out.ireduction = {
                "input_shape": self.value.shape,
                "axis": axis,
                "keepdims": keepdims,
                "scale": 1.0,
                "gpu": True,
                "type": "max",
                "fwdInput": self.value,
                "fwdOutput": out_val,
                "dimarr": np.array(self.value.shape, dtype=np.int32),
                "ndims": len(self.value.shape),
                "dim": axis,
            }
            return out

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
        gpu = _is_gpu(self.value)

        if gpu and axis is not None:
            out_val = self.value.min(dim=axis)
            out = tensor(out_val)
            out.node.child = [self]
            out.node.out = out.value
            out.ireduction = {
                "input_shape": self.value.shape,
                "axis": axis,
                "keepdims": keepdims,
                "scale": 1.0,
                "gpu": True,
                "type": "min",
                "fwdInput": self.value,
                "fwdOutput": out_val,
                "dimarr": np.array(self.value.shape, dtype=np.int32),
                "ndims": len(self.value.shape),
                "dim": axis,
            }
            return out

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
        return self._unary(np.squeeze(self.value, axis=axis), np.ones_like(self.value))

    def unsqueeze(self, axis):
        return self._unary(np.expand_dims(self.value, axis=axis), np.ones_like(self.value))

    def flatten(self):
        gpu = _is_gpu(self.value)
        original_shape = self.value.shape
        N = original_shape[0]

        if gpu:
            # cuten flatten: reshape in place then wrap as new tensor
            flat = cuten(data=None, dtype="float32")
            flat.main_ptr = self.value.main_ptr
            flat.size = self.value.size
            flat.shape = (N, int(self.value.size / N))
            out = tensor(flat)
        else:
            out = tensor(self.value.reshape(N, -1))

        out.flctx = original_shape
        out.node.child = [self]
        return out

    def __getitem__(self, key):
        out = tensor(self.value[key])
        out.node.child = [self]
        gradient = np.zeros_like(self.value)
        sliced = np.zeros_like(self.value, dtype=bool)
        sliced[key] = True
        gradient[sliced] = 1.0
        child_grad = [
            [self.value, gradient],
            [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)],
        ]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        return out

    # ─── Conv2D forward (C++ / CUDA accelerated) ─────────────
    def conv2d(self, W, stride=1, padding=0):
        gpu = _is_gpu(self.value)

        # Normalize stride/padding to (h, w) tuples
        if isinstance(stride, int):
            strideh, stridew = stride, stride
        else:
            strideh, stridew = stride
        if isinstance(padding, int):
            padh, padw = padding, padding
        else:
            padh, padw = padding

        if gpu:
            result = self.value.conv2d(W.value, strideh=strideh, stridew=stridew,
                                       padh=padh, padw=padw)
            out = tensor(result)
            out.iconv2d = (True, strideh, stridew, padh, padw)
            out.node.out = out.value
            out.node.child = [self, W]
            return out

        x = np.ascontiguousarray(self.value, dtype=np.float32)
        w = np.ascontiguousarray(W.value, dtype=np.float32)
        if x.ndim == 3:
            x = x[np.newaxis]

        if _USE_CPP:
            out_val = seera_cpp.conv2d_forward(x, w, strideh, stridew, padh, padw)
        else:
            N, C, H, W_in = x.shape
            F, _, KH, KW = w.shape
            OH = (H + 2 * padh - KH) // strideh + 1
            OW = (W_in + 2 * padw - KW) // stridew + 1
            col = tensor.im2col_batch(x, KH, KW, strideh, stridew, padh, padw)
            W_col = w.reshape(F, -1)
            out_val = np.einsum("fc,ncp->nfp", W_col, col).reshape(N, F, OH, OW)

        out = tensor(out_val)
        out.iconv2d = (True, strideh, stridew, padh, padw)
        out.node.out = out.value
        out.node.child = [self, W]
        return out

    # ─── MaxPool2D forward (C++ / CUDA accelerated) ──────────
    def maxpool2d(self, kernelsize, stride=1, padding=0):
        if not isinstance(kernelsize, tuple):
            raise ValueError("kernelsize should be a tuple (height, width)")
        KH, KW = kernelsize
        gpu = _is_gpu(self.value)

        # Normalize stride/padding to (h, w) tuples
        if isinstance(stride, int):
            strideh, stridew = stride, stride
        else:
            strideh, stridew = stride
        if isinstance(padding, int):
            padh, padw = padding, padding
        else:
            padh, padw = padding

        if gpu:
            result, mask = self.value.maxpool2d(KH, KW, strideh=strideh, stridew=stridew,
                                                 padh=padh, padw=padw)
            out = tensor(result)
            out.mpctx = (True, mask, self.value.shape, kernelsize, strideh, stridew, padh, padw)
            out.node.out = out.value
            out.node.child = [self]
            return out

        img = np.ascontiguousarray(self.value, dtype=np.float32)
        if img.ndim == 3:
            img = img[np.newaxis]

        if _USE_CPP:
            out_val, mask = seera_cpp.maxpool2d_forward(img, KH, KW, strideh, stridew, padh, padw)
            mask = np.array(mask, dtype=np.int32)
        else:
            N, C, H, W = img.shape
            OH = (H + 2 * padh - KH) // strideh + 1
            OW = (W + 2 * padw - KW) // stridew + 1
            col = tensor.im2col_batch(img, KH, KW, strideh, stridew, padh, padw)
            col = col.reshape(N, C, KH * KW, OH * OW)
            mask = np.argmax(col, axis=2).reshape(N, C, OH, OW)
            out_val = np.max(col, axis=2).reshape(N, C, OH, OW)

        out = tensor(out_val)
        out.mpctx = (True, mask, self.value.shape, kernelsize, strideh, stridew, padh, padw)
        out.node.out = out.value
        out.node.child = [self]
        return out

    # ─── Concatenate ─────────────────────────────────────────
    def concatenete(self, other):
        gpu = _is_gpu(self.value)

        if gpu:
            if len(self.value.shape) >= 3 and len(other.value.shape) >= 3:
                result = self.value.concatenate2D(other.value)
                out = tensor(result)
                out.iconcatenete = self.value.shape[1]
            else:
                result = self.value.concatenate1D(other.value)
                out = tensor(result)
                out.iconcatenete = self.value.shape[1]
            out.node.child = [self, other]
            out.node.out = out.value
            return out

        if self.value.ndim >= 3 and other.value.ndim >= 3:
            out = tensor(np.concatenate([self.value, other.value], axis=1))
            out.iconcatenete = self.value.shape[1]
        else:
            out = tensor(np.concatenate([self.value, other.value], axis=0))
            out.iconcatenete = self.value.shape[0]
        out.node.child = [self, other]
        out.node.out = out.value
        return out

    # ─── Unpool2D Nearest (C++ / CUDA accelerated) ───────────
    def Unpool2Dnearest(self, size):
        gpu = _is_gpu(self.value)
        sw, sh = size

        if gpu:
            result = self.value.unpool(sh, sw)
            out = tensor(result)
            out.node.child = [self]
            out.node.out = out.value
            out.unpoolctx = (True, self.value.shape, size)
            return out

        x = np.ascontiguousarray(self.value, dtype=np.float32)
        if x.ndim == 3:
            x = x[np.newaxis]

        if _USE_CPP:
            x_up = seera_cpp.unpooling_forward(x, sh, sw)
        else:
            x_up = np.repeat(np.repeat(x, sh, axis=2), sw, axis=3)

        out = tensor(x_up)
        out.node.child = [self]
        out.node.out = out.value
        out.unpoolctx = (True, self.value.shape, size)
        return out

    # ─── ConvTranspose2D forward (C++ / CUDA accelerated) ────
    def conv_transpose2d(self, W, stride=1, padding=0):
        gpu = _is_gpu(self.value)

        # Normalize stride/padding to (h, w) tuples
        if isinstance(stride, int):
            strideh, stridew = stride, stride
        else:
            strideh, stridew = stride
        if isinstance(padding, int):
            padh, padw = padding, padding
        else:
            padh, padw = padding

        if gpu:
            result = self.value.conv2d_transpose(W.value, strideh=strideh, stridew=stridew,
                                                  padh=padh, padw=padw)
            out = tensor(result)
            out.convTrans = (True, strideh, stridew, padh, padw)
            out.node.out = out.value
            out.node.child = [self, W]
            return out

        x = np.ascontiguousarray(self.value, dtype=np.float32)
        w = np.ascontiguousarray(W.value, dtype=np.float32)
        if x.ndim == 3:
            x = x[np.newaxis]

        if _USE_CPP:
            out_val = seera_cpp.conv_transpose2d_forward(x, w, strideh, stridew, padh, padw)
        else:
            # NumPy fallback
            N, Cin, H, Win = x.shape
            _, Cout, KH, KW = w.shape
            Hout = (H - 1) * strideh - 2 * padh + KH
            Wout = (Win - 1) * stridew - 2 * padw + KW
            col_row = Cout * KH * KW
            spatial_in = H * Win
            W_flat = w.reshape(Cin, col_row)
            X_flat = x.reshape(N, Cin, spatial_in)
            col = np.einsum('cr,ncs->nrs', W_flat, X_flat)
            out_val = tensor.col2im_batch(
                col.reshape(N, col_row, spatial_in),
                (N, Cout, Hout, Wout), KH, KW, strideh, stridew, padh, padw,
            )

        out = tensor(out_val)
        out.convTrans = (True, strideh, stridew, padh, padw)
        out.node.out = out.value
        out.node.child = [self, W]
        return out

    # ─── BatchNorm forward (C++ accelerated) ─────────────────
    # NOTE: BatchNorm backward is NOT GPU-accelerated (left as-is)
    def batchnorm(self, gamma, beta, running_mean, running_var,
                  training=True, momentum=0.1, eps=1e-5, mode="1d"):
        x = np.ascontiguousarray(self.value, dtype=np.float32)
        is_2d = (mode == "2d")

        if _USE_CPP:
            g = np.ascontiguousarray(gamma.value, dtype=np.float32)
            b = np.ascontiguousarray(beta.value, dtype=np.float32)
            rm = np.ascontiguousarray(running_mean, dtype=np.float32)
            rv = np.ascontiguousarray(running_var, dtype=np.float32)

            y, x_hat, std_inv = seera_cpp.batchnorm_forward(
                x, g, b, rm, rv, momentum, eps, training, is_2d
            )
            # Copy running stats back (they were updated in-place in C++)
            running_mean[:] = rm
            running_var[:] = rv

            N_elements = x.size // gamma.value.size
        else:
            reduce_axes = (0, 2, 3) if is_2d else (0,)
            broadcast_shape = (1, -1, 1, 1) if is_2d else (1, -1)
            if training:
                mu = np.mean(x, axis=reduce_axes)
                var = np.var(x, axis=reduce_axes)
                running_mean[:] = (1 - momentum) * running_mean + momentum * mu
                running_var[:] = (1 - momentum) * running_var + momentum * var
            else:
                mu, var = running_mean, running_var
            mu_b = mu.reshape(broadcast_shape)
            var_b = var.reshape(broadcast_shape)
            std_inv_scalar = 1.0 / np.sqrt(var_b + eps)
            x_hat = (x - mu_b) * std_inv_scalar
            gamma_b = gamma.value.reshape(broadcast_shape)
            beta_b = beta.value.reshape(broadcast_shape)
            y = gamma_b * x_hat + beta_b
            std_inv = (1.0 / np.sqrt(var + eps)).astype(np.float32)
            N_elements = x.size // gamma.value.size

        out = tensor(y)
        out.node.child = [self, gamma, beta]
        out.node.out = out.value
        out.ibatchnorm = {
            "x_hat": x_hat,
            "std_inv": std_inv,
            "gamma": gamma.value,
            "N_elements": N_elements,
            "is_2d": is_2d,
        }
        return out

    # ─── NumPy fallback im2col/col2im (used when C++ unavailable) ──
    @staticmethod
    def im2col_batch(X, KH, KW, strideh=1, stridew=1, padh=0, padw=0):
        N, C, H, W = X.shape
        OH = (H + 2 * padh - KH) // strideh + 1
        OW = (W + 2 * padw - KW) // stridew + 1
        if padh > 0 or padw > 0:
            X = np.pad(X, ((0, 0), (0, 0), (padh, padh), (padw, padw)), mode="constant")
        _, _, Hp, Wp = X.shape
        s_n, s_c, s_h, s_w = X.strides
        shape = (N, C, OH, OW, KH, KW)
        strides = (s_n, s_c, s_h * strideh, s_w * stridew, s_h, s_w)
        col = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        return np.ascontiguousarray(col.reshape(N, C * KH * KW, OH * OW))

    @staticmethod
    def col2im_batch(cols, X_shape, KH, KW, strideh=1, stridew=1, padh=0, padw=0):
        N, C, H, W = X_shape
        OH = (H + 2 * padh - KH) // strideh + 1
        OW = (W + 2 * padw - KW) // stridew + 1
        Hp, Wp = H + 2 * padh, W + 2 * padw
        X_padded = np.zeros((N, C, Hp, Wp), dtype=cols.dtype)
        cols_reshaped = cols.reshape(N, C, KH, KW, OH, OW)
        for i in range(KH):
            i_end = i + strideh * OH
            for j in range(KW):
                j_end = j + stridew * OW
                X_padded[:, :, i:i_end:strideh, j:j_end:stridew] += cols_reshaped[:, :, i, j, :, :]
        if padh > 0 or padw > 0:
            return X_padded[:, :, padh:Hp-padh if padh else Hp, padw:Wp-padw if padw else Wp]
        return X_padded

    # ─── Factory methods ─────────────────────────────────────
    @classmethod
    def zeros(cls, shape, dtype="float32", device="cpu"):
        if device == "cuda":
            return cls(cuten.zeros(shape), dtype=dtype, is_leaf=True)
        return cls(np.zeros(shape), dtype=dtype, is_leaf=True)

    @classmethod
    def ones(cls, shape, dtype="float32", device="cpu"):
        if device == "cuda":
            return cls(cuten.ones(shape), dtype=dtype, is_leaf=True)
        return cls(np.ones(shape), dtype=dtype, is_leaf=True)

    @classmethod
    def random(cls, shape, dtype="float32", device="cpu"):
        if device == "cuda":
            return cls(cuten(np.random.random(shape).astype(dtype)), dtype=dtype, is_leaf=True)
        return cls(np.random.random(shape), dtype=dtype, is_leaf=True)

    @classmethod
    def randn(cls, *shape, dtype="float32", device="cpu"):
        if device == "cuda":
            return cls(cuten(np.random.randn(*shape).astype(dtype)), dtype=dtype, is_leaf=True)
        return cls(np.random.randn(*shape), dtype=dtype, is_leaf=True)

    @classmethod
    def eye(cls, n, dtype="float32"):
        return cls(np.eye(n), dtype=dtype, is_leaf=True)

    @classmethod
    def arange(cls, start, stop=None, step=1, dtype="float32"):
        if stop is None: stop = start; start = 0
        return cls(np.arange(start, stop, step), dtype=dtype, is_leaf=True)

    @classmethod
    def linspace(cls, start, stop, num=50, dtype="float32"):
        return cls(np.linspace(start, stop, num), dtype=dtype, is_leaf=True)

    # ─── Utility ─────────────────────────────────────────────
    def detach(self):
        return tensor(self.value, dtype=self.dtype, is_leaf=True)

    def to_numpy(self):
        if _is_gpu(self.value):
            return self.value.to_host_f32().copy()
        return self.value.copy()

    def item(self):
        if _is_gpu(self.value):
            arr = self.value.to_host_f32()
            if arr.size == 1:
                return arr.item()
            raise ValueError("Can only convert tensors with a single element to Python scalars")
        if self.value.size == 1:
            return self.value.item()
        raise ValueError("Can only convert tensors with a single element to Python scalars")

    def __repr__(self):
        if _is_gpu(self.value):
            arr = self.value.to_host_f32()
            return f"Tensor(GPU)\n({arr},\nshape={self.value.shape})"
        return f"Tensor\n({self.value},\nshape={self.value.shape})"