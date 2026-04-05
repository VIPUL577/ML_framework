import numpy as np
from Seera_init import tensor as Tensor

# ── Import C++ engine if available ──
try:
    import seera_cpp
    _USE_CPP = True
except ImportError:
    _USE_CPP = False

# ─────────────────────────────────────────────────────────────
# Autograd engine for neural networks (batch-aware, C++ accelerated)
# ─────────────────────────────────────────────────────────────
class autograd4nn:

    def __init__(self, hook):
        self.hook = hook
        self.backward()

    # ─── Broadcast gradient reduction ────────────────────────
    @staticmethod
    def _reduce_grad(grad, target_shape):
        if grad.shape == target_shape:
            return grad
        ndim_diff = grad.ndim - len(target_shape)
        padded = (1,) * ndim_diff + tuple(target_shape)
        reduce_axes = tuple(i for i, s in enumerate(padded) if s == 1)
        out = grad.sum(axis=reduce_axes, keepdims=True)
        if ndim_diff > 0:
            out = out.reshape(target_shape)
        return out

    # ─── Conv backward (C++ or NumPy) ────────────────────────
    def conv_backward(self, dO, X, W, stride=1, padding=0):
        if X.ndim == 3: X = X[np.newaxis]
        if dO.ndim == 3: dO = dO[np.newaxis]

        if _USE_CPP:
            dX, dW = seera_cpp.conv2d_backward(
                np.ascontiguousarray(dO, dtype=np.float32),
                np.ascontiguousarray(X, dtype=np.float32),
                np.ascontiguousarray(W, dtype=np.float32),
                stride, padding,
            )
            return dX, dW

        # NumPy fallback
        N, C, H, W_in = X.shape
        F, _, KH, KW = W.shape
        X_col = Tensor.im2col_batch(X.astype(np.float32), KH, KW, stride, padding)
        dO_col = dO.reshape(N, F, -1)
        dW = np.einsum("nfp,ncp->fc", dO_col, X_col).reshape(W.shape)
        W_flat = W.reshape(F, -1)
        dX_col = np.einsum("fc,nfp->ncp", W_flat, dO_col)
        dX = Tensor.col2im_batch(dX_col.astype(np.float32), X.shape, KH, KW, stride, padding)
        return dX, dW

    # ─── MaxPool backward (C++ or NumPy) ─────────────────────
    def maxpool2d_unpool(self, dout, mask, input_shape, kernelsize, stride=1, padding=0):
        KH, KW = kernelsize
        if len(input_shape) == 3:
            input_shape = (1,) + input_shape
        N, C, H, W = input_shape

        if _USE_CPP:
            return seera_cpp.maxpool2d_backward(
                np.ascontiguousarray(dout, dtype=np.float32),
                np.ascontiguousarray(mask, dtype=np.int32),
                N, C, H, W, KH, KW, stride, padding,
            )

        # NumPy fallback
        _, _, OH, OW = dout.shape
        num_patches = OH * OW
        pool_size = KH * KW
        cols = np.zeros((N, C * pool_size, num_patches), dtype=np.float32)
        dout_flat = dout.reshape(N, C, -1)
        mask_flat = mask.reshape(N, C, -1)
        patch_idx = np.arange(num_patches)
        for c in range(C):
            row_indices = c * pool_size + mask_flat[:, c, :]
            for n in range(N):
                cols[n, row_indices[n], patch_idx] = dout_flat[n, c]
        return Tensor.col2im_batch(cols, input_shape, KH, KW, stride, padding)

    # ─── Upsample backward (C++ or NumPy) ────────────────────
    @staticmethod
    def upsample_backward(dout, input_shape, size):
        sw, sh = size
        if len(input_shape) == 3:
            input_shape = (1,) + tuple(input_shape)
        N, C, H, W = input_shape

        if _USE_CPP:
            return seera_cpp.upsample_backward(
                np.ascontiguousarray(dout, dtype=np.float32),
                N, C, H, W, sh, sw,
            )

        return dout.reshape(N, C, H, sh, W, sw).sum(axis=(3, 5))

    # ─── Main backward step ──────────────────────────────────
    def backward_step(self, nodeg):
        # ── MaxPool backward ──
        if nodeg.mpctx[0]:
            cp = nodeg.node.cp.astype(np.float32)
            nodeg.node.child[0].node.cp += self.maxpool2d_unpool(
                cp, nodeg.mpctx[1], nodeg.mpctx[2],
                nodeg.mpctx[3], nodeg.mpctx[4], nodeg.mpctx[5],
            )

        # ── Upsample backward ──
        elif nodeg.upctx[0]:
            cp = nodeg.node.cp.astype(np.float32)
            nodeg.node.child[0].node.cp += self.upsample_backward(
                cp, nodeg.upctx[1], nodeg.upctx[2],
            )

        # ── Flatten backward ──
        elif nodeg.flctx:
            nodeg.node.child[0].node.cp += nodeg.node.cp.reshape(nodeg.flctx)

        # ── Conv2D backward ──
        elif nodeg.iconv2d[0]:
            X = nodeg.node.child[0].value
            W = nodeg.node.child[1].value
            cp = nodeg.node.cp.astype(np.float32)
            dX, dW = self.conv_backward(cp, X, W,
                                        stride=nodeg.iconv2d[1],
                                        padding=nodeg.iconv2d[2])
            nodeg.node.child[0].node.cp += dX
            nodeg.node.child[1].node.cp += dW

        # ── Softmax backward (VJP, C++ accelerated) ──
        elif nodeg.isoftmax:
            s = nodeg.node.child_grad[0, 1]
            dout = nodeg.node.cp
            if _USE_CPP and s.ndim >= 2:
                dx = seera_cpp.softmax_vjp(
                    np.ascontiguousarray(s, dtype=np.float32),
                    np.ascontiguousarray(dout, dtype=np.float32),
                )
            else:
                dot = np.sum(dout * s, axis=-1, keepdims=True)
                dx = s * (dout - dot)
            nodeg.node.child[0].node.cp += dx

        # ── BatchNorm backward (C++ accelerated) ──
        elif nodeg.ibatchnorm is not None:
            ctx = nodeg.ibatchnorm
            dout = np.ascontiguousarray(nodeg.node.cp, dtype=np.float32)
            x_hat = ctx["x_hat"]
            std_inv = ctx["std_inv"]
            gamma = ctx["gamma"]
            M = ctx["N_elements"]
            is_2d = ctx["is_2d"]

            if _USE_CPP:
                dx, dgamma, dbeta = seera_cpp.batchnorm_backward(
                    dout,
                    np.ascontiguousarray(x_hat, dtype=np.float32),
                    np.ascontiguousarray(std_inv, dtype=np.float32),
                    np.ascontiguousarray(gamma, dtype=np.float32),
                    M, is_2d,
                )
            else:
                reduce_axes = (0, 2, 3) if is_2d else (0,)
                bshape = (1, -1, 1, 1) if is_2d else (1, -1)
                gamma_b = gamma.reshape(bshape)
                dgamma = np.sum(dout * x_hat, axis=reduce_axes)
                dbeta = np.sum(dout, axis=reduce_axes)
                dx_hat = dout * gamma_b
                std_inv_b = std_inv.reshape(bshape) if isinstance(std_inv, np.ndarray) and std_inv.ndim == 1 else std_inv
                dx = (1.0 / M) * std_inv_b * (
                    M * dx_hat
                    - np.sum(dx_hat, axis=reduce_axes).reshape(bshape)
                    - x_hat * np.sum(dx_hat * x_hat, axis=reduce_axes).reshape(bshape)
                )

            nodeg.node.child[0].node.cp += dx
            nodeg.node.child[1].node.cp += dgamma
            nodeg.node.child[2].node.cp += dbeta

        # ── Concatenate backward ──
        elif nodeg.iconcatenete:
            split = nodeg.iconcatenete
            cp = nodeg.node.cp
            if cp.ndim >= 4:
                nodeg.node.child[0].node.cp += cp[:, :split]
                nodeg.node.child[1].node.cp += cp[:, split:]
            else:
                nodeg.node.child[0].node.cp += cp[:split]
                nodeg.node.child[1].node.cp += cp[split:]

        # ── Reduction (sum/mean) backward ──
        elif nodeg.ireduction is not None:
            ctx = nodeg.ireduction
            cp = nodeg.node.cp
            input_shape = ctx["input_shape"]
            axis = ctx["axis"]
            keepdims = ctx["keepdims"]
            scale = ctx["scale"]
            if axis is not None and not keepdims:
                if isinstance(axis, int):
                    cp = np.expand_dims(cp, axis=axis)
                else:
                    for ax in sorted(axis):
                        cp = np.expand_dims(cp, axis=ax)
            grad = np.broadcast_to(cp, input_shape) * scale
            nodeg.node.child[0].node.cp += grad.astype(np.float32)

        # ── Matmul backward (C++ accelerated) ──
        elif nodeg.matm:
            A = nodeg.node.child[0]
            B = nodeg.node.child[1]
            dout = np.ascontiguousarray(nodeg.node.cp, dtype=np.float32)

            if _USE_CPP and dout.ndim == 2:
                # dA = dout @ B.T
                A.node.cp += seera_cpp.matmul(dout, np.ascontiguousarray(B.value.T, dtype=np.float32))
                # dB = A.T @ dout
                dB = seera_cpp.matmul(np.ascontiguousarray(A.value.T, dtype=np.float32), dout)
                B.node.cp += self._reduce_grad(dB, B.value.shape)
            else:
                A.node.cp += dout @ B.value.T
                dB = A.value.T @ dout
                B.node.cp += self._reduce_grad(dB, B.value.shape)

        # ── Element-wise ops backward (with broadcast reduction) ──
        else:
            for child_idx, child in enumerate(nodeg.node.child):
                if child_idx < 2:
                    local_grad = nodeg.node.child_grad[child_idx, 1]
                    if isinstance(local_grad, np.ndarray):
                        local_grad = np.asarray(local_grad, dtype=np.float32)
                    cp = nodeg.node.cp
                    if isinstance(cp, np.ndarray):
                        cp = np.asarray(cp, dtype=np.float32)

                    if _USE_CPP and local_grad.shape == cp.shape:
                        raw_grad = seera_cpp.mul(local_grad, cp)
                    else:
                        raw_grad = local_grad * cp

                    reduced = self._reduce_grad(raw_grad, child.value.shape)
                    if isinstance(child.node.cp, (int, float)):
                        child.node.cp = np.zeros_like(child.value, dtype=np.float32)
                    child.node.cp += reduced.astype(np.float32)

    # ─── Backward pass ───────────────────────────────────────
    def backward(self):
        self.hook.node.cp = np.ones_like(self.hook.value, dtype=np.float32)
        graph = list(self.buildgraph())
        for nodeg in graph:
            for child in nodeg.node.child:
                if not isinstance(child.node.cp, np.ndarray) or child.node.cp.shape != child.value.shape:
                    child.node.cp = np.zeros_like(child.value, dtype=np.float32)
        for nodeg in graph:
            self.backward_step(nodeg)

    # ─── Topological sort (DFS) ──────────────────────────────
    def buildgraph(self):
        visited = set()
        topo_order = []
        def connect(nodeg):
            if nodeg in visited: return
            visited.add(nodeg)
            for child in nodeg.node.child:
                connect(child)
            topo_order.append(nodeg)
        connect(self.hook)
        return reversed(topo_order)


# ─────────────────────────────────────────────────────────────
# Basic autograd (non-NN, kept for compatibility)
# ─────────────────────────────────────────────────────────────
class autograd:
    def __init__(self, hook):
        self.hook = hook
        self.backward()

    def backward_step(self, nodeg):
        for a in nodeg.node.child:
            if np.allclose(nodeg.node.child_grad[0, 0], a.node.out):
                nodeg.node.child_grad[0, 1] *= nodeg.node.cp
                a.node.cp += nodeg.node.child_grad[0, 1]
            elif np.allclose(nodeg.node.child_grad[1, 0], a.node.out):
                nodeg.node.child_grad[1, 1] *= nodeg.node.cp
                a.node.cp += nodeg.node.child_grad[1, 1]

    def backward(self):
        self.hook.node.cp = np.ones(self.hook.value.shape)
        graph = self.buildgraph()
        for nodeg in graph:
            self.backward_step(nodeg)

    def buildgraph(self):
        visited = set()
        topo_order = []
        def connect(nodeg):
            if nodeg in visited: return
            visited.add(nodeg)
            for child in nodeg.node.child:
                connect(child)
            topo_order.append(nodeg)
        connect(self.hook)
        return reversed(topo_order)