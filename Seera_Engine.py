import numpy as np
from Seera_init import tensor as Tensor

# ─────────────────────────────────────────────────────────────
# Autograd engine for neural networks (batch-aware)
# ─────────────────────────────────────────────────────────────
class autograd4nn:

    def __init__(self, hook):
        self.hook = hook
        self.backward()

    # ─── Broadcast gradient reduction ────────────────────────
    @staticmethod
    def _reduce_grad(grad, target_shape):
        """Reduce *grad* to *target_shape* by summing over broadcast dims.
        Handles the common case where bias (1, out) was broadcast to (N, out)
        and the gradient needs to be summed over the batch dim."""
        if grad.shape == target_shape:
            return grad
        ndim_diff = grad.ndim - len(target_shape)
        padded = (1,) * ndim_diff + tuple(target_shape)
        reduce_axes = tuple(i for i, s in enumerate(padded) if s == 1)
        out = grad.sum(axis=reduce_axes, keepdims=True)
        if ndim_diff > 0:
            out = out.reshape(target_shape)
        return out

    # ─── Conv backward (batched) ─────────────────────────────
    def conv_backward(self, dO, X, W, stride=1, padding=0):
        """Batched conv backward.
        dO: (N, F, OH, OW) — gradient of output
        X:  (N, C, H, W)   — input
        W:  (F, C, KH, KW) — filters
        Returns dX (N, C, H, W), dW (F, C, KH, KW)
        """
        if X.ndim == 3:
            X = X[np.newaxis]
        if dO.ndim == 3:
            dO = dO[np.newaxis]

        N, C, H, W_in = X.shape
        F, _, KH, KW = W.shape
        _, _, OH, OW = dO.shape

        # im2col of input: (N, C*KH*KW, OH*OW)
        X_col = Tensor.im2col_batch(X.astype(np.float32), KH, KW, stride, padding)

        # dO reshaped: (N, F, OH*OW)
        dO_col = dO.reshape(N, F, -1)

        # ── dW: sum over batch ──
        # dW = sum_n( dO_col[n] @ X_col[n].T )  → (F, C*KH*KW)
        dW = np.einsum("nfp,ncp->fc", dO_col, X_col)
        dW = dW.reshape(W.shape)

        # ── dX: backprop through im2col ──
        W_flat = W.reshape(F, -1)  # (F, C*KH*KW)
        # dX_col = W_flat.T @ dO_col  → (N, C*KH*KW, OH*OW)
        dX_col = np.einsum("fc,nfp->ncp", W_flat, dO_col)

        dX = Tensor.col2im_batch(
            dX_col.astype(np.float32), X.shape, KH, KW, stride, padding
        )
        return dX, dW

    # ─── MaxPool backward (batched) ──────────────────────────
    def maxpool2d_unpool(self, dout, mask, input_shape, kernelsize, stride=1, padding=0):
        """Batched maxpool backward.
        dout: (N, C, OH, OW)
        mask: (N, C, OH, OW) — argmax indices within each pool window
        input_shape: (N, C, H, W)
        """
        KH, KW = kernelsize
        N, C, H, W = input_shape
        _, _, OH, OW = dout.shape

        num_patches = OH * OW
        pool_size = KH * KW

        # Build col-format gradient: (N, C*KH*KW, OH*OW)
        cols = np.zeros((N, C * pool_size, num_patches), dtype=np.float32)
        dout_flat = dout.reshape(N, C, -1)   # (N, C, OH*OW)
        mask_flat = mask.reshape(N, C, -1)   # (N, C, OH*OW)

        patch_idx = np.arange(num_patches)
        for c in range(C):
            # cols[n, c * pool_size + mask[n,c,p], p] = dout[n, c, p]
            # Vectorized over N and patches
            row_indices = c * pool_size + mask_flat[:, c, :]  # (N, OH*OW)
            for n in range(N):
                cols[n, row_indices[n], patch_idx] = dout_flat[n, c]

        dinput = Tensor.col2im_batch(
            cols, input_shape, KH, KW, stride, padding
        )
        return dinput

    # ─── Upsample backward (batched) ─────────────────────────
    @staticmethod
    def upsample_backward(dout, input_shape, size):
        """Backward for nearest-neighbor upsample.
        dout: (N, C, H*sh, W*sw)   input_shape: (N, C, H, W)
        """
        sw, sh = size
        if len(input_shape) == 3:
            # Legacy 3D support
            C, H, W = input_shape
            dx = np.zeros((C, H, W), dtype=dout.dtype)
            for i in range(H):
                for j in range(W):
                    dx[:, i, j] = dout[:, i*sh:(i+1)*sh, j*sw:(j+1)*sw].sum(axis=(1, 2))
            return dx

        N, C, H, W = input_shape
        # Reshape and sum over upsampled blocks
        dx = dout.reshape(N, C, H, sh, W, sw).sum(axis=(3, 5))
        return dx

    # ─── Main backward step ──────────────────────────────────
    def backward_step(self, nodeg):
        # ── MaxPool backward ──
        if nodeg.mpctx[0]:
            cp = nodeg.node.cp.astype(np.float32)
            mask = nodeg.mpctx[1]
            input_shape = nodeg.mpctx[2]
            kernelsize = nodeg.mpctx[3]
            stride_val = nodeg.mpctx[4]
            pad_val = nodeg.mpctx[5]
            nodeg.node.child[0].node.cp += self.maxpool2d_unpool(
                cp, mask, input_shape, kernelsize, stride_val, pad_val
            )

        # ── Upsample backward ──
        elif nodeg.upctx[0]:
            cp = nodeg.node.cp.astype(np.float32)
            nodeg.node.child[0].node.cp += self.upsample_backward(
                cp, nodeg.upctx[1], nodeg.upctx[2]
            )

        # ── Flatten backward ──
        elif nodeg.flctx:
            nodeg.node.child[0].node.cp += nodeg.node.cp.reshape(nodeg.flctx)

        # ── Conv2D backward ──
        elif nodeg.iconv2d[0]:
            X = nodeg.node.child[0].value  # (N, C, H, W)
            W = nodeg.node.child[1].value  # (F, C, KH, KW)
            cp = nodeg.node.cp.astype(np.float32)  # (N, F, OH, OW)

            dX, dW = self.conv_backward(
                cp, X.astype(np.float32), W.astype(np.float32),
                stride=nodeg.iconv2d[1], padding=nodeg.iconv2d[2],
            )
            nodeg.node.child[0].node.cp += dX
            nodeg.node.child[1].node.cp += dW

        # ── Softmax backward (VJP) ──
        elif nodeg.isoftmax:
            s = nodeg.node.child_grad[0, 1]  # softmax output
            dout = nodeg.node.cp              # upstream gradient
            # VJP: dx = s * (dout - sum(dout * s, axis=-1, keepdims=True))
            dot = np.sum(dout * s, axis=-1, keepdims=True)
            dx = s * (dout - dot)
            nodeg.node.child[0].node.cp += dx

        # ── BatchNorm backward ──
        elif nodeg.ibatchnorm is not None:
            ctx = nodeg.ibatchnorm
            dout = nodeg.node.cp
            x_hat = ctx["x_hat"]
            std_inv = ctx["std_inv"]
            gamma = ctx["gamma"]
            reduce_axes = ctx["reduce_axes"]
            bshape = ctx["broadcast_shape"]
            M = ctx["N_elements"]  # number of elements per feature being averaged

            gamma_b = gamma.reshape(bshape)

            # dgamma, dbeta
            dgamma = np.sum(dout * x_hat, axis=reduce_axes)
            dbeta = np.sum(dout, axis=reduce_axes)

            # dx  (efficient fused formula)
            dx_hat = dout * gamma_b
            dx = (1.0 / M) * std_inv * (
                M * dx_hat
                - np.sum(dx_hat, axis=reduce_axes).reshape(bshape)
                - x_hat * np.sum(dx_hat * x_hat, axis=reduce_axes).reshape(bshape)
            )

            # Accumulate gradients
            nodeg.node.child[0].node.cp += dx              # input grad
            nodeg.node.child[1].node.cp += dgamma          # gamma grad
            nodeg.node.child[2].node.cp += dbeta            # beta grad

        # ── Concatenate backward ──
        elif nodeg.iconcatenete:
            split = nodeg.iconcatenete
            cp = nodeg.node.cp
            if cp.ndim >= 4:
                # Batched: split along axis 1 (channels)
                nodeg.node.child[0].node.cp += cp[:, :split]
                nodeg.node.child[1].node.cp += cp[:, split:]
            else:
                nodeg.node.child[0].node.cp += cp[:split]
                nodeg.node.child[1].node.cp += cp[split:]

        # ── Reduction (sum/mean) backward ──
        elif nodeg.ireduction is not None:
            ctx = nodeg.ireduction
            cp = nodeg.node.cp  # upstream gradient (reduced shape)
            input_shape = ctx["input_shape"]
            axis = ctx["axis"]
            keepdims = ctx["keepdims"]
            scale = ctx["scale"]

            # Expand cp back to input shape
            if axis is not None and not keepdims:
                # Re-insert the reduced dimension(s)
                if isinstance(axis, int):
                    cp = np.expand_dims(cp, axis=axis)
                else:
                    for ax in sorted(axis):
                        cp = np.expand_dims(cp, axis=ax)
            # Broadcast to input shape
            grad = np.broadcast_to(cp, input_shape) * scale
            nodeg.node.child[0].node.cp += grad.astype(np.float32)

        # ── Matmul backward (batched) ──
        elif nodeg.matm:
            A = nodeg.node.child[0]  # (N, in)
            B = nodeg.node.child[1]  # (in, out)
            dout = nodeg.node.cp     # (N, out)

            # dA = dout @ B.T  →  (N, in)
            A.node.cp += dout @ B.value.T

            # dB = A.T @ dout  → (in, out)  — sums over batch via matmul
            dB = A.value.T @ dout  # (in, N) @ (N, out) = (in, out)
            # Reduce to B's shape if needed (handles broadcasting)
            B.node.cp += self._reduce_grad(dB, B.value.shape)

        # ── Element-wise ops backward (with broadcast reduction) ──
        else:
            for child_idx, child in enumerate(nodeg.node.child):
                if child_idx < 2:
                    local_grad = nodeg.node.child_grad[child_idx, 1]
                    # child_grad stores object arrays — ensure float
                    if isinstance(local_grad, np.ndarray):
                        local_grad = np.asarray(local_grad, dtype=np.float32)
                    cp = nodeg.node.cp
                    if isinstance(cp, np.ndarray):
                        cp = np.asarray(cp, dtype=np.float32)
                    raw_grad = local_grad * cp
                    # Reduce broadcast dimensions so grad matches child shape
                    reduced = self._reduce_grad(raw_grad, child.value.shape)
                    if isinstance(child.node.cp, (int, float)):
                        child.node.cp = np.zeros_like(child.value, dtype=np.float32)
                    child.node.cp += reduced.astype(np.float32)

    # ─── Backward pass ───────────────────────────────────────
    def backward(self):
        self.hook.node.cp = np.ones_like(self.hook.value, dtype=np.float32)
        graph = list(self.buildgraph())
        # Initialize all gradient accumulators to proper-shaped zero arrays
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
            if nodeg in visited:
                return
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
            if nodeg in visited:
                return
            visited.add(nodeg)
            for child in nodeg.node.child:
                connect(child)
            topo_order.append(nodeg)

        connect(self.hook)
        return reversed(topo_order)