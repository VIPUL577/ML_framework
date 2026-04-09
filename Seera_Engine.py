import numpy as np
from Seera_init import tensor as Tensor, _where, _is_gpu
from cuTen import cuten

# ── Import C++ engine if available ──
try:
    import seera_cpp
    _USE_CPP = True
except ImportError:
    _USE_CPP = False

# ── Import CUDA engine if available ──
try:
    import seera_cuda
    _USE_CUDA = True
except ImportError:
    _USE_CUDA = False

# ─────────────────────────────────────────────────────────────
# Autograd engine for neural networks (batch-aware, C++ / CUDA accelerated)
# ─────────────────────────────────────────────────────────────
class autograd4nn:

    def __init__(self, hook):
        self.hook = hook
        self.backward()

    # ─── Broadcast gradient reduction (CPU only) ─────────────
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

    # ─── Broadcast gradient reduction (GPU — cuten) ──────────
    @staticmethod
    def _reduce_grad_gpu(grad, target_shape):
        """Reduce a cuten gradient to target_shape by summing broadcast dims."""
        if grad.shape == target_shape:
            return grad
        grad_ndim = len(grad.shape)
        target_ndim = len(target_shape)
        ndim_diff = grad_ndim - target_ndim
        padded = (1,) * ndim_diff + tuple(target_shape)
        # Sum along axes where target is 1 (broadcast dims)
        # Must reduce from highest dim to lowest to keep indices valid
        reduce_axes = [i for i, s in enumerate(padded) if s == 1]
        result = grad
        for ax in reversed(reduce_axes):
            result = result.sum(dim=ax)
        # Always force the final shape to match target_shape.
        # sum(dim=ax) removes the axis entirely (e.g. (N,out) dim=0 → (out,)),
        # but target may be (1,out). Fix shape metadata to match.
        if result.shape != target_shape:
            result.shape = target_shape
            result.size = 1
            for d in target_shape:
                result.size *= d
        return result

    # ─── Conv backward (C++ / CUDA / NumPy) ──────────────────
    def conv_backward(self, dO, X, W, strideh=1, stridew=1, paddingh=0, paddingw=0):
        gpu = _is_gpu(X)

        if gpu:
            batchN, C, H, W_in = X.shape
            F, _Ck, KH, KW = W.shape
            dX_ptr = seera_cuda.cuda_malloc_f32(X.size)
            dW_ptr = seera_cuda.cuda_malloc_f32(W.size)
            seera_cuda.cuda_conv2d_bwd(
                W.main_ptr, X.main_ptr, dO.main_ptr,
                dX_ptr, dW_ptr,
                batchN, C, H, W_in, F, KH, KW,
                strideh, stridew, paddingh, paddingw,
            )
            dX = cuten(data=None, dtype="float32")
            dX.main_ptr = dX_ptr
            dX.shape = X.shape
            dX.size = X.size

            dW = cuten(data=None, dtype="float32")
            dW.main_ptr = dW_ptr
            dW.shape = W.shape
            dW.size = W.size
            return dX, dW

        if isinstance(X, np.ndarray) and X.ndim == 3: X = X[np.newaxis]
        if isinstance(dO, np.ndarray) and dO.ndim == 3: dO = dO[np.newaxis]

        if _USE_CPP:
            dX, dW = seera_cpp.conv2d_backward(
                np.ascontiguousarray(dO, dtype=np.float32),
                np.ascontiguousarray(X, dtype=np.float32),
                np.ascontiguousarray(W, dtype=np.float32),
                strideh, stridew, paddingh, paddingw,
            )
            return dX, dW

        # NumPy fallback
        N, C, H, W_in = X.shape
        F, _, KH, KW = W.shape
        X_col = Tensor.im2col_batch(X.astype(np.float32), KH, KW, strideh, stridew, paddingh, paddingw)
        dO_col = dO.reshape(N, F, -1)
        dW = np.einsum("nfp,ncp->fc", dO_col, X_col).reshape(W.shape)
        W_flat = W.reshape(F, -1)
        dX_col = np.einsum("fc,nfp->ncp", W_flat, dO_col)
        dX = Tensor.col2im_batch(dX_col.astype(np.float32), X.shape, KH, KW, strideh, stridew, paddingh, paddingw)
        return dX, dW

    # ─── MaxPool backward (C++ / CUDA / NumPy) ───────────────
    def maxpool2d_unpool(self, dout, mask, input_shape, kernelsize, strideh=1, stridew=1, paddingh=0, paddingw=0):
        KH, KW = kernelsize
        gpu = _is_gpu(dout)

        if gpu:
            if len(input_shape) == 3:
                input_shape = (1,) + input_shape
            N, C, H, W = input_shape
            dX_ptr = seera_cuda.cuda_malloc_f32(N * C * H * W)
            seera_cuda.cuda_memset(dX_ptr, 0, N * C * H * W * 4)
            seera_cuda.cuda_maxpool_bwd(
                dout.main_ptr, mask.main_ptr, dX_ptr,
                N, C, H, W, KH, KW,
                paddingh, paddingw, strideh, stridew,
            )
            dX = cuten(data=None, dtype="float32")
            dX.main_ptr = dX_ptr
            dX.shape = input_shape
            dX.size = N * C * H * W
            return dX

        if len(input_shape) == 3:
            input_shape = (1,) + input_shape
        N, C, H, W = input_shape

        if _USE_CPP:
            return seera_cpp.maxpool2d_backward(
                np.ascontiguousarray(dout, dtype=np.float32),
                np.ascontiguousarray(mask, dtype=np.int32),
                N, C, H, W, KH, KW, strideh, stridew, paddingh, paddingw,
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
        return Tensor.col2im_batch(cols, input_shape, KH, KW, strideh, stridew, paddingh, paddingw)

    # ─── Unpooling backward (C++ / CUDA / NumPy) ─────────────
    @staticmethod
    def unpooling_backward(dout, input_shape, size):
        sw, sh = size
        gpu = _is_gpu(dout)

        if gpu:
            if len(input_shape) == 3:
                input_shape = (1,) + tuple(input_shape)
            N, C, H, W = input_shape
            dx_ptr = seera_cuda.cuda_malloc_f32(N * C * H * W)
            seera_cuda.cuda_unpooling_bwd(
                dout.main_ptr, dx_ptr,
                N, C, H, W, sh, sw,
            )
            dx = cuten(data=None, dtype="float32")
            dx.main_ptr = dx_ptr
            dx.shape = input_shape
            dx.size = N * C * H * W
            return dx

        if len(input_shape) == 3:
            input_shape = (1,) + tuple(input_shape)
        N, C, H, W = input_shape

        if _USE_CPP:
            return seera_cpp.unpooling_backward(
                np.ascontiguousarray(dout, dtype=np.float32),
                N, C, H, W, sh, sw,
            )

        return dout.reshape(N, C, H, sh, W, sw).sum(axis=(3, 5))

    # ─── ConvTranspose2D backward (C++ / CUDA / NumPy) ───────
    def conv_transpose2d_backward(self, dO, X, W, strideh=1, stridew=1, paddingh=0, paddingw=0):
        gpu = _is_gpu(X)

        if gpu:
            batchN, Cin, H, Win = X.shape
            _Cin2, Cout, KH, KW = W.shape
            dX_ptr = seera_cuda.cuda_malloc_f32(X.size)
            dW_ptr = seera_cuda.cuda_malloc_f32(W.size)
            seera_cuda.cuda_conv2DTranspose_bwd(
                W.main_ptr, X.main_ptr, dO.main_ptr,
                dX_ptr, dW_ptr,
                batchN, Cin, H, Win, Cout, KH, KW,
                strideh, stridew, paddingh, paddingw,
            )
            dX = cuten(data=None, dtype="float32")
            dX.main_ptr = dX_ptr
            dX.shape = X.shape
            dX.size = X.size

            dW = cuten(data=None, dtype="float32")
            dW.main_ptr = dW_ptr
            dW.shape = W.shape
            dW.size = W.size
            return dX, dW

        if isinstance(X, np.ndarray) and X.ndim == 3: X = X[np.newaxis]
        if isinstance(dO, np.ndarray) and dO.ndim == 3: dO = dO[np.newaxis]

        if _USE_CPP:
            dX, dW = seera_cpp.conv_transpose2d_backward(
                np.ascontiguousarray(dO, dtype=np.float32),
                np.ascontiguousarray(X, dtype=np.float32),
                np.ascontiguousarray(W, dtype=np.float32),
                strideh, stridew, paddingh, paddingw,
            )
            return dX, dW

        # NumPy fallback
        N, Cin, H, Win = X.shape
        _, Cout, KH, KW = W.shape
        Hout = (H - 1) * strideh - 2 * paddingh + KH
        Wout = (Win - 1) * stridew - 2 * paddingw + KW
        col_row = Cout * KH * KW
        spatial_in = H * Win

        # im2col on dout
        col_dout = Tensor.im2col_batch(
            dO.astype(np.float32), KH, KW, strideh, stridew, paddingh, paddingw
        )  # (N, col_row, spatial_in)

        W_flat = W.reshape(Cin, col_row)
        X_flat = X.reshape(N, Cin, spatial_in)

        # dW = sum_n( X_flat[n] @ col_dout[n].T )
        dW = np.einsum('ncs,nrs->cr', X_flat, col_dout).reshape(W.shape)

        # dX_flat = W_flat @ col_dout
        dX = np.einsum('cr,nrs->ncs', W_flat, col_dout).reshape(X.shape)

        return dX, dW

    # ─── Main backward step ──────────────────────────────────
    def backward_step(self, nodeg):
        gpu = _is_gpu(nodeg.node.cp)

        # ── MaxPool backward ──
        if nodeg.mpctx[0]:
            if gpu:
                dx = self.maxpool2d_unpool(
                    nodeg.node.cp, nodeg.mpctx[1], nodeg.mpctx[2],
                    nodeg.mpctx[3], nodeg.mpctx[4], nodeg.mpctx[5], nodeg.mpctx[6], nodeg.mpctx[7],
                )
                nodeg.node.child[0].node.cp = nodeg.node.child[0].node.cp + dx
            else:
                cp = nodeg.node.cp.astype(np.float32)
                nodeg.node.child[0].node.cp += self.maxpool2d_unpool(
                    cp, nodeg.mpctx[1], nodeg.mpctx[2],
                    nodeg.mpctx[3], nodeg.mpctx[4], nodeg.mpctx[5], nodeg.mpctx[6], nodeg.mpctx[7],
                )

        # ── Unpooling backward ──
        elif nodeg.unpoolctx[0]:
            if gpu:
                dx = self.unpooling_backward(
                    nodeg.node.cp, nodeg.unpoolctx[1], nodeg.unpoolctx[2],
                )
                nodeg.node.child[0].node.cp = nodeg.node.child[0].node.cp + dx
            else:
                cp = nodeg.node.cp.astype(np.float32)
                nodeg.node.child[0].node.cp += self.unpooling_backward(
                    cp, nodeg.unpoolctx[1], nodeg.unpoolctx[2],
                )

        # ── ConvTranspose2D backward ──
        elif nodeg.convTrans[0]:
            X = nodeg.node.child[0].value
            W = nodeg.node.child[1].value
            if gpu:
                dX, dW = self.conv_transpose2d_backward(nodeg.node.cp, X, W,
                                                         strideh=nodeg.convTrans[1],
                                                         stridew=nodeg.convTrans[2],
                                                         paddingh=nodeg.convTrans[3],
                                                         paddingw=nodeg.convTrans[4])
                nodeg.node.child[0].node.cp = nodeg.node.child[0].node.cp + dX
                nodeg.node.child[1].node.cp = nodeg.node.child[1].node.cp + dW
            else:
                cp = nodeg.node.cp.astype(np.float32)
                dX, dW = self.conv_transpose2d_backward(cp, X, W,
                                                         strideh=nodeg.convTrans[1],
                                                         stridew=nodeg.convTrans[2],
                                                         paddingh=nodeg.convTrans[3],
                                                         paddingw=nodeg.convTrans[4])
                nodeg.node.child[0].node.cp += dX
                nodeg.node.child[1].node.cp += dW

        # ── Flatten backward ──
        elif nodeg.flctx:
            if gpu:
                # reshape the gradient back to original shape
                reshaped = nodeg.node.cp.reshape(nodeg.flctx)
                nodeg.node.child[0].node.cp = nodeg.node.child[0].node.cp + reshaped
            else:
                nodeg.node.child[0].node.cp += nodeg.node.cp.reshape(nodeg.flctx)

        # ── Transpose backward ──
        elif nodeg.itranspose:
            cp = nodeg.node.cp
            if gpu:
                nodeg.node.child[0].node.cp = nodeg.node.child[0].node.cp + cp.T
            else:
                nodeg.node.child[0].node.cp += cp.T

        # ── Conv2D backward ──
        elif nodeg.iconv2d[0]:
            X = nodeg.node.child[0].value
            W = nodeg.node.child[1].value
            if gpu:
                dX, dW = self.conv_backward(nodeg.node.cp, X, W,
                                            strideh=nodeg.iconv2d[1],
                                            stridew=nodeg.iconv2d[2],
                                            paddingh=nodeg.iconv2d[3],
                                            paddingw=nodeg.iconv2d[4])
                nodeg.node.child[0].node.cp = nodeg.node.child[0].node.cp + dX
                nodeg.node.child[1].node.cp = nodeg.node.child[1].node.cp + dW
            else:
                cp = nodeg.node.cp.astype(np.float32)
                dX, dW = self.conv_backward(cp, X, W,
                                            strideh=nodeg.iconv2d[1],
                                            stridew=nodeg.iconv2d[2],
                                            paddingh=nodeg.iconv2d[3],
                                            paddingw=nodeg.iconv2d[4])
                nodeg.node.child[0].node.cp += dX
                nodeg.node.child[1].node.cp += dW

        # ── Softmax backward (VJP, C++ / CUDA accelerated) ──
        elif nodeg.isoftmax:
            s = nodeg.node.child_grad[0, 1]
            dout = nodeg.node.cp
            if gpu:
                # s and dout are cuten objects
                if len(s.shape) < 2:
                    raise ValueError("[Engine]: Softmax VJP requires at least 2D")
                N = 1
                for d in s.shape[:-1]:
                    N *= d
                C = s.shape[-1]
                dx_ptr = seera_cuda.cuda_malloc_f32(s.size)
                seera_cuda.cuda_softmax_vjp(s.main_ptr, dout.main_ptr, dx_ptr, N, C)
                dx = cuten(data=None, dtype="float32")
                dx.main_ptr = dx_ptr
                dx.shape = s.shape
                dx.size = s.size
                nodeg.node.child[0].node.cp = nodeg.node.child[0].node.cp + dx
            else:
                if _USE_CPP and s.ndim >= 2:
                    dx = seera_cpp.softmax_vjp(
                        np.ascontiguousarray(s, dtype=np.float32),
                        np.ascontiguousarray(dout, dtype=np.float32),
                    )
                else:
                    dot = np.sum(dout * s, axis=-1, keepdims=True)
                    dx = s * (dout - dot)
                nodeg.node.child[0].node.cp += dx

        # ── BatchNorm backward (C++ accelerated, NO GPU) ──
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
            if gpu:
                child0_val = nodeg.node.child[0].value
                child1_val = nodeg.node.child[1].value
                if len(cp.shape) >= 4:
                    dx1, dx2 = cuten.concatenate2D_backward(cp, child0_val, child1_val)
                else:
                    dx1, dx2 = cuten.concatenate1D_backward(cp, child0_val, child1_val)
                nodeg.node.child[0].node.cp = nodeg.node.child[0].node.cp + dx1
                nodeg.node.child[1].node.cp = nodeg.node.child[1].node.cp + dx2
            else:
                if cp.ndim >= 4:
                    nodeg.node.child[0].node.cp += cp[:, :split]
                    nodeg.node.child[1].node.cp += cp[:, split:]
                else:
                    nodeg.node.child[0].node.cp += cp[:split]
                    nodeg.node.child[1].node.cp += cp[split:]

        # ── Reduction (sum/mean/max/min) backward ──
        elif nodeg.ireduction is not None:
            ctx = nodeg.ireduction
            cp = nodeg.node.cp
            if ctx.get("gpu", False):
                input_shape = ctx["input_shape"]
                ndims = ctx["ndims"]
                dim = ctx["dim"]
                dimarr = ctx["dimarr"]
                
                in_size = 1
                for d in input_shape:
                    in_size *= d

                input_shape_, in_size_, ndims, dim, dimarr = self._reduction_meta(dim,ndims,dimarr,input_shape)
                
                rtype = ctx.get("type", None)

                if rtype == "max":
                    dA_ptr = seera_cuda.cuda_malloc_f32(in_size)
                    seera_cuda.cuda_memset(dA_ptr, 0, in_size * 4)
                    seera_cuda.cuda_max_bwd(
                        cp.main_ptr, ctx["fwdInput"].main_ptr, ctx["fwdOutput"].main_ptr,
                        dA_ptr, ndims, dim, dimarr,
                    )
                    dA = cuten(data=None, dtype="float32")
                    dA.main_ptr = dA_ptr
                    dA.shape = input_shape
                    dA.size = in_size
                    nodeg.node.child[0].node.cp = nodeg.node.child[0].node.cp + dA

                elif rtype == "min":
                    dA_ptr = seera_cuda.cuda_malloc_f32(in_size)
                    seera_cuda.cuda_memset(dA_ptr, 0, in_size * 4)
                    seera_cuda.cuda_min_bwd(
                        cp.main_ptr, ctx["fwdInput"].main_ptr, ctx["fwdOutput"].main_ptr,
                        dA_ptr, ndims, dim, dimarr,
                    )
                    dA = cuten(data=None, dtype="float32")
                    dA.main_ptr = dA_ptr
                    dA.shape = input_shape
                    dA.size = in_size
                    nodeg.node.child[0].node.cp = nodeg.node.child[0].node.cp + dA

                elif ctx["scale"] == 1.0:
                    # sum backward
                    dA_ptr = seera_cuda.cuda_malloc_f32(in_size)
                    
                    seera_cuda.cuda_sum_bwd(cp.main_ptr, dA_ptr, ndims, dim, dimarr)
                    dA = cuten(data=None, dtype="float32")
                    dA.main_ptr = dA_ptr
                    dA.shape = input_shape
                    dA.size = in_size
                    nodeg.node.child[0].node.cp = nodeg.node.child[0].node.cp + dA

                else:
                    # mean backward
                    dA_ptr = seera_cuda.cuda_malloc_f32(in_size)
                    seera_cuda.cuda_mean_bwd(cp.main_ptr, dA_ptr, ndims, dim, dimarr)
                    
                    dA = cuten(data=None, dtype="float32")
                    dA.main_ptr = dA_ptr
                    dA.shape = input_shape
                    dA.size = in_size
                    nodeg.node.child[0].node.cp = nodeg.node.child[0].node.cp + dA

            else:
                # CPU reduction backward
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

        # ── Matmul backward (C++ / CUDA accelerated) ──
        elif nodeg.matm:
            A = nodeg.node.child[0]
            B = nodeg.node.child[1]
            dout = nodeg.node.cp

            if gpu:
                # A.value, B.value are cuten; dout is cuten
                M = A.value.shape[-2]
                K = A.value.shape[-1]
                N = B.value.shape[-1]

                Nbatch = 1
                if len(A.value.shape)>2:
                    Nbatch=  A.value.shape[0]


                dA_ptr = seera_cuda.cuda_malloc_f32(A.value.size)
                dB_ptr = seera_cuda.cuda_malloc_f32(B.value.size)

                seera_cuda.cuda_matmul_bwd(
                    A.value.main_ptr, B.value.main_ptr, dout.main_ptr,
                    dA_ptr, dB_ptr,
                    M, N, K, Nbatch,
                )

                dA = cuten(data=None, dtype="float32")
                dA.main_ptr = dA_ptr
                dA.shape = A.value.shape
                dA.size = A.value.size

                dB = cuten(data=None, dtype="float32")
                dB.main_ptr = dB_ptr
                dB.shape = B.value.shape
                dB.size = B.value.size

                A.node.cp = A.node.cp + dA
                B.node.cp = B.node.cp + dB
            else:
                dout = np.ascontiguousarray(dout, dtype=np.float32)
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
                    cp = nodeg.node.cp

                    if gpu:
                        # Both local_grad and cp are cuten
                        raw_grad = local_grad * cp
                        # Broadcast gradient reduction on GPU
                        if raw_grad.shape != child.value.shape:
                            raw_grad = self._reduce_grad_gpu(raw_grad, child.value.shape)
                        child.node.cp = child.node.cp + raw_grad
                    else:
                        if isinstance(local_grad, np.ndarray):
                            local_grad = np.asarray(local_grad, dtype=np.float32)
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
        gpu = _is_gpu(self.hook.value)
        if gpu:
            self.hook.node.cp = cuten.ones_like(self.hook.value)
        else:
            self.hook.node.cp = np.ones_like(self.hook.value, dtype=np.float32)

        graph = list(self.buildgraph())
        for nodeg in graph:
            for child in nodeg.node.child:
                if gpu:
                    if not isinstance(child.node.cp, cuten) or child.node.cp.shape != child.value.shape:
                        child.node.cp = cuten.zeros_like(child.value)
                else:
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
    
    @staticmethod
    def _reduction_meta( dim: int, ndims:int,dimarr,input_shape):
        """Return (out_shape, out_size, ndims, dim, dimarr_numpy) for a
        reduction that collapses *dim* from self.shape."""
        if dim < 0:
            dim = ndims + dim
        if dim < 0 or dim >= ndims:
            raise ValueError(f"[Engine]: dim {dim} out of range for shape {input_shape}")

        out_shape = tuple(d for i, d in enumerate(input_shape) if i != dim)
        if not out_shape:
            out_shape = (1,)
        out_size = 1
        for d in out_shape:
            out_size *= d

        dimarr = np.array(input_shape, dtype=np.int32)
        return out_shape, out_size, ndims, dim, dimarr


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