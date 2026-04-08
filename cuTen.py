from __future__ import annotations
import numpy as np
import seera_cuda
# cuten is a library, for all the GPU operations, just like numpy, infact any value it takes in is directly transferrd
# to GPU, it supports all the operations as listed in seera_engine_cuda. but ONLY operates and not tracks which will be done by tensor package. 
# the tensor class will take in cuten tensors just like it took numpy tensors. 
# so hence depending upon the type this will significatly ease the the architecture.
# Fun
class cuten:
    # ── Broadcasting helpers ─────────────────────────────────────────
    @staticmethod
    def _pad_shape_4d(shape):
        """Left-pad shape to 4 dimensions with 1s."""
        s = tuple(shape)
        while len(s) < 4:
            s = (1,) + s
        return s

    @staticmethod
    def _broadcast_out_shape(a_shape, b_shape):
        """Compute the broadcast output shape (NumPy rules)."""
        ndim = max(len(a_shape), len(b_shape))
        a_pad = (1,) * (ndim - len(a_shape)) + tuple(a_shape)
        b_pad = (1,) * (ndim - len(b_shape)) + tuple(b_shape)
        out = []
        for a, b in zip(a_pad, b_pad):
            if a == b:
                out.append(a)
            elif a == 1:
                out.append(b)
            elif b == 1:
                out.append(a)
            else:
                raise ValueError(f"[cuTen]: Shapes {a_shape} and {b_shape} not broadcastable")
        return tuple(out)
    def __init__(self, data, dtype="float32"):
        self.supported_types = ["float32","int32","int16"]
        self.fill_alloc_dtype = {
            "float32":seera_cuda.to_device_f32,
            "int32":seera_cuda.to_device_i32,
            "int16":seera_cuda.to_device_i16
            
        }
        if dtype not in self.supported_types :
            raise ValueError("[cuTen]:Not Supported by cuTen, try in CPU")

        if isinstance(data, np.ndarray):
            # Transfer numpy array to GPU immediately
            arr = data.astype(dtype)
            self._allocate_convert_to_gpu(arr,arr.shape,arr.size,dtype)
 

        elif isinstance(data, (list, tuple)):
            # Convert to numpy first, then to GPU
            arr = np.array(data, dtype=dtype)
            self._allocate_convert_to_gpu(arr,arr.shape,arr.size,dtype)
            
        else:
            self.main_ptr = None; 
            self.shape = None
            self.size = None
            self.dtype = dtype
                
            

    def _allocate_convert_to_gpu(self,data:np.ndarray, shape, size, dtype):
        # print(f"the dtype is {dtype}")
        self.main_ptr = self.fill_alloc_dtype[dtype](data)
        self.shape = shape
        self.size = size
        self.dtype = dtype
    
    def to_host_f32(self):
        return seera_cuda.to_host_f32(self.main_ptr,self.shape)
        
    def to_host_i32(self):
        return seera_cuda.to_host_i32(self.main_ptr,self.shape)
    
    @classmethod
    def ones_like(cls, arr:cuten):
        return cuten(np.ones(arr.shape,arr.dtype))
    
    
    @classmethod
    def ones_like_fromnumpy(cls, arr:np.ndarray):
        return cuten(arr,dtype=str(arr.dtype))
    
    
    @classmethod
    def zeros_like(cls, arr:cuten):
        return cuten(np.zeros(arr.shape,arr.dtype))
    
    
    @classmethod
    def zeros_like_fromnumpy(cls, arr:np.ndarray):
        return cuten(arr,dtype=str(arr.dtype))

    @classmethod
    def zeros(cls, shape):
        shape = tuple(shape)
        size = 1 ; 
        for dim in shape:
            size *= dim
        main_ptr = seera_cuda.cuda_zeros_f(size)
        
        sel = cuten(data=None,dtype="float32") 
        sel.main_ptr = main_ptr
        sel.shape = shape
        sel.size = size
        return sel
     
    @classmethod
    def ones(cls, shape):
        shape = tuple(shape)
        size = 1 ; 
        for dim in shape:
            size *= dim
        main_ptr = seera_cuda.cuda_ones_f(size)
        
        sel = cuten(data=None,dtype="float32") 

        sel.main_ptr = main_ptr
        sel.shape = shape
        sel.size = size
        
        return sel        

    def __add__(self, other):
        if isinstance(other,cuten):
            if self.shape == other.shape:
                # Fast path: same shape, use element-wise kernel
                c = cuten.ones_like(self)
                seera_cuda.cuda_elemadd(self.main_ptr, other.main_ptr, c.main_ptr, self.size)
                return c
            # Broadcast path
            out_shape = cuten._broadcast_out_shape(self.shape, other.shape)
            a4 = cuten._pad_shape_4d(self.shape)
            b4 = cuten._pad_shape_4d(other.shape)
            out_size = seera_cuda.compute_out_size_4d(*a4, *b4)
            if out_size < 0:
                raise ValueError(f"[cuTen]: Shapes {self.shape} and {other.shape} not broadcastable")
            out_ptr = seera_cuda.cuda_malloc_f32(out_size)
            seera_cuda.broadcast_add_4d(self.main_ptr, other.main_ptr, out_ptr, *a4, *b4)
            result = cuten(data=None, dtype="float32")
            result.main_ptr = out_ptr
            result.shape = out_shape
            result.size = out_size
            return result
            
        if isinstance(other,int) or isinstance(other,float):
            seera_cuda.cuda_scaler_add_f(self.main_ptr, float(other),self.size )
        
            return self 
    
    def __mul__(self, other):
    
        if isinstance(other,cuten):
            if self.shape == other.shape:
                # Fast path: same shape, use element-wise kernel
                c = cuten.ones_like(self)
                seera_cuda.cuda_elemmult(self.main_ptr, other.main_ptr, c.main_ptr, self.size)
                return c
            # Broadcast path
            out_shape = cuten._broadcast_out_shape(self.shape, other.shape)
            a4 = cuten._pad_shape_4d(self.shape)
            b4 = cuten._pad_shape_4d(other.shape)
            out_size = seera_cuda.compute_out_size_4d(*a4, *b4)
            if out_size < 0:
                raise ValueError(f"[cuTen]: Shapes {self.shape} and {other.shape} not broadcastable")
            out_ptr = seera_cuda.cuda_malloc_f32(out_size)
            seera_cuda.broadcast_mul_4d(self.main_ptr, other.main_ptr, out_ptr, *a4, *b4)
            result = cuten(data=None, dtype="float32")
            result.main_ptr = out_ptr
            result.shape = out_shape
            result.size = out_size
            return result
            
        if isinstance(other,int) or isinstance(other,float):
            seera_cuda.cuda_scaler_multiply_f(self.main_ptr, float(other),self.size )
        
            return self 
        
        raise ValueError("[cuTen]: Not Supported, Can be Device Mismatch")
        
        
    def __pow__(self, other):
        if isinstance(other,int) or isinstance(other,float):
            seera_cuda.cuda_power_of(self.main_ptr, float(other),self.size )
        
            return self 
    
    def __neg__(self):    return self * (-1)
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rsub__(self, other): return other + (self * -1)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1
    
    def reshape(self,shape:tuple):
        size = 1 ; 
        for dim in shape:
            size *= dim
        if self.size != size:
            raise ValueError(f"[cuTen]: Cannot reshape cuten of size: {self.size} to shape {shape}")
        
        self.shape = shape
        
    def __repr__(self):
        np_local = seera_cuda.to_host_f32(self.main_ptr,self.shape)
        print(np_local)
        
        return "IT DOES WORK"

    # ==================================================================
    #  Helper: run a unary activation kernel that writes (out, grad)
    #  cuTen doesn't track grad so we allocate a throwaway buffer.
    # ==================================================================
    def _unary_activation(self, kernel_fn):
        """Run a CUDA unary activation that fills (out, grad). Returns a new cuten with the out buffer."""
        out_ptr = seera_cuda.cuda_malloc_f32(self.size)
        grad_ptr = seera_cuda.cuda_malloc_f32(self.size)   # throwaway
        kernel_fn(self.main_ptr, out_ptr, grad_ptr, self.size)
        seera_cuda.cuda_free(grad_ptr)

        result = cuten(data=None, dtype="float32")
        result.main_ptr = out_ptr
        result.shape = self.shape
        result.size = self.size
        return result

    # ==================================================================
    #  1) ReLU
    # ==================================================================
    def relu(self) -> cuten:
        return self._unary_activation(seera_cuda.cuda_relu_fwd)

    # ==================================================================
    #  2) Sigmoid
    # ==================================================================
    def sigmoid(self) -> cuten:
        return self._unary_activation(seera_cuda.cuda_sigmoid_fwd)

    # ==================================================================
    #  3) Tanh
    # ==================================================================
    def tanh(self) -> cuten:
        return self._unary_activation(seera_cuda.cuda_tanh_fwd)

    # ==================================================================
    #  4) Log
    # ==================================================================
    def log(self) -> cuten:
        return self._unary_activation(seera_cuda.cuda_log_fwd)

    # ==================================================================
    #  5) Exp
    # ==================================================================
    def exp(self) -> cuten:
        return self._unary_activation(seera_cuda.cuda_exp_fwd)

    # ==================================================================
    #  6) Absolute
    # ==================================================================
    def absolute(self) -> cuten:
        return self._unary_activation(seera_cuda.cuda_abs_fwd)

    def abs(self) -> cuten:
        """Alias for absolute()."""
        return self.absolute()

    # ==================================================================
    #  7) Sqrt
    # ==================================================================
    def sqrt(self) -> cuten:
        return self._unary_activation(seera_cuda.cuda_sqrt_fwd)

    # ==================================================================
    #  8) Clip
    # ==================================================================
    def clip(self, lo: float, hi: float) -> cuten:
        """Element-wise clamp to [lo, hi]."""
        out_ptr = seera_cuda.cuda_malloc_f32(self.size)
        grad_ptr = seera_cuda.cuda_malloc_f32(self.size)
        seera_cuda.cuda_clip_fwd(self.main_ptr, float(lo), float(hi),
                                 out_ptr, grad_ptr, self.size)
        seera_cuda.cuda_free(grad_ptr)

        result = cuten(data=None, dtype="float32")
        result.main_ptr = out_ptr
        result.shape = self.shape
        result.size = self.size
        return result

    # ==================================================================
    #  9) Softmax  (along the last dimension)
    # ==================================================================
    def softmax(self) -> cuten:
        """Softmax over the last dimension. Input must be >= 2-D."""
        if len(self.shape) < 2:
            raise ValueError("[cuTen]: softmax requires at least a 2-D tensor")
        N = 1
        for d in self.shape[:-1]:
            N *= d
        C = self.shape[-1]

        out_ptr = seera_cuda.cuda_malloc_f32(self.size)
        seera_cuda.cuda_softmax_fwd(self.main_ptr, out_ptr, N, C)

        result = cuten(data=None, dtype="float32")
        result.main_ptr = out_ptr
        result.shape = self.shape
        result.size = self.size
        return result

    # ==================================================================
    #  10) Matmul   A @ B  (supports batched GEMM)
    # ==================================================================
    def matmul(self, other: cuten) -> cuten:
        """
        Matrix multiply self @ other.
        self  : shape (..., M, K)  or (M, K)
        other : shape (..., K, N)  or (K, N)
        Batch dimensions of self and other must match (if present).
        Uses cuda_matmul(A, B, C, M, N, K, Nbatch).
        """
        if len(self.shape) < 2 or len(other.shape) < 2:
            raise ValueError("[cuTen]: matmul requires at least 2-D tensors")

        M = self.shape[-2]
        K = self.shape[-1]
        N = other.shape[-1]
        if other.shape[-2] != K:
            raise ValueError(
                f"[cuTen]: matmul: inner dimensions must agree, got {self.shape} @ {other.shape}")

        # batch count
        self_batch = self.shape[:-2]
        other_batch = other.shape[:-2]

        if self_batch and other_batch and self_batch != other_batch:
            raise ValueError(
                f"[cuTen]: matmul: batch dims must match, got {self_batch} vs {other_batch}")
        batch_shape = self_batch if self_batch else other_batch
        Nbatch = 1
        for d in batch_shape:
            Nbatch *= d

        out_shape = tuple(batch_shape) + (M, N) if batch_shape else (M, N)
        out_size = 1
        for d in out_shape:
            out_size *= d

        out_ptr = seera_cuda.cuda_malloc_f32(out_size)
        seera_cuda.cuda_matmul(self.main_ptr, other.main_ptr, out_ptr,
                               M, N, K, Nbatch)

        result = cuten(data=None, dtype="float32")
        result.main_ptr = out_ptr
        result.shape = out_shape
        result.size = out_size
        return result

    def __matmul__(self, other):
        return self.matmul(other)

    # ==================================================================
    #  Helper: compute output shape & dimarr for reduction ops
    # ==================================================================
    def _reduction_meta(self, dim: int):
        """Return (out_shape, out_size, ndims, dim, dimarr_numpy) for a
        reduction that collapses *dim* from self.shape."""
        ndims = len(self.shape)
        if dim < 0:
            dim = ndims + dim
        if dim < 0 or dim >= ndims:
            raise ValueError(f"[cuTen]: dim {dim} out of range for shape {self.shape}")

        out_shape = tuple(d for i, d in enumerate(self.shape) if i != dim)
        if not out_shape:
            out_shape = (1,)
        out_size = 1
        for d in out_shape:
            out_size *= d

        dimarr = np.array(self.shape, dtype=np.int32)
        return out_shape, out_size, ndims, dim, dimarr

    # ==================================================================
    #  11) Sum  — reduction along a given dimension
    # ==================================================================
    def sum(self, dim: int) -> cuten:
        out_shape, out_size, ndims, dim, dimarr = self._reduction_meta(dim)
        out_ptr = seera_cuda.cuda_malloc_f32(out_size)
        seera_cuda.cuda_sum_fwd(self.main_ptr, out_ptr, ndims, dim, dimarr)

        result = cuten(data=None, dtype="float32")
        result.main_ptr = out_ptr
        result.shape = out_shape
        result.size = out_size
        return result

    # ==================================================================
    #  12) Mean — reduction along a given dimension
    # ==================================================================
    def mean(self, dim: int) -> cuten:
        out_shape, out_size, ndims, dim, dimarr = self._reduction_meta(dim)
        out_ptr = seera_cuda.cuda_malloc_f32(out_size)
        seera_cuda.cuda_mean_fwd(self.main_ptr, out_ptr, ndims, dim, dimarr)

        result = cuten(data=None, dtype="float32")
        result.main_ptr = out_ptr
        result.shape = out_shape
        result.size = out_size
        return result

    # ==================================================================
    #  13) Max — reduction along a given dimension
    # ==================================================================
    def max(self, dim: int) -> cuten:
        out_shape, out_size, ndims, dim, dimarr = self._reduction_meta(dim)
        out_ptr = seera_cuda.cuda_malloc_f32(out_size)
        seera_cuda.cuda_max_fwd(self.main_ptr, out_ptr, ndims, dim, dimarr)

        result = cuten(data=None, dtype="float32")
        result.main_ptr = out_ptr
        result.shape = out_shape
        result.size = out_size
        return result

    # ==================================================================
    #  14) Min — reduction along a given dimension
    # ==================================================================
    def min(self, dim: int) -> cuten:
        out_shape, out_size, ndims, dim, dimarr = self._reduction_meta(dim)
        out_ptr = seera_cuda.cuda_malloc_f32(out_size)
        seera_cuda.cuda_min_fwd(self.main_ptr, out_ptr, ndims, dim, dimarr)

        result = cuten(data=None, dtype="float32")
        result.main_ptr = out_ptr
        result.shape = out_shape
        result.size = out_size
        return result

    # ==================================================================
    #  15) Conv2D   — NCHW layout
    #      self  : (N, C, H, W)     — input feature map
    #      kernel: (F, C, KH, KW)   — filter bank
    # ==================================================================
    def conv2d(self, kernel: cuten, strideh: int = 1, stridew: int = 1,
               padh: int = 0, padw: int = 0) -> cuten:
        """Forward conv2d. self is image (N,C,H,W), kernel is (F,C,KH,KW)."""
        if len(self.shape) != 4 or len(kernel.shape) != 4:
            raise ValueError("[cuTen]: conv2d expects 4-D tensors (N,C,H,W) and (F,C,KH,KW)")

        batchN, C, H, W = self.shape
        F, _Ck, KH, KW = kernel.shape

        OH = (H + 2 * padh - KH) // strideh + 1
        OW = (W + 2 * padw - KW) // stridew + 1
        out_shape = (batchN, F, OH, OW)
        out_size = batchN * F * OH * OW

        out_ptr = seera_cuda.cuda_malloc_f32(out_size)
        seera_cuda.cuda_conv2d_fwd(self.main_ptr, kernel.main_ptr, out_ptr,
                                   batchN, C, H, W, F, KH, KW,
                                   padh, padw, strideh, stridew)

        result = cuten(data=None, dtype="float32")
        result.main_ptr = out_ptr
        result.shape = out_shape
        result.size = out_size
        return result

    # ==================================================================
    #  16) MaxPool2D  — NCHW layout
    #      Returns (pooled_output, mask_cuten)
    #      mask is int16 on GPU — stored as dtype "int16" cuten
    # ==================================================================
    def maxpool2d(self, KH: int, KW: int,
                  strideh: int = 1, stridew: int = 1,
                  padh: int = 0, padw: int = 0):
        """Forward max-pool. Returns (output, mask) where mask is int16 cuten."""
        if len(self.shape) != 4:
            raise ValueError("[cuTen]: maxpool2d expects 4-D tensor (N,C,H,W)")

        batchN, C, H, W = self.shape
        OH = (H + 2 * padh - KH) // strideh + 1
        OW = (W + 2 * padw - KW) // stridew + 1
        out_shape = (batchN, C, OH, OW)
        out_size = batchN * C * OH * OW

        out_ptr = seera_cuda.cuda_malloc_f32(out_size)
        mask_ptr = seera_cuda.cuda_malloc_i16(out_size)
        seera_cuda.cuda_maxpool_fwd(self.main_ptr, out_ptr, mask_ptr,
                                    batchN, C, H, W, KH, KW,
                                    padh, padw, strideh, stridew)

        result = cuten(data=None, dtype="float32")
        result.main_ptr = out_ptr
        result.shape = out_shape
        result.size = out_size

        mask = cuten(data=None, dtype="int16")
        mask.main_ptr = mask_ptr
        mask.shape = out_shape
        mask.size = out_size

        return result, mask

    # ==================================================================
    #  17) Unpool (nearest-neighbour upsample)
    #      self : (N, C, H, W)
    #      sh, sw : scale factors
    # ==================================================================
    def unpool(self, sh: int, sw: int) -> cuten:
        """Nearest-neighbour upsample by factors (sh, sw)."""
        if len(self.shape) != 4:
            raise ValueError("[cuTen]:unpool expects 4-D tensor (N,C,H,W)")

        batchN, C, H, W = self.shape
        out_shape = (batchN, C, H * sh, W * sw)
        out_size = batchN * C * (H * sh) * (W * sw)

        out_ptr = seera_cuda.cuda_malloc_f32(out_size)
        seera_cuda.cuda_unpooling_fwd(self.main_ptr, out_ptr,
                                      batchN, C, H, W, sh, sw)

        result = cuten(data=None, dtype="float32")
        result.main_ptr = out_ptr
        result.shape = out_shape
        result.size = out_size
        return result

    # ==================================================================
    #  18) ConvTranspose2D
    #      self   : (N, Cin, H, W)          — input
    #      kernel : (Cin, Cout, KH, KW)     — transposed-conv filter
    # ==================================================================
    def conv2d_transpose(self, kernel: cuten,
                         strideh: int = 1, stridew: int = 1,
                         padh: int = 0, padw: int = 0) -> cuten:
        """Forward transposed convolution."""
        if len(self.shape) != 4 or len(kernel.shape) != 4:
            raise ValueError(
                "[cuTen]: conv2d_transpose expects 4-D tensors (N,Cin,H,W) and (Cin,Cout,KH,KW)")

        batchN, Cin, H, W = self.shape
        _Cin2, Cout, KH, KW = kernel.shape
        Hout = (H - 1) * strideh - 2 * padh + KH
        Wout = (W - 1) * stridew - 2 * padw + KW
        out_shape = (batchN, Cout, Hout, Wout)
        out_size = batchN * Cout * Hout * Wout

        out_ptr = seera_cuda.cuda_malloc_f32(out_size)
        seera_cuda.cuda_conv2DTranpose_fwd(self.main_ptr, kernel.main_ptr,
                                           out_ptr, batchN, Cin, H, W,
                                           Cout, KH, KW,
                                           strideh, stridew, padh, padw)

        result = cuten(data=None, dtype="float32")
        result.main_ptr = out_ptr
        result.shape = out_shape
        result.size = out_size
        return result
    
    # ==================================================================
    #  19) Concatenate
    #       for 1D case:
    #         self: (N,n1) , other (N,n2) -> reten (N,n1+n2)
    #       for 2D case:
    #         self: (N,c1,H,W) , other (N,c2,H,W) -> reten (N,c1+c2,H,W)
    
    # ================================================================== 
    
    @staticmethod
    def _concatenate(ptr1, ptr2, ptr3 , n1, n2):
        print(type(ptr1))
        
        seera_cuda.cuda_memcopy_devicetodevice(ptr3,ptr1,n1)
        seera_cuda.cuda_memcopy_devicetodevice(ptr3+n1*4,ptr2,n2) 

        return ptr3 
    
    @staticmethod
    def _concatenate_backward(ptr_out, ptr1, ptr2, n1, n2):
        # dout → dx1
        seera_cuda.cuda_memcopy_devicetodevice(ptr1, ptr_out, n1)
        
        # dout → dx2 (offset by n1)
        seera_cuda.cuda_memcopy_devicetodevice(ptr2, ptr_out + n1 * 4, n2)
    
    def concatenate2D(self,other):
        if(self.shape[0]!=other.shape[0]):
            raise ValueError(f"[cuTen]: No of Batches should be same, received {self.shape[0]} and {other.shape[0]}")
        if(self.shape[3]!=other.shape[3] or self.shape[2]!=other.shape[2]):
            raise ValueError(f"[cuTen]: H x W should be same, received ({self.shape[2]},{self.shape[3]}) and ({other.shape[2]},{other.shape[3]})")
        N = self.shape[0]
        n1 = int(self.size/N)
        n2 = int(other.size/N)
        new_ptr = seera_cuda.cuda_malloc_f32(self.size+other.size) 
        print(type(new_ptr))
        for i in range (0,N):
            ptr1 = self.main_ptr + (n1*i*4)
            ptr2 = other.main_ptr + (n2*i*4)      
            ptr3 = new_ptr + ((n1+n2)*i*4)
            
            self._concatenate(ptr1,ptr2,ptr3,n1,n2)

        reten = cuten(data=None)
        reten.main_ptr = new_ptr
        reten.size = self.size+other.size
        reten.shape = (self.shape[0], self.shape[1]+other.shape[1],self.shape[2],self.shape[3])
        
        return reten
    
    @classmethod
    def concatenate2D_backward(cls, dout, self, other):
        if dout.shape[0] != self.shape[0]:
            raise ValueError("[cuTen]: Batch mismatch in backward")

        N = self.shape[0]
        n1 = int(self.size / N)
        n2 = int(other.size / N)

        # allocate gradients
        dx1_ptr = seera_cuda.cuda_malloc_f32(self.size)
        dx2_ptr = seera_cuda.cuda_malloc_f32(other.size)

        for i in range(N):
            ptr_out = dout.main_ptr + ((n1 + n2) * i * 4)
            ptr1 = dx1_ptr + (n1 * i * 4)
            ptr2 = dx2_ptr + (n2 * i * 4)

            cls._concatenate_backward(ptr_out, ptr1, ptr2, n1, n2)

        dx1 = cuten(data=None)
        dx1.main_ptr = dx1_ptr
        dx1.size = self.size
        dx1.shape = self.shape

        dx2 = cuten(data=None)
        dx2.main_ptr = dx2_ptr
        dx2.size = other.size
        dx2.shape = other.shape

        return dx1, dx2
    
        
    def concatenate1D(self,other):
        if(self.shape[0]!=other.shape[0]):
            raise ValueError(f"[cuTen]: No of Batches should be same, received {self.shape[0]} and {other.shape[0]}")
        
        N = self.shape[0]
        n1 = self.shape[1]
        n2 = other.shape[1]
        new_ptr = seera_cuda.cuda_malloc_f32(self.size+other.size) 
        
        for i in range (0,N):
            ptr1 = self.main_ptr + (n1*i*4)
            ptr2 = other.main_ptr + (n2*i*4)      
            ptr3 = new_ptr + ((n1+n2)*i*4)
            
            self._concatenate(ptr1,ptr2,ptr3,n1,n2)

        reten = cuten(data=None)
        reten.main_ptr = new_ptr
        reten.size = self.size+other.size
        reten.shape = (self.shape[0], self.shape[1]+other.shape[1])
        
        return reten
    
    @classmethod
    def concatenate1D_backward(cls, dout, self, other):
        if dout.shape[0] != self.shape[0]:
            raise ValueError("[cuTen]: Batch mismatch in backward")

        N = self.shape[0]
        n1 = self.shape[1]
        n2 = other.shape[1]

        dx1_ptr = seera_cuda.cuda_malloc_f32(self.size)
        dx2_ptr = seera_cuda.cuda_malloc_f32(other.size)

        for i in range(N):
            ptr_out = dout.main_ptr + ((n1 + n2) * i * 4)
            ptr1 = dx1_ptr + (n1 * i * 4)
            ptr2 = dx2_ptr + (n2 * i * 4)

            cls._concatenate_backward(ptr_out, ptr1, ptr2, n1, n2)

        dx1 = cuten(data=None)
        dx1.main_ptr = dx1_ptr
        dx1.size = self.size
        dx1.shape = self.shape

        dx2 = cuten(data=None)
        dx2.main_ptr = dx2_ptr
        dx2.size = other.size
        dx2.shape = other.shape

        return dx1, dx2 
    
    # ==================================================================
    # 20) Flatten input should be batched
    # ================================================================== 
    def flatten(self):
        self.reshape((self.shape[0],int(self.size/self.shape[0])))    
            
                 
                  
        