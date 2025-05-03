import numpy as np
from numba import njit,jit
class node:
    def __init__(self, child_grad, out=0, node_no=0):
        self.out = out
        self.child_grad = np.array(child_grad, dtype=object)
        # Initialize gradient accumulator with zeros
        if isinstance(child_grad, list) and len(child_grad) > 0 and len(child_grad[0]) > 0:
            self.cp = np.zeros_like(child_grad[0][0])
        else:
            self.cp = 0
        self.child = []
    @property
    def grad(self):
        return self.cp

class tensor(node):
    def __init__(self, value, dtype="float32", is_leaf=False):
        self.value = np.array(value).astype(dtype)
        child_grad = ([np.zeros_like(self.value), np.zeros_like(self.value)],
                      [np.zeros_like(self.value), np.zeros_like(self.value)])
        self.node = node(child_grad)
        self.is_leaf = is_leaf
        self.dtype = dtype
        ###abnormal gradients###
        self.matm = False
        self.isoftmax=False
        self.iconv2d=(False,1,0)
        self.upctx=(False,0,0)
        self.flctx=0#flatten
        self.mpctx=(False,0,0,0,1,0)
        self.iconcatenete=0
        # print(self.upctx)
        
        self.convTrans=False
        ########################
        if is_leaf:
            self.node.out = self.value
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = tensor(other * np.ones_like(self.value))
        elif not isinstance(other, tensor):
            other = tensor(other)
        out = tensor(self.value + other.value)
        child_grad = [[self.value, np.ones_like(self.value)], [other.value, np.ones_like(other.value)]]
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
        child_grad = [[self.value, other.value], [other.value, self.value]]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        out.node.child = [self, other]
    
        return out
    
    def __pow__(self, other):
        if not isinstance(other, float):
            other = float(other)
        out = tensor(self.value**other)
        gradient = other * (self.value**(other-1))
        child_grad = [[self.value, gradient], [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)]]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        out.node.child = [self]
        return out
    
    def sin(self):
        out = tensor(np.sin(self.value))
        gradient = np.cos(self.value)
        child_grad = [[self.value, gradient], [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)]]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        out.node.child = [self]
      
        return out
    
    def tan(self):
        out = tensor(np.tan(self.value))
        gradient = 1 / (np.cos(self.value) ** 2)
        child_grad = [[self.value, gradient], [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)]]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        out.node.child = [self]
        return out
    
    def cos(self):
        out = tensor(np.cos(self.value))
        gradient = -1 * np.sin(self.value)
        child_grad = [[self.value, gradient], [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)]]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        out.node.child = [self]
        return out
    
    def tanh(self):
        out = tensor(np.tanh(self.value))
        gradient = 1 - (np.tanh(self.value)) ** 2
        child_grad = [[self.value, gradient], [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)]]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        out.node.child = [self]
        
        return out
    
    def relu(self):
        out = tensor(np.where(self.value > 0, self.value, 0))
        gradient = np.where(self.value > 0, 1, 0)
        child_grad = [[self.value, gradient], [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)]]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        out.node.child = [self]
        return out
    
    def sigmoid(self):
        s = 1 / (1 + np.e ** (-self.value))
        out = tensor(s)
        gradient = s * (1 - s)
        child_grad = [[self.value, gradient], [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)]]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        out.node.child = [self]
        
        return out
    
    def log(self):
        s = np.log(self.value)
        out = tensor(s)
        gradient = 1 / self.value
        child_grad = [[self.value, gradient], [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)]]
        out.node.child_grad = np.array(child_grad, dtype=object)
        out.node.out = out.value
        out.node.child = [self]
        
        return out
    
    def softmax(self):
        exps = np.exp(self.value - np.max(self.value))
        s = exps / np.sum(exps)
        out = tensor(s)
        
        jacobian = np.diagflat(s) - np.outer(s, s)
        out.isoftmax=True
        child_grad=np.empty((1,2),dtype=object)
        child_grad[0,0]=self.value
        child_grad[0,1]=jacobian
        out.node.child_grad = child_grad
        out.node.out = out.value
        out.node.child = [self]
        return out

    
    def matmul(self, other):
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

    @property
    def shape(self):
        return self.value.shape
    
    def T(self):
        transposed = self.value.T
        out = tensor(transposed, dtype = self.dtype,is_leaf=self.is_leaf)
        self.node=out.node
        self.matm = out.matm
 
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
    
    @classmethod
    def zeros(cls, shape, dtype="float32"):
        """Create a tensor of zeros with the given shape."""
        return cls(np.zeros(shape), dtype=dtype, is_leaf=True)
    
    @classmethod
    def ones(cls, shape, dtype="float32"):
        """Create a tensor of ones with the given shape."""
        return cls(np.ones(shape), dtype=dtype, is_leaf=True)
    
    @classmethod
    def random(cls, shape, dtype="float32"):
        """Create a tensor with random values between 0 and 1."""
        return cls(np.random.random(shape), dtype=dtype, is_leaf=True)
    
    @classmethod
    def randn(cls, *shape, dtype="float32"):
        """Create a tensor with random values from standard normal distribution."""
        return cls(np.random.randn(*shape), dtype=dtype, is_leaf=True)
    
    @classmethod
    def eye(cls, n, dtype="float32"):
        """Create an identity matrix of size n."""
        return cls(np.eye(n), dtype=dtype, is_leaf=True)
    
    @classmethod
    def arange(cls, start, stop=None, step=1, dtype="float32"):
        """Create a tensor with evenly spaced values within a given interval."""
        if stop is None:
            stop = start
            start = 0
        return cls(np.arange(start, stop, step), dtype=dtype, is_leaf=True)
    ##prabhu insabko dekh lijiyega
    # def reshape(self, *shape):
    #     """Reshape the tensor to the given shape."""
    #     out = tensor(self.value.reshape(*shape))
    #     out.node.child = [self]
    #     # Identity gradient for reshape
    #     gradient = np.ones(self.value.shape)
    #     child_grad = [[self.value, gradient], [np.full(self.value.shape, np.nan), np.full(gradient.shape, np.nan)]]
    #     out.node.child_grad = np.array(child_grad)
    #     out.node.out = out.value
    #     return out
    # def reshape4self(self, *shape):
    #     """Reshape the tensor to the given shape."""
    #     self.value=self.value.reshape(*shape)

    
    def sum(self):
        s = np.sum(self.value)
        out = tensor(s)
        
        gradient = np.ones_like(self.value)
        
        child_grad = [[self.value, gradient],
                    [np.full_like(self.value, np.nan), np.full_like(gradient, np.nan)]]
        out.node.child_grad = np.array(child_grad, dtype=object)
        
        out.node.out = out.value
        out.node.child = [self]
        
        return out
    
    def mean(self, axis=None, keepdims=False):
        """Mean of tensor elements over a given axis."""
        out = tensor(np.mean(self.value, axis=axis, keepdims=keepdims))
        out.node.child = [self]
        # Gradient for mean is 1/N for each element
        if axis is None:
            n = self.value.size
        else:
            n = self.value.shape[axis]
        gradient = np.ones_like(self.value) / n
        child_grad = [[self.value, gradient], [np.full(self.value.shape, np.nan), np.full(gradient.shape, np.nan)]]
        out.node.child_grad = np.array(child_grad)
        out.node.out = out.value
        
        return out
    
    def max(self, axis=None, keepdims=False):
        """Maximum of tensor elements over a given axis."""
        out_value = np.max(self.value, axis=axis, keepdims=keepdims)
        out = tensor(out_value)
        out.node.child = [self]
        # Gradient for max is 1 at the max positions, 0 elsewhere
        if axis is None:
            gradient = np.where(self.value == np.max(self.value), 1.0, 0.0)
        else:
            # This is a simplification - proper backprop for max requires more care
            gradient = np.zeros_like(self.value)
            max_indices = np.argmax(self.value, axis=axis)
            # Set gradient at max positions (this is approximate)
            gradient.flat[max_indices] = 1.0
        child_grad = [[self.value, gradient], [np.full(self.value.shape, np.nan), np.full(gradient.shape, np.nan)]]
        out.node.child_grad = np.array(child_grad)
        out.node.out = out.value
        return out
    
    def min(self, axis=None, keepdims=False):
        """Minimum of tensor elements over a given axis."""
        out_value = np.min(self.value, axis=axis, keepdims=keepdims)
        out = tensor(out_value)
        out.node.child = [self]
        # Gradient for min is 1 at the min positions, 0 elsewhere
        if axis is None:
            gradient = np.where(self.value == np.min(self.value), 1.0, 0.0)
        else:
            # This is a simplification - proper backprop for min requires more care
            gradient = np.zeros_like(self.value)
            min_indices = np.argmin(self.value, axis=axis)
            # Set gradient at min positions (this is approximate)
            gradient.flat[min_indices] = 1.0
        child_grad = [[self.value, gradient], [np.full(self.value.shape, np.nan), np.full(gradient.shape, np.nan)]]
        out.node.child_grad = np.array(child_grad)
        out.node.out = out.value
        return out
    
    def exp(self):
        """Element-wise exponential."""
        out = tensor(np.exp(self.value))
        out.node.child = [self]
        # Gradient for exp is exp(x)
        gradient = np.exp(self.value)
        child_grad = [[self.value, gradient], [np.full(self.value.shape, np.nan), np.full(gradient.shape, np.nan)]]
        out.node.child_grad = np.array(child_grad)
        out.node.out = out.value
        return out
    
    def abs(self):
        """Element-wise absolute value."""
        out = tensor(np.abs(self.value))
        out.node.child = [self]
        # Gradient for abs is sign(x)
        gradient = np.sign(self.value)
        child_grad = [[self.value, gradient], [np.full(self.value.shape, np.nan), np.full(gradient.shape, np.nan)]]
        out.node.child_grad = np.array(child_grad)
        out.node.out = out.value
        return out
    
    def sqrt(self):
        """Element-wise square root."""
        out = tensor(np.sqrt(self.value))
        out.node.child = [self]
        # Gradient for sqrt is 0.5/sqrt(x)
        gradient = 0.5 / np.sqrt(self.value)
        child_grad = [[self.value, gradient], [np.full(self.value.shape, np.nan), np.full(gradient.shape, np.nan)]]
        out.node.child_grad = np.array(child_grad)
        out.node.out = out.value
        return out
    
    
    
    def squeeze(self, axis=None):
        """Remove single-dimensional entries from the shape of the tensor."""
        out = tensor(np.squeeze(self.value, axis=axis))
        out.node.child = [self]
        # Identity gradient for squeeze
        gradient = np.ones_like(self.value)
        child_grad = [[self.value, gradient], [np.full(self.value.shape, np.nan), np.full(gradient.shape, np.nan)]]
        out.node.child_grad = np.array(child_grad)
        out.node.out = out.value
        return out
    
    def unsqueeze(self, axis):
        """Insert a new axis that will appear at the axis position in the expanded tensor shape."""
        out = tensor(np.expand_dims(self.value, axis=axis))
        out.node.child = [self]
        # Identity gradient for unsqueeze
        gradient = np.ones_like(self.value)
        child_grad = [[self.value, gradient], [np.full(self.value.shape, np.nan), np.full(gradient.shape, np.nan)]]
        out.node.child_grad = np.array(child_grad)
        out.node.out = out.value
        return out
    
    def flatten(self):
        original_shape = self.value.shape
        out = tensor(self.value.reshape(-1,1))
        out.flctx = original_shape  # Save shape for backward
        out.node.child=[self]
        return out
    
    def clip(self, min_val, max_val):
        """Element-wise clip of tensor values."""
        out = tensor(np.clip(self.value, min_val, max_val))
        out.node.child = [self]
        # Gradient for clip is 1 where not clipped, 0 where clipped
        gradient = np.ones_like(self.value)
        gradient[self.value < min_val] = 0
        gradient[self.value > max_val] = 0
        child_grad = [[self.value, gradient], [np.full(self.value.shape, np.nan), np.full(gradient.shape, np.nan)]]
        out.node.child_grad = np.array(child_grad)
        out.node.out = out.value
        return out
    
    def __getitem__(self, key):
        """Support for tensor indexing."""
        out = tensor(self.value[key])
        out.node.child = [self]
        # Gradient for indexing is complex; this is a simplification
        gradient = np.zeros_like(self.value)
        # Set gradients at the indexed positions
        sliced_positions = np.zeros_like(self.value, dtype=bool)
        sliced_positions[key] = True
        gradient[sliced_positions] = 1.0
        child_grad = [[self.value, gradient], [np.full(self.value.shape, np.nan), np.full(gradient.shape, np.nan)]]
        out.node.child_grad = np.array(child_grad)
        out.node.out = out.value
        return out
    
    def __repr__(self):
        """String representation of the tensor."""
        return f'''Tensor\n({self.value},\nshape={self.value.shape})'''
    
    # def __str__(self):
    #     """String representation of the tensor."""
    #     return str(self.value)
    
    def detach(self):
        """Create a new tensor detached from the computation graph."""
        return tensor(self.value, dtype=self.dtype, is_leaf=True)
    
    def to_numpy(self):
        """Convert tensor to numpy array."""
        return self.value.copy()
    
    def item(self):
        """Return the tensor as a standard Python number."""
        if self.value.size == 1:
            return self.value.item()
        else:
            raise ValueError("Can only convert tensors with a single element to Python scalars")
    
    @classmethod
    def linspace(cls, start, stop, num=50, dtype="float32"):
        """Create a tensor with evenly spaced values within a given interval."""
        return cls(np.linspace(start, stop, num), dtype=dtype, is_leaf=True)
    @staticmethod
    @njit
    def im2col(X, filter_height, filter_width, stride=1, pad=0):
        """
        X: input of shape (C, H, W)
        Returns:
            cols: 2D array of shape (C * filter_height * filter_width, number_of_patches)
        """
        C, H, W = X.shape
        H_out = (H + 2 * pad - filter_height) // stride + 1
        W_out = (W + 2 * pad - filter_width) // stride + 1

        # Manual padding (np.pad not supported in Numba)
        X_padded = np.zeros((C, H + 2 * pad, W + 2 * pad), dtype=X.dtype)
        X_padded[:, pad:pad + H, pad:pad + W] = X

        # Output column buffer
        cols = np.empty((C * filter_height * filter_width, H_out * W_out), dtype=X.dtype)
        col_idx = 0

        for y in range(H_out):
            for x in range(W_out):
                patch = X_padded[:, y*stride : y*stride+filter_height,
                                    x*stride : x*stride+filter_width]
                cols[:, col_idx] = np.ascontiguousarray(patch).reshape(-1)
                col_idx += 1

        return cols
    
    
    
    
    # @staticmethod
    # def im2col(img, kernel_h, kernel_w, stride=1, padding=0):
    #     C, H, W = img.shape
    #     out_h = (H + 2 * padding - kernel_h) // stride + 1
    #     out_w = (W + 2 * padding - kernel_w) // stride + 1

    #     # Padding the input
    #     img_padded = np.pad(img, ((0, 0), (padding, padding), (padding, padding)), mode='constant')

    #     col = np.zeros((C, kernel_h, kernel_w, out_h, out_w))

    #     for y in range(kernel_h):
    #         y_max = y + stride * out_h
    #         for x in range(kernel_w):
    #             x_max = x + stride * out_w
    #             col[:, y, x, :, :] = img_padded[:, y:y_max:stride, x:x_max:stride]

    #     col = col.transpose(3, 4, 0, 1, 2).reshape(out_h * out_w, -1)
    #     return col
    
    def conv2d(self, W, stride=1, padding=0):

        x=self.value
        C, H, W_in = x.shape
        F, _, KH, KW = W.value.shape

        out_h = (H + 2 * padding - KH) // stride + 1
        out_w = (W_in + 2 * padding - KW) // stride + 1

        col = self.im2col(x, KH, KW, stride, padding).T  # shape: (out_h*out_w, C*KH*KW)
        W_col = W.value.reshape(F, -1).T                        # shape: (C*KH*KW, F)
        # print(col.shape,W_col.shape)
        out = col@ W_col                                 # (out_h*out_w, F)                                   # broadcast bias

        out = tensor(out.reshape(out_h, out_w, F).transpose(2, 0, 1))
        out.iconv2d=(True,stride,padding)

        # out.node.child_grad =child_grad_obj
        out.node.out = out.value
        out.node.child = [self, W]
        return out
    

    def maxpool2d(self,kernelsize, stride=1, padding=0):
        """
        img: (C, H, W) input image
        Returns:
            out: (C, out_h, out_w) pooled output
            mask: indices of max elements for each pooling window (for backprop)
        """
        if not isinstance(kernelsize,tuple):
            raise ValueError("The size should in tuple form (hieght,weight)")
        img=self.value
        kernel_h=kernelsize[0]
        kernel_w=kernelsize[1]
        
        C, H, W = img.shape
        col = self.im2col(img, kernel_h, kernel_w, stride, padding)  # (out_h*out_w, C * kH * kW)
        
        # Reshape to (out_h*out_w, C, kH*kW)
        col = col.reshape(-1, C, kernel_h * kernel_w)

        # Max along the last axis (within each pooling window)
        out = np.max(col, axis=2)  # shape: (out_h * out_w, C)
        mask = np.argmax(col, axis=2)  # shape: (out_h * out_w, C)

        # Reshape to (C, out_h, out_w)
        out_h = (H + 2 * padding - kernel_h) // stride + 1
        out_w = (W + 2 * padding - kernel_w) // stride + 1
        out = tensor(out.transpose(1, 0).reshape(C, out_h, out_w))
        mask = mask.transpose(1, 0).reshape(C, out_h, out_w)
        out.mpctx=(True,mask,self.value.shape,kernelsize,stride,padding)
        out.node.out = out.value
        out.node.child = [self]
        return out
    
    def concatenete(self,other):
        if len(self.value.shape) >3 or len(other.value.shape) >3 :
            raise ValueError("the inputs should be of 3 dimensions (channels X hieght X width)")
        out=tensor(np.concatenate([self.value,other.value]))
        out.node.child=[self,other]
        out.node.out=out.value
        out.iconcatenete=self.value.shape[0] 
        return out
    def UpSample2Dnearest(self, size):
        """
        Forward pass for nearest neighbor upsampling.

        Args:
            x (ndarray): Input array of shape (batch_size, channels, height, width)
            scale_factor (int): Factor by which to upsample the input

        Returns:
            ndarray: Upsampled array of shape (batch_size, channels, height*scale_factor, width*scale_factor)
        """
        x=self.value
        x_upsampled = np.repeat(np.repeat(x, size[1], axis=1), size[0], axis=2)
        out=tensor(x_upsampled)
        out.node.child=[self]
        out.node.out=out.value
        out.upctx=(True,self.value.shape,size)
        return out
    
    
    
    
        