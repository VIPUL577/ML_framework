import numpy as np
from numba import njit,jit
from Seera_init import tensor as Tensor

#######AUTOGRAD 4 Neural Network########
class autograd4nn:
    
    def __init__(self, hook):
        self.hook = hook
        self.backward()
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
    @staticmethod   
    @njit
    def col2im(cols, X_shape, KH, KW, stride=1, pad=0):
        C, H, W = X_shape
        H_out = (H + 2 * pad - KH) // stride + 1
        W_out = (W + 2 * pad - KW) // stride + 1
        cols = cols.astype(np.float32)
        X_padded = np.zeros((C, H + 2 * pad, W + 2 * pad), dtype=np.float32)
        col = 0
        for y in range(H_out):
            for x in range(W_out):
                patch = np.ascontiguousarray(cols[:, col]).reshape(C, KH, KW)
                X_padded[:, y*stride:y*stride+KH, x*stride:x*stride+KW] += patch
                col += 1

        if pad == 0:
            return X_padded
        return X_padded[:, pad:-pad, pad:-pad]

    # @njit
    def maxpool2d_unpool(self,dout, mask, input_shape, kernelsize, stride=1, padding=0):
        """
        dout: (C, out_h, out_w) - gradient from next layer
        mask: (C, out_h, out_w) - indices of max values during forward
        input_shape: (C, H, W) - original input shape before pooling
        """
        kernel_h, kernel_w=kernelsize[0],kernelsize[1]
        
        C, H, W = input_shape
        out_h, out_w = dout.shape[1], dout.shape[2]

        # Number of output patches
        num_patches = out_h * out_w

        # We will build (C * kernel_h * kernel_w, num_patches) like in im2col
        cols = np.zeros((C * kernel_h * kernel_w, num_patches))#, dtype=np.float32

        # Flatten for easier indexing
        dout_flat = dout.reshape(C, -1)
        mask_flat = mask.reshape(C, -1)

        for c in range(C):
            # Each column gets the dout at the position of max index
            cols[c * kernel_h * kernel_w + mask_flat[c], np.arange(num_patches)] = dout_flat[c]

        # Use col2im to map back to original input shape
        # print(cols)
        dinput = self.col2im(cols.astype(np.float32), input_shape, kernel_h, kernel_w, stride, padding)

        return dinput


    # @njit
    def conv_backward(self,dO, X, W, stride=1, padding=0):
        """
        dO: gradient of the output (out_channels, Ho, Wo)
        X: input tensor (in_channels, Hi, Wi)
        W: weight tensor (out_channels, in_channels, KH, KW)
        stride: convolution stride
        padding: padding size (assumes equal padding on all sides)
        """
        out_channels, in_channels, KH, KW = W.shape

        if padding > 0:
            X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding)), 
                            mode='constant')
        else:
            X_padded = X
        # .astype(np.float32)
        # Convert to column matrices with padded input
        X_col = self.im2col(X_padded.astype(np.float32), KH, KW, stride)  # (in*KH*KW, Ho*Wo)
        dO_col = dO.reshape(out_channels, -1)  # (out, Ho*Wo)
        # print(X_col.shape)
        # Filter gradients (single matrix multiply)
        dW = dO_col @ X_col.T  # (out, in*KH*KW)
        dW = dW.reshape(W.shape)
        
        # Input gradients
        W_flat = W.reshape(out_channels, -1)  # (out, in*KH*KW)
        dX_col = W_flat.T @ dO_col  # (in*KH*KW, Ho*Wo)
        
        # Convert back to input shape, handling padding in col2im
        dX = self.col2im(dX_col.astype(np.float32), X.shape, KH, KW, stride, padding)
        
        return dX, dW
    @staticmethod
    @njit
    def upsample_backward(dout, input_shape, size):
        """
        Backward pass for nearest neighbor upsampling.

        Args:
            dout (ndarray): Gradient of the loss with respect to the output of upsampling,
                            of shape (batch_size, channels, height*size[1], width*size[0])
            input_shape (tuple): Shape of the original input (batch_size, channels, height, width)
            size (tuple): The upsampling factor as (scale_w, scale_h)

        Returns:
            dx (ndarray): Gradient with respect to the input, of shape (batch_size, channels, height, width)
        """
        scale_w, scale_h = size
        channels, height, width = input_shape
        dx = np.zeros(input_shape)

        for i in range(height):
            for j in range(width):
                dx[ :, i, j] = np.sum(np.sum(dout[ :, i*scale_h:(i+1)*scale_h, j*scale_w:(j+1)*scale_w], axis=2), axis=1)
        return dx





    def backward_step(self, nodeg):
        if nodeg.mpctx[0]:
            # print("hello")/
            cp=nodeg.node.cp.astype(np.float32)
            inp=nodeg.mpctx[1]#.astype(np.float32)
            nodeg.node.child[0].node.cp+=self.maxpool2d_unpool(cp,inp,nodeg.mpctx[2],nodeg.mpctx[3],nodeg.mpctx[4],nodeg.mpctx[5])
            # print(f"unpool parent {nodeg},child {nodeg.node.child[0]}")
        elif nodeg.upctx[0]:
            cp=nodeg.node.cp.astype(np.float32)
            nodeg.node.child[0].node.cp += self.upsample_backward(cp,nodeg.upctx[1],nodeg.upctx[2])
            
            
        elif nodeg.flctx:
            nodeg.node.child[0].node.cp += (nodeg.node.cp.reshape(nodeg.flctx))
            # print(f"flatten parent {nodeg},child {nodeg.node.child}")
            
            # print(nodeg.node.child[0].node.cp)
            
        elif nodeg.iconv2d[0]:
            X = nodeg.node.child[0].value  # (C, H, W)
            W = nodeg.node.child[1].value  # (F, C, KH, KW)
            cp = nodeg.node.cp.astype(np.float32)             # (F, out_H, out_W)
           
            nodeg.node.child[0].node.cp,nodeg.node.child[1].node.cp=self.conv_backward(cp,X.astype(np.float32),W.astype(np.float32),stride=nodeg.iconv2d[1],padding=nodeg.iconv2d[2])

        elif nodeg.isoftmax:
            jacobian = nodeg.node.child_grad[0, 1]
            nodeg.node.child[0].node.cp += jacobian @ nodeg.node.cp
            
        elif nodeg.iconcatenete:
            nodeg.node.child[0].node.cp +=nodeg.node.cp[:nodeg.iconcatenete]
            nodeg.node.child[1].node.cp +=nodeg.node.cp[nodeg.iconcatenete:]
            
        elif not nodeg.matm:
            for child_idx, child in enumerate(nodeg.node.child):
                if child_idx == 0:  # First child
                    # nodeg.node.child_grad[0,1] *= nodeg.node.cp
                    child.node.cp += (nodeg.node.child_grad[0,1]*nodeg.node.cp)
                elif child_idx == 1:  
                    # nodeg.node.child_grad[1,1] *= nodeg.node.cp
                    child.node.cp += (nodeg.node.child_grad[1,1]*nodeg.node.cp)
        
        else:
            A = nodeg.node.child[0]  
            B = nodeg.node.child[1]  
            A.node.cp += nodeg.node.cp @ B.value.T
            B.node.cp += A.value.T @ nodeg.node.cp
            
            

    def backward(self):
        self.hook.node.cp = np.ones_like(self.hook.value)
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
#########################################################################################
#######NORMAL AUTOGRAD########
class autograd:
    def __init__(self,hook):
        self.hook=hook
        self.backward()
        
        
    def backward_step(self, nodeg):
        for a in nodeg.node.child:
            
            if np.allclose(nodeg.node.child_grad[0,0], a.node.out) :
                nodeg.node.child_grad[0,1] *= nodeg.node.cp
                a.node.cp += nodeg.node.child_grad[0,1]
            elif np.allclose(nodeg.node.child_grad[1,0], a.node.out) :
                nodeg.node.child_grad[1,1] *= nodeg.node.cp
                a.node.cp += nodeg.node.child_grad[1,1]
                
    def backward(self):
        self.hook.node.cp = np.ones(self.hook.value.shape)
        graph = self.buildgraph()
        for nodeg in (graph):
            self.backward_step(nodeg)
            
    # Depth first search
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