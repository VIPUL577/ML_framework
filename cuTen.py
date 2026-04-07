from __future__ import annotations
import numpy as np
import seera_cuda
# cuten is a library, for all the GPU operations, just like numpy, infact any value it takes in is directly transferrd
# to GPU, it supports all the operations as listed in seera_engine_cuda. but ONLY operates and not tracks which will be done by tensor package. 
# the tensor class will take in cuten tensors just like it took numpy tensors. 
# so hence depending upon the type this will significatly ease the the architecture.
# Fun
class cuten:
    def __init__(self, data, dtype="float32"):
        self.supported_types = ["float32","int32","int16"]
        self.fill_alloc_dtype = {
            "float32":seera_cuda.to_device_f32,
            "int32":seera_cuda.to_device_i32,
            "int16":seera_cuda.to_device_i16
            
        }
        if dtype not in self.supported_types :
            raise ValueError("Not Supported by cuTen, try in CPU")

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
        print(f"the dtype is {dtype}")
        self.main_ptr = self.fill_alloc_dtype[dtype](data)
        self.shape = shape
        self.size = size
        self.dtype = dtype
        
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
            if self.shape != other.shape:
                raise ValueError(f"Should be of same shape for Elements Add one is of {self.shape} and {other.shape}")
            c= cuten.ones_like(self)
            seera_cuda.cuda_elemadd(self.main_ptr,other.main_ptr, c.main_ptr, self.size)
            return c
            
        if isinstance(other,int) or isinstance(other,float):
            seera_cuda.cuda_scaler_add_f(self.main_ptr, float(other),self.size )
        
            return self 
    
    def __mul__(self, other):
    
        if isinstance(other,cuten):
            if self.shape != other.shape:
                raise ValueError(f"Should be of same shape for Elements Multiplication Add one is of {self.shape} and {other.shape}")

            c= cuten.ones_like(self)
            seera_cuda.cuda_elemmult(self.main_ptr,other.main_ptr, c.main_ptr, self.size)
            return c
            
        if isinstance(other,int) or isinstance(other,float):
            seera_cuda.cuda_scaler_multiply_f(self.main_ptr, float(other),self.size )
        
            return self 
        
        raise ValueError("Not Supported, Can be Device Mismatch")
        
        
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
            raise ValueError(f"Cannot reshape cuten of size: {self.size} to shape {shape}")
        
        self.shape = shape
        
    def __repr__(self):
        np_local = seera_cuda.to_host_f32(self.main_ptr,self.shape)
        print(np_local)
        
        return "IT DOES WORK"
    
        
    
        
                
    
        