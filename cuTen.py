import numpy as np
import seera_cuda
# cuten is a library, for all the GPU operations, just like numpy, infact any value it takes in is directly transferrd
# to GPU, it supports all the operations as listed in seera_engine_cuda. but ONLY operates.
# the tensor class will take in cuten tensors just like it took numpy tensors. 
# so hence depending upon the type this will significatly ease the the architecture
class cuten:
    def __init__(self, data, dtype="float32"):
        if isinstance(data, np.ndarray):
            # Transfer numpy array to GPU immediately
            arr = data.astype(dtype)
            self.shape = arr.shape
            self.dtype = dtype

        elif isinstance(data, (list, tuple)):
            # Convert to numpy first, then to GPU
            arr = np.array(data, dtype=dtype)
            self.shape = arr.shape
            self.dtype = dtype

        elif isinstance(data, ):
        
        