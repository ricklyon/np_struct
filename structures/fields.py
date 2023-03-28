
import numpy as np
from copy import deepcopy as dcopy


class npfield(np.ndarray):

    def __new__(cls, input_=None, bits=None, doc=None, dtype=None, enum=None):
        
        input_ = 0 if np.all(input_ == None) else input_

        ## cast single values as arrays
        input_ = [input_] if not isinstance(input_, (tuple, list, np.ndarray)) else input_
        
        ## set dtype based on class name
        dtype = np.dtype(cls.__name__.lower())
        
        obj = np.asarray(input_, dtype=dtype).view(cls)
        
        ## assign member variables
        obj.bits = bits
        obj.doc = doc
        obj.enum = enum

        return obj

    def __array_finalize__(self, obj):
        ## required method of subclasses of numpy. Sets unique member variables of new instances
        if obj is None: return

        self.bits = getattr(obj, 'bits', None)
        self.doc = getattr(obj, 'doc', "")
        self.enum = getattr(obj, 'enum', None)
        
    def __array_wrap__(self, out_arr, context=None):
        
        dtype = np.dtype(self.__class__.__name__.lower())
        
        return out_arr.astype(dtype)

class uint8(npfield):
    pass

class int8(npfield):
    pass

class uint16(npfield):
    pass

class int16(npfield):
    pass

class uint32(npfield):
    pass

class int32(npfield):
    pass

class float64(npfield):
    pass

class float32(npfield):
    pass
