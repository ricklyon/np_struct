
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
        ## the LSB position that this member takes in the bitfield
        obj._bit_pos = None
        # bit mask has 1s in the bit positions where this bitfield member applies, 0s elsewhere
        obj._bit_mask = None
        ## pointer to the base of the bitfield (LSB member)
        obj._bit_base = None

        return obj

    def __array_finalize__(self, obj):
        ## required method of subclasses of numpy. Sets unique member variables of new instances
        if obj is None: return

        self.bits = getattr(obj, 'bits', None)
        self.doc = getattr(obj, 'doc', "")
        self.enum = getattr(obj, 'enum', None)
        self._bit_pos = getattr(obj, '_bit_pos', None)
        self._bit_base = getattr(obj, '_bit_base', None)
        self._bit_mask = getattr(obj, '_bit_mask', None)
        
    def __array_wrap__(self, out_arr, context=None):
        
        dtype = np.dtype(self.__class__.__name__.lower())
        
        return out_arr.astype(dtype)

    def set_value(self, value):
                        
        # Update the bitfield base object if this field belongs to a bitfield    
        if self._bit_base is not None:
            value &= self._bit_mask
            self._bit_base |= (value << self._bit_pos)

        # set with enum value
        elif self.enum is not None:
            pass
        
        # set the value of this array without changing the dtype. this will broadcast the value to match the shape of 
        # this field
        self[:] = value

    def set_value_from_base(self):
        ## pull the bitfield member value from the base object
        value = self._bit_base & (self._bit_mask << self._bit_pos)
        self[:] = (value >> self._bit_pos)

    def get_value(self):
        # return enum value
        return self

    def set_bitfield_pos(self, bit_pos):
        # initalizes the bitfield masks for this member
        self._bit_mask = 2**(self.bits) - 1
        self._bit_pos = bit_pos

    def set_bitfield_base(self, base):
        # sets base as the base pointer and appends this object to the list of bit members
        self._bit_base = base
        base.add_member(self)


class BitfieldBase(np.ndarray):
    """
    Holds the full value of a bitfield and keeps track of bitfield members.
    """
    def __new__(cls, dtype):
        
        obj = np.asarray([0], dtype=dtype).view(cls)
        
        ## bitfield members whose _bit_base pointers point to this object
        obj._bit_members = []

        return obj

    def __array_finalize__(self, obj):
        ## required method of subclasses of numpy. Sets unique member variables of new instances
        if obj is None: return
        # reset new bit-members list when the object is modified or copied
        self._bit_members = []

    def set_value(self, value):
        # set the value of all the bitfield members
        self[:] = value

        for b in self._bit_members:
            b.set_value_from_base()

    def get_value(self):
        return self

    def add_member(self, obj):
        self._bit_members.append(obj)


class UInt8(npfield):
    pass

class Int8(npfield):
    pass

class UInt16(npfield):
    pass

class Int16(npfield):
    pass

class UInt32(npfield):
    pass

class Int32(npfield):
    pass



