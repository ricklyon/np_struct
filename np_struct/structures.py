import copy
import numpy as np
from . fields import npfield
from copy import deepcopy as dcopy
from collections import OrderedDict as od

_PROTECTED_FIELD_NAMES = ["value", "dtype", "shape", "unpack", "byte_order", "get_size"]

_BYTE_ORDER_TOKENS = ("=", "<", ">", "|")

_SUPPORTED_NP_TYPES = (np.uint8, np.uint16, np.uint32, np.int16, np.int32, np.int64, np.float32, np.float64, np.complex128)

class StructMeta(type):

    def __new__(metacls, cls, bases, classdict):
        
        ## ignore the Packet and Struct classes themselves, we only want the metaclass to apply to subclasses of these
        if cls in ['Packet', 'Struct']:
            return super().__new__(metacls, cls, bases, classdict)

        # all valid numpy types found in the class declaration go here
        cls_defs = {}   

        # initialize the bitfield base pointer and the position
        bit_fields = {}
        cur_bit_pos = 0
        cur_bit_dtype = None
        cur_bit_base = None
        ## walk through class definitions finding all supported numpy types, build bit fields, and attach enums
        for key, item in classdict.items():

            ## ignore any class definitions that aren't supported numpy types
            if not isinstance(item, (np.ndarray, Struct, npfield) + _SUPPORTED_NP_TYPES):
                continue

            ## error if any private variables are used in class definition, or if there is a naming collision
            if hasattr(np.ndarray, key) or hasattr(Struct, key):
                raise RuntimeError('Protected field name: ({})'.format(key))
            
            item = type(item)([item]) if not hasattr(item, '__len__') else item
            
            # handle bit fields. the attribute 'bits' of items is an integer that determines how wide the item is in 
            # the bitfield. the item position in the bitfield is determined by it's order in the class 
            # definition. bit fields are defined the same as c++ with incrementing bit significance-- the MSB is last 
            # in the bitfield definition.
            if getattr(item, 'bits', None) is not None:

                # reset bitfield counters if dtype does not match the base or we are not currently in a bitfield
                if cur_bit_pos == 0 or item.dtype != cur_bit_dtype:
                    # create bitfield base for the next bit field members and reset position counter
                    cur_bit_dtype = item.dtype
                    cur_bit_pos = 0
                    cur_bit_base = key+'_base'
                    cls_defs[cur_bit_base] = item

                bit_fields[key] = (cur_bit_base, cur_bit_pos, item.bits)
                # increment the bit position
                cur_bit_pos += item.bits

            else:
                # reset the bit counter and base value
                cur_bit_dtype = None
                cur_bit_pos = 0

                ## add each item to the cls definition dictionary
                cls_defs[key] = item

        if len(cls_defs) < 1:
            raise ValueError('Empty structures not supported. Ensure members are supported types.')

        # set the maximum string length of the items in the class. Used for printing
        classdict['_printwidth'] = max(len(k) for k in cls_defs.keys()) + 3

        classdict['_item_cls'] = {k: v.__class__ for k,v in cls_defs.items()}

        # pass items found in class definition to constructor so it can add all fields as instance members
        classdict['_cls_defs'] = cls_defs

        classdict['_bit_fields'] = bit_fields

        # remove all items from the class so they won't appear as members
        [classdict.pop(key) for key, value in cls_defs.items() if key in classdict.keys()]

        return super().__new__(metacls, cls, bases, classdict)


class Struct(np.ndarray, metaclass=StructMeta):

    def __new__(cls, input_=None, shape=None, byte_order='<', **kwargs):

        dtype = od()    

        if input_ is not None:
            shape = input_.shape if shape is None else shape
            dtype = input_.dtype
            dtype.newbyteorder(byte_order)
            obj = np.zeros(shape, dtype=dtype).view(cls)
            obj[:] = input_
            return obj
        
        for key, item in cls._cls_defs.items():
            
            if key in kwargs.keys():
                kwval = kwargs[key]
                shape_k = (1,) if not hasattr(kwval, '__len__') else np.array(kwval).shape
            else:
                shape_k = item.shape
 
            dtype[key] = (key, item.dtype, shape_k)

        # update dtype and set structure items dictionary as instance member
        dtype = np.dtype([d for d in dtype.values()])
        dtype.newbyteorder(byte_order)

        shape = (1,) if shape is None else shape
        obj = np.zeros(shape, dtype=dtype).view(cls)
        
        for key, item in cls._cls_defs.items():

            if key in  kwargs.keys():
                obj[key] = kwargs[key]
            else:
                obj[key] = item 
    
        return obj
    
    def __setitem__(self, key, value):
        if isinstance(key, str) and key in self._bit_fields.keys():
            base, pos, bits = self._bit_fields[key]
            mask = 2**(bits) - 1
            # invert the mask in order to clear the current value
            fullmask = 2**(getattr(self, base).itemsize *8) - 1
            inv_mask = fullmask ^ (mask << pos)

            self[base] &= inv_mask
            self[base] |= ((value & mask) << pos)

        else:            
            super().__setitem__(key, value)

    def __getitem__(self, key):


        if isinstance(key, (int, tuple, slice)) and self.shape == (1,):
            return super().__getitem__(key)

        if isinstance(key, str) and key in self._bit_fields.keys():
            base, pos, bits = self._bit_fields[key]
            mask = 2**(bits) - 1
            base_value = self[base] & (mask << pos)
            return (base_value >> pos)

        elif isinstance(key, str) and key in self._item_cls.keys():
            if self.shape == (1,):
                return super().__getitem__(0)[key].view(self._item_cls[key])
            else:
                return super().__getitem__(key).view(self._item_cls[key])


        ret = super().__getitem__(key)
        
        if isinstance(ret, np.void):
            return ret[None].view(self.__class__)
        else:
            return ret


    def unpack(self, bytes):
        """ 
        Unpacks byte data into the structured array for this object. 
        """
        self[:] = np.frombuffer(bytes, dtype=self.dtype)

    def get_size(self):
        return self.itemsize * self.size

    def __getattribute__(self, key):

        if key in ['_item_cls', '_bit_fields']:
            return super().__getattribute__(key)

        elif key in self._item_cls.keys() or key in self._bit_fields.keys():
            return self[key]

        else:
            return super().__getattribute__(key)

    def __setattr__(self, key, value):
        if key in self._item_cls.keys() or key in self._bit_fields.keys():
            self[key] = value
        else:
            raise ValueError('structure ({}) has no attribute: {}'.format(self.__class__.__name__, key))


    def __repr__(self):
        return str(self)
    
    def __str__(self, tabs=''):
        base_name = self.__class__.__bases__[0].__name__

        shape_str = self.shape if self.shape != (1,) else ''
        build = '{} {}: {}\n'.format(base_name, self.__class__.__name__, shape_str)
        tabs_item = tabs + '    '

        # print first and last items
        if self.shape != (1,):
            idx = ([0]*len(self.shape), [-1]*len(self.shape))
        else:
            idx = (tuple(),)

        for i, item_i in enumerate(idx):
            if len(idx) > 1:
                build += tabs + '[\n'
            for k, v in self._cls_defs.items():
                item = getattr(self[*tuple(item_i)], k)

                key_tab = ' '*(self._printwidth-len(str(k))-1)

                if isinstance(item, Struct):
                    tabs_struct = tabs_item + key_tab + '    '
                    field_str = key_tab + item.__str__(tabs_struct)
                    build += tabs_item + str(k)+':'+field_str+'\n'

                elif hasattr(v, 'bits') and v.bits is not None:
                        fields = [b_k for b_k, b in self._bit_fields.items() if b[0] == k]
                        for f in fields:
                            b_item = getattr(self, f)
                            value_str = str(b_item).replace('\n', '\n\t\t'+tabs_item+key_tab)

                            _, pos, bits = self._bit_fields[f]
                            bits_str = r'({}:{})'.format(bits + pos, pos)

                            key_tab = ' '*(self._printwidth-len(str(f))-1)
                            field_str = key_tab + str(b_item.dtype.name) + bits_str + value_str


                            build += tabs_item + str(f)+':'+field_str+'\n'
                else:

                    value_str = str(item)
                    value_str = value_str.replace('\n', '\n\t'+tabs_item+key_tab)

                    field_str = key_tab + str(item.dtype.name) + value_str

                    build += tabs_item + str(k) + ':'+field_str+'\n'

            if len(idx) > 1:
                build += tabs + ']\n'

            if np.prod(self.shape) > 2 and i < 1:
                build += tabs + '...\n'

        return build[:-1]