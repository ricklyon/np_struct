import copy
import numpy as np
from . fields import npfield, BitfieldBase
from copy import deepcopy as dcopy
from collections import OrderedDict as od

_PROTECTED_FIELD_NAMES = ["value", "dtype", "shape", "unpack", "byte_order", "get_byte_size"]

_BYTE_ORDER_TOKENS = ("=", "<", ">", "|")

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
            if not isinstance(item, (np.ndarray, Struct)):
                continue

            ## error if any private variables are used in class definition, or if there is a naming collision
            if hasattr(np.ndarray, key) or hasattr(Struct, key):
                raise RuntimeError('Protected field name: ({})'.format(key))
            
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

    def __new__(cls, input_=None, shape=None, **kwargs):

        dtype = od()    
        
        for key, item in cls._cls_defs.items():

            dtype[key] = (key, item.dtype, item.shape)

        # update dtype and set structure items dictionary as instance member
        dtype = np.dtype([d for d in dtype.values()])

        shape = (1,) if shape is None else shape
        obj = np.zeros(shape, dtype=dtype).view(cls)

        return obj
    
    def __setitem__(self, key, value):
        print(key, value)

        if isinstance(key, str) and key in self._bit_fields.keys():
            base, pos, bits = self._bit_fields[key]
            mask = 2**(bits) - 1
            value &= mask
            self[base] |= (value << pos)

        else:

            super().__setitem__(key, value)

        

    def __getitem__(self, key):

        print(key)

        if isinstance(key, (int, tuple, slice)) and self.shape == (1,):
            return super().__getitem__(key)

        # make work for larger shapes
        if isinstance(key, int):
            return super().__getitem__(key)[None].view(self.__class__)

        if key in self._bit_fields.keys():
            base, pos, bits = self._bit_fields[key]
            mask = 2**(bits) - 1
            base_value = self[base] & (mask << pos)
            return (base_value >> pos)

        elif key in self._item_cls.keys():
            if self.shape == (1,):
                return super().__getitem__(0)[key].view(self._item_cls[key])
            else:
                return super().__getitem__(key).view(self._item_cls[key])

        else:
            return super().__getitem__(key)



    #     print(key)
    #     # if self._setter:
    #     #     super().__getitem__(key)
    #     if key in self.items.keys():
    #         return self.items[key]
    #     elif self.shape == (1,):
    #         return super().__getitem__(0)[key]
    #     else:
    #         return super().__getitem__(key)


    # def __init__(self, shape=None, byte_order='<', **kwargs):
        
        

        # self.items = np.full(shape, self.items_values, dtype='object')


    #     # flag that tells the __setitem__ method to set members normally
    #     self._setter = True

    #     # make a copy of the class defined dtype, struct items for this instance, and bitfield members
    #     dtype = dcopy(self._cls_dtype)
    #     self._items = {k: self[k] for k in self._cls_items.keys()}#dcopy(self._cls_items)
        
    #     # add all items to object dictionary so it's accessible as a member.
    #     self.__dict__.update(**self._items)

    #     # copy each item in the class definition and add to the _items list. 
    #     for key,item in self._cls_defs.items():

    #         # get variable length if it exists
    #         item_kwarg = kwargs.pop(key, None)
    #         # check if variable shape was given for this structure member
    #         if item_kwarg is not None:
    #             # broadcast to shape. Structs or lists of structs cannot be defined with variable lengths
    #             inst_item = type(item)(item_kwarg)
    #             # overwrite the dtype and value in the structure dictionary
    #             dtype[key] = (key, item.dtype, item.shape)
    #             self._items[key] = inst_item

    #         # create copy of bitfield member since members are not included in the structure list and weren't
    #         # copied already.
    #         # the base bitfield has the same name as the lowest arg, it is included in the structure dictionary, but 
    #         # not the cls def, so it won't be a instance member.
    #         if getattr(item, 'bits', None) is not None:
    #             # set bitfield pointers to the copied items
    #             if isinstance(self._items.get(key, None), BitfieldBase):
    #                 base = self._items[key]

    #             item = dcopy(item)
    #             item.set_bitfield_base(base)
    #             # add as a instance member but not as an item since non-base bitfield members
    #             # are not included in the structure value.
    #             self.__dict__[key] = item

    #         # # set the byte order of all embedded structure objects
    #         # if isinstance(item, Struct):
    #         #     self._items[key]._set_byte_order(byte_order)

    #     # update dtype and set structure items dictionary as instance member
    #     dtype = np.dtype([d for d in dtype.values()])

    #     # this sets the byte order for all child items, except other structured arrays.
    #     self.dtype = dtype.newbyteorder(byte_order)
    #     self.byte_order = byte_order

    #     self._setter = False

    # def __deepcopy__(self, memo):
    #     cls = self.__class__
    #     obj = cls.__new__(cls)

    #     self._setter = True

    #     # this part is the same as the standard deepcopy
    #     for k, v in self.__dict__.items():
    #         setattr(obj, k, dcopy(v))

    #     # update the pointers of bitfield members
    #     items = obj._items

    #     for key, item in obj._cls_defs.items():
    #         # the bases are in the structure dictionary and not in the class definitions.
    #         # get the base from the structure and reset it's member list
    #         if isinstance(items.get(key, None), BitfieldBase):
    #             base = items[key]
    #             base._bit_members = []

    #             # update the pointers of each bitfield member
    #             if hasattr(item, 'set_bitfield_base'):
    #                 getattr(obj, key).set_bitfield_base(base)

    #         # re-link the class members with to the structure members since both were copied
    #         if key in self._items.keys():
    #             obj.__dict__[key] = obj._items[key]

    #     obj._setter = False
    #     return obj
        
    # def _set_byte_order(self, byte_order):
    #     self._setter = True
    #     self.byte_order = byte_order
    #     self.dtype = self.dtype.newbyteorder(byte_order)
    #     self._setter = False
        
    #     # update the byte order recursively for all nested struct objects
    #     for v in self._items.values():
    #         if isinstance(v, Struct):
    #             v._set_byte_order(byte_order)

    # def __contains__(self, name):
    #     return name in self.__dict__.keys()

    # # def get_value(self):

    # #     value = []
    # #     for v in self._items.values():
    # #         v_ = v.get_value() if hasattr(v, 'get_value') else v
    # #         value.append(v_)

    # #     return np.array([tuple(value)], dtype=self.dtype)

    # def set_value(self, value):
    #     # unpacks value into the member items of the structure
    #     for i, v in enumerate(self._items.values()):
    #         # struct values are wrapped in an additional layer
    #         if hasattr(v, 'set_value'):
    #             v.set_value(value[i])
    #         else:
    #             v[:] = value[i]

    #     self[:] = value

    # # def __len__(self):
    # #     return self.shape[0]

    # # def __bytes__(self):
    # #     return bytes(self.get_value()

    # def unpack(self, byte_data):
    #     """ 
    #     Unpacks byte data into the structured array for this object. 
    #     """
    #     value = np.frombuffer(byte_data, dtype=self.dtype)[0]
    #     self[:] = np.frombuffer(byte_data, dtype=self.dtype)[0]
    #     # self.set_value(value)

    # def __setattr__(self, name, value):

    #     if name == '_setter' or self._setter or (name not in self._cls_defs.keys()): 
    #         super().__setattr__(name, value)
    #         return

    #     item = getattr(self, name)
    #     if hasattr(item, 'set_value'):
    #         item.set_value(value)
    #     else:
    #         item[:] = value

    #     self[name] = item
    #     # self[:] = self.get_value()



    # def get_byte_size(self):
    #     self._setter = True
    #     self._bsize = len(bytes(self)) if not hasattr(self, '_bsize') or self._bsize == None else self._bsize
    #     self._setter = False

    #     return self._bsize

    # def __repr__(self):
    #     return str(self)
    
    # def __str__(self, tabs='', newline=False):
    #     bstr = 'x' + bytes(self).hex().upper()
    #     bstr = '' if len(bstr) > 24 else ' (' + bstr + ')'
    #     base_name = self.__class__.__bases__[0].__name__
    #     name = r'{} {}{}'.format(base_name, self.__class__.__name__, bstr)
        
    #     build = tabs+str(name) + ':\n' if newline else str(name) + ':\n'
    #     tabs = tabs + '    '

    #     for k in self._cls_defs.keys():
    #         item = getattr(self, k)

    #         key_tab = ' '*(self._printwidth-len(str(k))-1)

    #         if isinstance(item, Struct):
    #             field_str = key_tab + item.__str__(tabs+ ' '*(self._printwidth))
    #         else:
    #             ## recast data type in big endian
    #             dstr = '>' + item.dtype.str[1:]

    #             if getattr(item, 'bits', None) is not None:
    #                 value_str = r'({}:{})'.format(item.bits + item._bit_pos, item._bit_pos) + str(item)
    #                 p_item = (item & item._bit_mask) << item._bit_pos
                    
    #             else:
    #                 value_str = str(item)
    #                 p_item = item.get_value() if hasattr(item, 'get_value') else item

    #             v1 = p_item.astype(dstr)
    #             b0 = bytes(v1).hex().upper()
    #             byte_str = ' (0x{})'.format(b0) if len(b0) <= 24 else ''
    #             field_str = key_tab + str(item.dtype.name) + value_str + byte_str

    #         build += tabs + str(k)+':'+field_str+'\n'
    #     return build[:-1]