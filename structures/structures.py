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
        ## dictionary of dtypes of each item found in the class
        dtype = od()      
        ## dictionary of all structure members
        items = od()

        # initialize the bitfield base pointer and the position
        bit_base, bit_pos = None, 0

        ## walk through class definitions finding all supported numpy types, build bit fields, and attach enums
        for key, item in classdict.items():

            ## ignore any class definitions that aren't supported numpy types
            if not isinstance(item, (np.ndarray, Struct)):
                continue

            ## error if any private variables are used in class definition, or if there is a naming collision
            if key in _PROTECTED_FIELD_NAMES or key[0] == '_':
                raise RuntimeError('Protected field name: ({})'.format(key))
            
            # handle bit fields. the attribute 'bits' of items is an integer that determines how wide the item is in 
            # the bitfield. the item position in the bitfield is determined by it's order in the class 
            # definition. bit fields are defined the same as c++ with incrementing bit significance-- the MSB is last 
            # in the bitfield definition.
            if getattr(item, 'bits', None) is not None:

                # reset bitfield counters if dtype does not match the base or we are not currently in a bitfield
                if bit_base is None or item.dtype != bit_base.dtype:
                    # create bitfield base for the next bit field members and reset position counter
                    bit_base = BitfieldBase(dtype=item.dtype)
                    bit_pos = 0

                    # bit field bases are included in the structured array value, so add the dtype for this item
                    dtype[key] = (key, item.dtype, item.shape)
                    items[key] = bit_base

                # set the bit position and base
                item.set_bitfield_pos(bit_pos)
                # increment the bit position
                bit_pos = bit_pos + item.bits

            else:
                # reset the bit counter and base value
                bit_base, bit_pos = None, 0

                # add the item dtype
                dtype[key] = (key, item.dtype, item.shape)
                items[key] = item

            ## add each item to the cls definition dictionary
            cls_defs[key] = item

        # set the maximum string length of the items in the class. Used for printing
        classdict['_printwidth'] = max(len(k) for k in cls_defs.keys()) + 3

        # pass items found in class definition to constructor so it can add all fields as instance members
        classdict['_cls_defs'] = cls_defs

        # set the dtype member of the class.
        classdict['_cls_dtype'] = dtype
        classdict['_cls_items'] = items

        # remove all items from the class so they won't appear as members
        for key, value in cls_defs.items():
            classdict.pop(key)

        return super().__new__(metacls, cls, bases, classdict)


class Struct(metaclass=StructMeta):

    def __init__(self, byte_order='<', **kwargs):
        
        # flag that tells the __setitem__ method to set members normally
        self._setter = True

        # make a copy of the class defined dtype, struct items for this instance, and bitfield members
        dtype = dcopy(self._cls_dtype)
        items = dcopy(self._cls_items)
        
        # copy each item in the class definition and add to the _struct_items list. 
        for key,item in self._cls_defs.items():

            # get variable length if it exists
            shape = kwargs.pop(key, None)
            
            # check if variable shape was given for this structure member
            if shape is not None:
                # broadcast to shape. Structs or lists of structs cannot be defined with variable lengths
                item = np.broadcast_to(item, shape).copy()
                # overwrite the dtyp and value in the structure dictionary
                dtype[key] = (key, item.dtype, shape)
                items[key] = item

            # set bitfield pointers to the copied items
            if isinstance(items.get(key, None), BitfieldBase):
                base = items[key]

            # create copy of bitfield member since members are not included in the structure list and weren't
            # copied already.
            # the base bitfield has the same name as the lowest arg, it is included in the structure dictionary, but 
            # not the cls def, so it won't be a instance member.
            if getattr(item, 'bits', None) is not None:
                item = dcopy(item)
                item.set_bitfield_base(base)
            else:
                item = items[key]

            # add all items to object dictionary so it's accessible as a member.
            self.__dict__[key] = item

            # set the byte order of all embedded structure objects
            if isinstance(item, Struct):
                item._set_byte_order(byte_order)

        # set structure shape so it's compatible with other numpy types
        self.shape = (len(items),)

        # update dtype and set structure items dictionary as instance member
        dtype = np.dtype([d for d in dtype.values()])

        # this sets the byte order for all child items, except other structured arrays.
        self.dtype = dtype.newbyteorder(byte_order)
        self.byte_order = byte_order

        self._struct_items = items

        self._setter = False

    def __deepcopy__(self, memo):
        cls = self.__class__
        obj = cls.__new__(cls)

        self._setter = True

        # this part is the same as the standard deepcopy
        for k, v in self.__dict__.items():
            setattr(obj, k, dcopy(v))

        # update the pointers of bitfield members
        items = obj._struct_items

        for key, item in obj._cls_defs.items():
            # the bases are in the structure dictionary and not in the class definitions.
            # get the base from the structure and reset it's member list
            if isinstance(items.get(key, None), BitfieldBase):
                base = items[key]
                base._bit_members = []

            # update the pointers of each bitfield member
            if hasattr(item, 'set_bitfield_base'):
                getattr(obj, key).set_bitfield_base(base)

        obj._setter = False
        return obj
        
    def _set_byte_order(self, byte_order):
        self._setter = True
        self.byte_order = byte_order
        self.dtype = self.dtype.newbyteorder(byte_order)
        self._setter = False
        
        # update the byte order recursively for all nested struct objects
        for v in self._struct_items.values():
            if isinstance(v, Struct):
                v._set_byte_order(byte_order)

    def __contains__(self, name):
        return name in self.__dict__.keys()

    def get_value(self):

        value = []
        for v in self._struct_items.values():
            value.append(v.get_value())

        return np.array([tuple(value)], dtype=self.dtype)

    def set_value(self, value):
        # unpacks value into the member items of the structure

        for i, v in enumerate(self._struct_items.values()):
            v.set_value(value[i])

    def __len__(self):
        return self.shape[0]

    def __bytes__(self):
        return bytes(self.get_value())

    def unpack(self, byte_data):
        """ 
        Unpacks byte data into the structured array for this object. 
        """
        value = np.frombuffer(byte_data, dtype=self.dtype)[0]
        self.set_value(value)

    def __setattr__(self, name, value):
        if name == '_setter' or self._setter: 
            super().__setattr__(name, value)

        elif name in self._cls_defs.keys():
            getattr(self, name).set_value(value)

        else:
            raise ValueError("Invalid field name: '{}'".format(name))


    def get_byte_size(self):
        self._setter = True
        self._bsize = len(bytes(self)) if self._bsize == None else self._bsize
        self._setter = False

        return self._bsize

    def __repr__(self):
        return str(self)
    
    def __str__(self, tabs='', newline=False):
        bstr = 'x' + bytes(self).hex().upper()
        bstr = '' if len(bstr) > 24 else ' (' + bstr + ')'
        base_name = self.__class__.__bases__[0].__name__
        name = r'{} {}{}'.format(base_name, self.__class__.__name__, bstr)
        
        build = tabs+str(name) + ':\n' if newline else str(name) + ':\n'
        tabs = tabs + '    '

        for k in self._cls_defs.keys():
            item = getattr(self, k)

            key_tab = ' '*(self._printwidth-len(str(k))-1)

            if isinstance(item, Struct):
                field_str = key_tab + item.__str__(tabs+ ' '*(self._printwidth))
            else:
                ## recast data type in big endian
                dstr = '>' + item.dtype.str[1:]

                if getattr(item, 'bits', None) is not None:
                    value_str = r'({}:{})'.format(item.bits + item._bit_pos, item._bit_pos) + str(item)
                    p_item = (item & item._bit_mask) << item._bit_pos
                    
                else:
                    value_str = str(item)

                v1 = p_item.astype(dstr)
                b0 = bytes(v1).hex().upper()
                byte_str = ' (0x{})'.format(b0) if len(b0) <= 24 else ''
                field_str = key_tab + str(item.dtype.name) + value_str + byte_str

            build += tabs + str(k)+':'+field_str+'\n'
        return build[:-1]