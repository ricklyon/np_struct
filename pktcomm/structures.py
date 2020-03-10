import copy
import numpy as np

supported_dtypes = (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64, np.float32, np.float64)


class StructMeta(type):

    def __call__(cls, *args, **kwargs):
        dct = dict(cls.__dict__)
        name = cls.__name__
        bases = cls.__bases__

        ## pull out all class variables of cFields type from class dictionary
        ## these will be added to the instance dictionary
        fields = {}
        enums = {}
        slices = {}
        for i, (key, value) in enumerate(dct.items()):
            
            _enum = None
            _slice = None

            if isinstance(value, tuple) and len(value) == 2:

                if isinstance(value[1], slice):
                    value, _slice = value
                else:
                    value, _enum = value

            if isinstance(value, (np.ndarray, supported_dtypes, cstruct)):

                if key in ['value', 'dtype', 'shape'] or key[0] == '_':
                    raise RuntimeError('Protected field name: ({})'.format(key))
                
                if not isinstance(value, cstruct) and len(value.shape) == 0:
                    value = np.array([value], dtype=value.dtype)

                fields[key], enums[key], slices[key]  = value, _enum, _slice

        for key, value in fields.items():
            dct.pop(key)

        ## add class variable printwidth to speed up printing out member variables
        #prtw = max([len(k) for k,v in fields.items()]) + 3

        dct['_printwidth'] = max(len(k) for k in fields.keys()) + 3
        dct['_oldcls'] = cls
        dct['_enum'] = enums
        dct['_slice'] = slices

        ## call type's __new__ to create new class definition with updated class dictionary
        ## call __new__ of class definition to create class instance
        cls = type.__new__(StructMeta, name, bases, dct)
        self = cls.__new__(cls, bases, dct)

        for key, value in fields.items():
            self.__dict__[key] = value.__copy__()

        ## call __init__ and return initizialed object
        self.__init__(*args, **kwargs)
        return self

class cstruct(metaclass=StructMeta):
    
    def __init__(self, order='<', **kwargs):
    
        self._defs = dict(**vars(self))
        self._order = order
        
        self.shape = ()
        self.dtype = self._build_dtype()
        self._value = self._build_value()
        self._bsize = len(bytes(self))
        
    
    def _build_dtype(self):
        dtype = []
        for k,v in self._defs.items():
            dtype.append((k, v.dtype, v.shape))

        return np.dtype(dtype)

    def _build_value(self):
        value = []
        for k,v in self._defs.items():
            value.append(self._get_feild_value(k,v))

        return np.array([tuple(value)], dtype=self.dtype)

    def _build_bitfeilds(self):
        base_ = None
        for k,v in self._defs.items():
            if self._slice[key] != None:
                if base_ == None:
                    base_ = v
                elif v.__class__ == base_.__class__:
                    base_ 
                else:
                    pass

    def _get_feild_value(self, key, item):
        if isinstance(item, cstruct):
            return item.value
        elif self._enum[key] != None:
            return [self._enum[key](v).value for v in item]
        else:
            return item

    def __setitem__(self, key, value):
        self.value[key] = value
            
    def _unpack_value(self, value):

        for i, (k,v) in enumerate(self._defs.items()):
            # if isinstance(v, cstruct):
            #     v.value = value[i]
            # else:
            v[:] = value[i]

    def __len__(self):
        return len(self.value[0])

    def _parse_enum(self, key, value):
        if self._enum[key] == None:
            return value
        elif isinstance(value, self._enum[key]):
            return value.value
        else:
            return value

    @property
    def value(self):
        return self._build_value()

    @value.setter
    def value(self, value):
        self._value = self._unpack_value(value)

    def __bytes__(self):
        return bytes(self.value)

    def unpack(self, byte_data):
        self.value = np.frombuffer(byte_data, dtype=self.dtype)[0]

    def __setattr__(self, name, value):
        if name != '_defs' and (name in self._defs.keys()):
            self._defs[name][:] = self._parse_enum(name, value)

        else:
            super().__setattr__(name, value)

    def __copy__(self):

        dct = copy.deepcopy(self.__dict__)
        inst = self._oldcls()
        inst.__dict__.update(dct)
        return inst

    def get_byte_size(self):
        return self._bsize

    def byte_str(self):
        return 'x' + bytes(self).hex().upper()
    
    def __str__(self, tabs=''):
        bstr = self.byte_str()
        bstr = '' if len(bstr) > 17 else ' (' + bstr + ')'
        name = r'{} {}{}'.format(self.__class__.__bases__[0].__name__, self.__class__.__name__, bstr)
        str0 = str(name) + ':\n'
        tabs = tabs+'    '
        for k,v in self._defs.items():
            if isinstance(v, cstruct):
                str1 = v.__str__(tabs+ ' '*(self._printwidth))
            else:
                if self._enum[k] != None:
                    str3 = str([self._enum[k](vv) for vv in v])
                else:
                    str3 = str(v)
                str1 = str(v.dtype) + str3 + ' (0x' + str(bytes(v).hex().upper()) + ')'

            str0 += tabs + str(k)+':'+' '*(self._printwidth-len(str(k))-1)+str1+'\n'
        return str0[:-1]

    def set_order(self, order):
        for k,v in self._defs.items():
            v.set_order(order)

        super().set_order(order)