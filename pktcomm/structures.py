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
        self._defs_keys = [k for k,v in self._defs.items()]
        self._defs_list = []
        self._bfeild_list = []
        dtype = []

        ## walk through feild definitions and build dtype and bit feilds
        base = None
        for k,v in self._defs.items():

            if self._slice[k] != None:
                if np.any(base == None) or base.dtype != v.dtype:
                    base = v.copy()
                    self._defs_list.append((k, base, None))
                    dtype.append((k, v.dtype, v.shape))

                self._bfeild_list.append((k, v, self._slice[k], base))
            else:
                base = None
                self._defs_list.append((k, v, self._enum[k]))
                dtype.append((k, v.dtype, v.shape))

        self._order = order
        self.shape = ()
        self.dtype = np.dtype(dtype)
        self._value = self._build_value()
        self._bsize = len(bytes(self))
        

    def _build_value(self):

        self._build_bitfeilds()
        value = []
        for (k, v, _enum) in self._defs_list:
            value.append(self._get_feild_value(k,v))

        return np.array([tuple(value)], dtype=self.dtype)

    def _build_bitfeilds(self):
        ## set the value of each bitfeild reference to match the member variable values
        for (k, v, _slice, bref) in self._bfeild_list:
            size = len(bytes(v))
            maxstart = (size*8)-1
            maxvalue =  2**(size*8)-1

            rmask = maxvalue >> (maxstart-_slice.start)
            lmask = maxvalue << _slice.stop

            mask = lmask & rmask

            bref_val = bref.copy()

            setval = (v << _slice.stop) & mask

            bref_val = bref_val & (~mask)
            bref_val = bref_val | setval
            bref[:] = bref_val

    def _get_feild_value(self, key, item):
        ## returns the feild value for csctruct, enum or numpy type
        if isinstance(item, cstruct):
            return item.value
        elif self._enum[key] != None:
            return [self._enum[key](v).value for v in item]
        else:
            return item
            
    def _unpack_value(self, value):

        for i, (k, v, _enum) in enumerate(self._defs_list):
            if isinstance(v, cstruct):
                v.value = value[i]
            else:
                v[:] = value[i]

        self._unpack_bitfeilds()
        return value

    def _unpack_bitfeilds(self):

        for (k, v, _slice, bref) in self._bfeild_list:
            size = len(bytes(v))
            maxstart = (size*8)-1
            maxvalue =  2**(size*8)-1

            rmask = maxvalue >> (maxstart-_slice.start)
            lmask = maxvalue << _slice.stop

            mask = lmask & rmask
            v[:] = (bref & mask ) >> _slice.stop

    def __len__(self):
        return len(self.value[0])

    def _parse_feild(self, key, value):
        if self._slice[key] != None:
            bits = self._slice[key].start - self._slice[key].stop +1
            size = (bits // 8) + 1
            maxvalue =  2**(size*8)-1
            return value & (maxvalue >> ((size*8)-bits))
        elif self._enum[key] == None:
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
            self._defs[name][:] = self._parse_feild(name, value)

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
                if self._slice[k] != None:
                    str3 = r'({}:{})'.format(self._slice[k].start, self._slice[k].stop) + str(v)
                elif self._enum[k] != None:
                    str3 = str([self._enum[k](vv) for vv in v])
                else:
                    str3 = str(v)
                
                bstr1 = '0x' + str(bytes(v).hex().upper())
                bstr1 = '' if len(bstr1) > 17 else ' (' + bstr1 + ')'
                str1 = str(v.dtype) + str3 + bstr1

            str0 += tabs + str(k)+':'+' '*(self._printwidth-len(str(k))-1)+str1+'\n'
        return str0[:-1]

    def set_order(self, order):
        for k,v in self._defs.items():
            v.set_order(order)

        super().set_order(order)