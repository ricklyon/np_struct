import copy
import numpy as np

supported_dtypes = (
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.uint64,
    np.int64,
    np.float32,
    np.float64,
    np.string_
)

_PROTECTED_FIELD_NAMES = ["value", "dtype", "shape", "unpack", "byte_order", "get_byte_size"]

_BYTE_ORDER_TOKENS = ("=", "<", ">", "|")

class StructMeta(type):

    def __new__(metacls, cls, bases, classdict):
        
        ## ignore the Packet and Struct classes themselves, we only want the metaclass to apply to subclasses of these
        if cls == 'Packet' or cls == 'Struct':
            return super().__new__(metacls, cls, bases, classdict)

        cls_defs = {}   ## all valid numpy types found in the class declaration go here
        enums = {}      ## built-in enum classes found next to numpy types
        slices = {}     ## slices used for bit fields
        structarray = {}## flag for each field indicating whether or not it's a Struct
        dtype = []      ## dtype of each numpy type
        bf_bases = {}   ## if a bit field is found, this points to the numpy object used in the structured array

        ## walk through class definitions finding all supported numpy types, build bit fields, and attach enums
        base_key, base, bnum = None, None, 0
        for i, (key, value) in enumerate(classdict.items()):
            _enum  = None
            _bitnum = None
            _slice = None
            _sarray = False

            ## second item in field definition can be a bit field number or enum class
            if isinstance(value, tuple) and len(value) == 2:

                if isinstance(value[1], int):
                    value, _bitnum = value
                else:
                    base, bnum = None, 0
                    value, _enum = value
            else:
                base, bnum = None, 0

            ## ignore any class definitions that aren't supported numpy types
            if not isinstance(value, (np.ndarray, supported_dtypes, Struct)):
                base, bnum = None, 0
                continue
            
            ## error if any private variables are used in class definition, or if there is a naming collision
            if key in _PROTECTED_FIELD_NAMES or key[0] == '_':
                raise RuntimeError('Protected field name: ({})'.format(key))
            
            _dtype = (key, value.dtype, value.shape)
            
            ## Structs can't have enums or be part of bitfields
            if not isinstance(value, Struct):

                ## make base numpy types into single element arrays of that type
                if len(value.shape) == 0:
                    value = np.array([value], dtype=value.dtype)
                    _dtype = (key, value.dtype, value.shape)

                ## flag any arrays whose elements are Structs, these are handled differently than normal arrays. 
                ## it's convenient and fast to just have a flag we can query.
                if isinstance(value[0], Struct):
                    _sarray = True
                    _dtype = (key, value[0].value.dtype, value.shape)
                    base, bnum = None, 0

                ## keep track of what bit we are on if this definition is part of bitfield
                elif _bitnum != None:
                    _dtype = None
                    if np.any(base == None) or base.dtype != value.dtype:
                        base_key, base, bnum = key, value, 0
                        _dtype = (key, value.dtype, value.shape)
                    
                    ## use slice object to index bit field 
                    ## if the current class definition is not the base for the bitfield, it will not
                    ## be part of the structured array and will not have a dtype
                    _slice = slice((bnum + _bitnum)-1, bnum)
                    bnum += _bitnum

            if _dtype != None:
                dtype.append(_dtype)

            ## bit field bases are referenced by a key until we create a Struct object
            bf_bases[key] = base_key if base != None else None
            enums[key], slices[key]  = _enum, _slice
            cls_defs[key] = value
            structarray[key] = _sarray

        classdict['_printwidth'] = max(len(k) for k in cls_defs.keys()) + 3
        classdict['_oldcls'] = metacls
        classdict['_enum'] = enums
        classdict['_slice'] = slices
        classdict['_cls_defs'] = cls_defs
        classdict['dtype'] = np.dtype(dtype)
        classdict['_structarray'] = structarray
        classdict['_bf_bases'] = bf_bases

        for key, value in cls_defs.items():
            classdict.pop(key)

        ncls = super().__new__(metacls, cls, bases, classdict)

        return ncls

class Struct(metaclass=StructMeta):

    def __init__(self, byte_order='<', **kwargs):
    
        self._setter = True
        self._byte_order = None
        self._defs = {}
        self._defs_list = []
        self._bfield_list = []

        dtype = []
        dshapes = len(kwargs) > 0
        
        base, bnum = None, 0
        for k,v in self._cls_defs.items():

            if self._slice[k] != None:
                
                if self._bf_bases[k] == k:
                    value = v.copy()
                    base = value
                    self._defs_list.append([k, base, None])
                    if dshapes:
                        dtype.append((k, v.dtype, v.shape))

                ## copy value even if this key is a base for the bitfield.
                ## the reference 'base' that all the bitfields point to should be only accessible
                ## in the bfield_list, not as a member variable.
                value = v.copy()
                self._bfield_list.append((k, value, self._slice[k], base))

            else:
                shape = kwargs.pop(k, None)
                if np.any(shape != None):
                    value = np.broadcast_to(v, shape.shape).copy()
                    value[:] = shape
                    #dtype.append((k, value.dtype, value.shape))
                elif self._structarray[k]:
                    value = copy.deepcopy(v)
                else:
                    value = v.__copy__()

                self._defs_list.append([k, value, self._enum[k]])
                if dshapes:
                    dtype.append((k, value.dtype, value.shape))

            self._defs[k] = value
            self.__dict__[k] = value

        self.shape = ()
        self._bsize = None
        if dshapes:   
            self.dtype = np.dtype(dtype)

        self._set_order(byte_order)
        self._setter = False
        
    def _set_order(self, byte_order):
        self._setter = True
        self._byte_order = byte_order
        self.dtype = self.dtype.newbyteorder(byte_order)
        self._setter = False

        for (k, v, _enum) in self._defs_list:
            if isinstance(v, Struct):
                v._set_order(byte_order)

    def __contains__(self, name):
        return name in self._defs.keys()

    def _build_value(self):

        self._pack_bitfields()
        value = []
        for (k, v, _enum) in self._defs_list:
            value.append(self._get_field_value(k,v))

        return np.array([tuple(value)], dtype=self.dtype)

    def _pack_bitfields(self):
        ## set the value of each bitfield reference to match the member variable values
        for (k, v, _slice, bref) in self._bfield_list:
            setval, mask = self._mask_bitfield(v, _slice)

            bref_val = bref.copy()

            bref_val = bref_val & (~mask)
            bref_val = bref_val | setval
            bref[:] = bref_val

    def _mask_bitfield(self, value, _slice):
            size = len(bytes(value))
            maxstart = (size*8)-1
            maxvalue =  2**(size*8)-1

            rmask = maxvalue >> (maxstart-_slice.start)
            lmask = maxvalue << _slice.stop

            mask = lmask & rmask
            return (value << _slice.stop) & mask, mask

    def _unpack_bitfields(self):

        for (k, v, _slice, bref) in self._bfield_list:
            size = len(bytes(v))
            maxstart = (size*8)-1
            maxvalue =  2**(size*8)-1

            rmask = maxvalue >> (maxstart-_slice.start)
            lmask = maxvalue << _slice.stop

            mask = lmask & rmask
            v[:] = (bref & mask ) >> _slice.stop

    def _get_field_value(self, key, item):
        ## returns the field value for csctruct, enum or numpy type
        if isinstance(item, Struct):
            return item.value
        elif self._enum[key] != None:
            return [self._enum[key](v).value for v in item]
        elif self._structarray[key]:
            r = np.array([v.value for v in item], dtype=item[0].dtype)
            return r.squeeze()
        else:
            return item
            
    def _unpack_value(self, value):

        for i, (k, v, _enum) in enumerate(self._defs_list):
            if isinstance(v, Struct):
                v.value = value[i]
            elif self._structarray[k]:
                for j, item in enumerate(v):
                    item.value = value[i][j]
            else:
                v[:] = value[i]

        self._unpack_bitfields()
        return value

    def __len__(self):
        return len(self.value[0])

    def _parse_field(self, key, value):

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
        """ structured array of the Struct object. Calling this property rebuilds the structured array 
            so any changes made to the class attributes are reflected in the structured array
        """
        return self._build_value()

    @value.setter
    def value(self, value):
        self._unpack_value(value)

    def __bytes__(self):
        return bytes(self.value)

    def unpack(self, byte_data):
        """ Unpacks byte data into the structured array for this object. Basically just a wrapper around numpy.frombuffer.
        """
        self.value = np.frombuffer(byte_data, dtype=self.dtype)[0]

    def __setattr__(self, name, value):
        if name == '_setter' or name == 'value' or self._setter: 
            super().__setattr__(name, value)

        elif name in self._defs.keys():
            if isinstance(value, Struct):
                self._defs[name].value = value.value[0]
            elif self._structarray[name]:
                for j, item in enumerate(self._defs[name]):
                    item.value = value[j].value[0]
            else:
                self._defs[name][:] = self._parse_field(name, value)

        else:
            raise ValueError("{} has no field '{}'".format(self.__class__.__name__, name))


    def __copy__(self):
        dct = copy.deepcopy(self.__dict__)
        inst = self.__class__()
        inst.__dict__.update(dct)
        return inst

    def get_byte_size(self):
        self._setter = True
        self._bsize = len(bytes(self)) if self._bsize == None else self._bsize
        self._setter = False

        return self._bsize
    
    def __str__(self, tabs='', newline=False):
        bstr = 'x' + bytes(self).hex().upper()
        bstr = '' if len(bstr) > 24 else ' (' + bstr + ')'
        base_name = self.__class__.__bases__[0].__name__
        name = r'{} {}{}'.format(base_name, self.__class__.__name__, bstr)
        
        build = tabs+str(name) + ':\n' if newline else str(name) + ':\n'
        tabs = tabs + '    '
        for k,v in self._defs.items():
            key_tab = ' '*(self._printwidth-len(str(k))-1)

            if isinstance(v, Struct):
                field_str = key_tab + v.__str__(tabs+ ' '*(self._printwidth))
            elif isinstance(v[0], Struct):
                field_str = key_tab+'[ \n'
                for vv in v:
                    field_str += vv.__str__(tabs+ ' '*(self._printwidth), newline=True) +'\n'
                field_str += (tabs+' '*len(str(k))+key_tab+' ]')
            else:
                ## recast data type in order to print correct byte order
                dstr = self._byte_order + v.dtype.str[1:]
                v1 = v.astype(dstr)

                if self._slice[k] != None:
                    value_str = r'({}:{})'.format(self._slice[k].start, self._slice[k].stop) + str(v)
                    v1, mask = self._mask_bitfield(v1, self._slice[k])
                elif self._enum[k] != None:
                    value_str = str([self._enum[k](vv) for vv in v])
                else:
                    value_str = str(v)

                b0 = bytes(v1).hex().upper()
                byte_str = ' (0x{})'.format(b0) if len(b0) <= 24 else ''
                field_str = key_tab + str(v.dtype.name) + value_str + byte_str

            build += tabs + str(k)+':'+field_str+'\n'
        return build[:-1]