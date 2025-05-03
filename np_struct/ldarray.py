import numpy as np
import datetime as dt
from collections import OrderedDict
from copy import deepcopy as dcopy
import datetime

def check_shapes(a: tuple, b: tuple):
    """ 
    Check that the shape tuples a and b match
    """

    if len(a) != len(b):
        return False
    
    # check that the length of each dimension matches
    return all([a[i] == b[i] for i in range(len(a))])


def datetime_idx_handler(v, coords: np.ndarray):
    """
    Index handler for datetime coords
    """
    # convert coords to timestamp.
    coords_ts = np.array([v.timestamp() for v in coords])

    # cast date selection coords to datetime, then get timestamp
    if isinstance(v, datetime.date):
        v_ts = dt.datetime(year=v.year, month=v.month, day=v.day).timestamp()
    # get timestamp from datetime selections
    elif isinstance(v, datetime.datetime):
        v_ts = v.timestamp()
    # if str, treat as UTC formatted string, use same format as is printed with the ldarray __str__ method.
    elif isinstance(v, str):
        v_ts = datetime.datetime.strptime(v, '%Y-%m-%dT%H:%M').timestamp()

    # get index of the minimum distance to the indexing timestamp
    return np.argmin(np.abs(coords_ts - v_ts))


class Coords(OrderedDict):
    """ 
    Labeled dimension coordinates for ldarray. Conditions values to work as indices, but otherwise, same as 
    an Ordered Dictionary. 

    Accepts the "idx_precision" kwarg that will not be included in dictionary, 
    but can be optionally used to specify index precision for each dimension. Value of idx_precision is
    the maximum distance selections can be from the defined coordinates without an error being raised.
    Alternatively, the index precision can be set after the constructor is called with ``set_idx_precision()``
    
    Examples
    --------
    >>> coords = Coords(a=[1.2, 2.4, 3.1], b=[4,5,6], idx_precision=dict(a=1e-6))

    """

    def __init__(self, **kwargs):
        # Pop idx_precision from kwargs. Floating point indices default to 3 decimal precision.
        self.idx_precision = kwargs.pop('idx_precision', {})

        # initialize look up table for exact dimensional labels (integers)
        self.idx_label_lut = {}

        # dictionary of custom indexing handlers
        self.idx_handlers = kwargs.pop('idx_handlers', {})

        # Call OrderedDict __init__ to create dictionary of values, calls __setitem__ with each entry
        super().__init__(**kwargs)

            
    def set_precision(self, **kwargs):
        """ 
        Sets precision for coordinates. Accepts key value pairs where key is dimensional key
        and value is index precision. Precision value can be less or greater than 1, default precision is 1e-6.

        Precision is the maximum distance a index can be from a defined coordinate without an error being raised.

        Examples
        --------
        >>> coord = Coords(a=[1.2, 2.4, 3.1], b=[4,5,6])
        >>> coord.set_precision(b=1, a=1e-3)
        """

        # Update precisions only if the key already exists in idx_precision.
        # This ensures only floats have idx_precision specified.
        for k, v in kwargs.items():
            if k not in self.keys():
                raise ValueError(f"Unrecognized dimension: {k}.")
            
            self.idx_precision[k] = v

    def set_handler(self, **kwargs):
        """ 
        Sets a custom index handler for one (or multiple) dimension.

        Handlers must accept a single coordinate value, and a an array of all the coordinate values.
        Must return a standard integer index into the coordinate array.

        Examples
        --------
        >>> def ex_handler(coord, coordinates):
        ...     return np.argmin(np.abs(coordinates - coord))

        >>> coords = Coords(a=[1.2, 2.4, 3.1], b=[4,5,6])
        >>> coords.set_handler(b=ex_handler)
        """

        for k,v in kwargs.items():
            if k not in self.keys():
                raise ValueError(f"Unrecognized dimension: {k}")
            
            self.idx_handlers[k] = v

    @property
    def shape(self):
        """ 
        Shape of the ldarray that uses this coordinate map.
        """
        return tuple([len(v) for k,v in self.items()])
    
    def pop(self, key: str):
        # remove the key from the precision and handler dictionaries if it exists.
        self.idx_precision.pop(key, None)
        self.idx_handlers.pop(key, None)
        super().pop(key)
    
    def index(self, idx: tuple) -> tuple:
        """ 
        Index the coordinates with standard integer indices.
        """
        return tuple([v[idx[i]] for i, (k, v) in enumerate(self.items())])

    def __setitem__(self, k, v):
        # adds new values to the dictionary
    
        # cast as numpy array
        v = np.atleast_1d(v)

        # cast dates (only day/month/year) to more general datetime objects
        if isinstance(v[0], datetime.date):
            v = np.array([dt.datetime(year=d.year, month=d.month, day=d.day) for d in v])

        f64 = np.dtype(np.float64)
        f32 = np.dtype(np.float32)

        # Provide default values for index precision if the values are floats
        if (k not in self.idx_precision.keys()) and (v.dtype in [f64, f32]):
            if len(v) == 1:
                self.idx_precision[k] = 1e-10
            else:
                self.idx_precision[k] = np.average(np.diff(v))
        
        elif isinstance(v[0], datetime.datetime):
            # use index handler for datetime objects
            self.idx_handlers[k] = datetime_idx_handler
            
        # add to lookup table otherwise
        else:
            self.idx_label_lut[k] = {vv:i for i,vv in enumerate(v)}

        # call ordered dictionary __setitem__
        super().__setitem__(k, v)

    
    def get_axis_idx(self, key):
        """ 
        Returns the axis (dimension) index that 'key' has in the lddarray that uses this lddim.
        """
        return list(self.keys()).index(key)

    def __str__(self):
        # breaks out each key-value pair into it's own line for easier reading 
        s = '{\n'
        for k,v in self.items():
            s += k + ': ' + v.__repr__() + '\n'
        return s+ '}'


    def __repr__(self):
        return self.__str__()



class ldarray(np.ndarray):
    """ 
    Labeled numpy array. A drastically scaled down version of xarray's DataArray. Arrays behave exactly the same as 
    standard numpy arrays (no need to use .values), and supports indexing with coordinates.

    Math operations that change the coordinates or array shape (i.e. sum or transpose) silently revert the labeled array 
    to a standard numpy array without coordinates.

    Real or complex-valued arrays can be saved with the normal ``np.save()`` function, and
    loaded with ``ldarray.load()``.

    Examples
    --------

    >>> coords = dict(a=[1,2], b=['data1', 'data2', 'data3'])
    >>> ld = ldarray([[10, 11, 12],[13, 14, 15]], coords=coords, dtype=np.float64)
    >>> ld
    ldarray([[10, 11, 12],
             [13, 14, 15]])
    Coordinates: (2, 3)
      a: [1 2]
      b: ['data1' 'data2' 'data3']

    Normal indexing works the same with standard numpy arrays,

    >>> ld[:, 2]
    ldarray([12, 15])
    Coordinates: (2,)
      a: [1 2]

    Including advanced indexing with other numpy arrays,

    >>> ld[:, np.array([0, 2])]
    ldarray([[10, 12],
             [13, 15]])
    Coordinates: (2, 2)
      a: [1 2]
      b: ['data1' 'data3']

    Values can be selected or set by coordinate,

    >>> ld.sel(b="data1")
    ldarray([10, 13])
    Coordinates: (2,)
        a: [1 2]

    >>> ld[dict(a=2)] = [1, 2, 3]
    >>> ld
    ldarray([[10, 11, 12],
            [ 1,  2,  3]])
    Coordinates: (2, 3)
    a: [1 2]
    b: ['data1' 'data2' 'data3']

    Coordinate indexing can be done with slices, endpoint is inclusive,

    >>> ld.sel(b=slice('data2', 'data3'))
    ldarray([[11, 12],
            [14, 15]])
    Coordinates: (2, 2)
      a: [1 2]
      b: ['data2' 'data3']

    The coordinates will be dropped if the shape is changed by a math operation. In this case the user is responsible
    for casting the array back into a ldarray if needed. 
    >>> ld.sum(axis=0)
    array([23, 25, 27])

    >>> ld.T
    ldarray([[10,  1],
             [11,  2],
             [12,  3]])
      
    Indexing tolerance can be set on a per-dimension basis, 

    >>> coords = Coords(a=[1.2, 2.4, 3.1], b=[4,5], idx_precision=dict(a=1e-2))
    >>> ld2 = ldarray([[10, 11],[12, 13],[14, 15]], coords=coords, dtype=np.float64)

    >>> # this will raise an error because the selection is further than 1e-2 away from any coordinate value
    >>> ld2[dict(a=1.21)]

    >>> # but this will work
    >>> ld2[dict(a=1.201)]


    """
    def __new__(cls, data=None, coords=None, dtype=None):

        # cast coords as a OrderedDictionary type
        if not isinstance(coords, Coords):
            coords = Coords(**coords)
            
        # create 0 filled array if no data is given in the constructor
        if data is None:
            obj = np.zeros(coords.shape, dtype=dtype).view(cls)

        # cast input data to ldarray type
        else:             
            obj = np.asarray(data).view(cls)

            # If dim is not compatible with the data shape return a standard numpy array
            if (coords is None) or (not check_shapes(obj.shape, coords.shape)):
                raise TypeError(
                    "Coordinates of shape {} are not compatible with data of shape {}.".format(coords.shape, obj.shape)
                )

        # copy coords and assign as member variable
        setattr(obj, "coords", coords)

        return obj
    
    def __array_finalize__(self, obj):
        # required method of subclasses of numpy. Sets unique member variables of new instances
        
        # if called from __new__, obj will be none. Skip this method and let __new__ handle the coordinate assignments
        if obj is None: 
            return
        
        # array finalize is called when array is cast to a new type, indexed, or whenever a new array with a different
        # shape is created (i.e. transpose). By default, drop the coordinates which are most likely out of date now.
        # Coordinates will be added back by lower level functions if the shape stayed the same.
        self.coords = None


    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        # for some math functions, the shape will change. Drop the coordinates and revert to a standard numpy array
        # rather than support every math function like xarray does. 

        args = []
        for input_ in inputs:
            if isinstance(input_, ldarray):
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        if outputs:
            out_args = []
            for output in outputs:
                if isinstance(output, ldarray):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)

        # if the shape happens to be the same during the math operation, restore the coordinates
        if isinstance(results, np.ndarray) and self.coords and check_shapes(results.shape, self.coords.shape):
            results = results.view(ldarray)
            results.coords = dcopy(self.coords)
        else:
            results = results.view(np.ndarray)

        return results
    

    def sel(self, **keys):
        return self[keys]


    def __getitem__(self, key):
        try:
            return self.getitem(key)
        except Exception:
            return super().__getitem__(key)


    def getitem(self, key):
        # called whenever array is indexed
        
        # if coords were dropped, use the normal ndarray __getitem__, dictionary coordinates will raise an error here
        if self.coords is None:
            return super().__getitem__(key)

        # if index is a dictionary, use the dimension labels to index
        if isinstance(key, dict):
            # get standard numpy indices, will be a tuple of slices of length equal to the
            # number of dimensions.
            idx = self._coord2idx(key)

            # index object with this __getitem__ method. Not recursive because the index value
            # is no longer a dictionary.
            obj = self[idx]

            return obj

        # index is a standard index of slices or integers so pass key to the numpy indexing routine.
        # this object will have the same coords (not copied) as the object it was indexed from since numpy uses 
        # the __array_finalize__ method declared above to construct the new array.
        obj = super(ldarray, self).__getitem__(key)
        
        # shape length can be greater after indexing if np.newaxis was used. In this case just
        # return a standard numpy array and make the user responsible for adding dimensional labels.
        if len(obj.shape) > len(self.shape):
            return obj.view(np.ndarray)

        # obj could be a single value. In this case the object is not a numpy array and has no coordinates, 
        # just return the object
        if not len(obj.shape):
            return obj

        # the coords of obj have been dropped by array_finalize, start with the coords of the un-indexed object 
        ncoords = dcopy(self.coords)

        # At this point, we need to index the dimension dictionary so it matches the obj data,
        # and remove axis that were indexed out completely.

        # Cast index key as a tuple if it's a single value
        nkey = tuple(key) if isinstance(key, (tuple, list)) else (key,)

        # Initialize list of indices for each dimension that will be used to index the label arrays in dim. 
        # Length is the original array shape length so it matches ndim.
        idx = [slice(None,None) for i in range(len(self.shape))]
        
        # step through index keys and update idx with the appropriate keys.
        # Keys are always in order of the array dimensions, but axis can be skipped with the Ellipsis operator.
        idx_i = 0 
        for ii, k in enumerate(nkey):
            # jump the current index (idx_i) ahead if there is an Ellipsis.
            if isinstance(k, type(Ellipsis)):
                # key after an Ellipsis indexes the dimension starting from the end of the key list
                idx_i = len(idx) - (len(nkey) - idx_i) 

            else:
                # update idx with the key, if no key is given for a axis it defaults to ':'
                idx[idx_i] = k

            idx_i += 1

        # use idx to index each array of dimension labels in ndim
        for i, (k,v) in enumerate(self.coords.items()):
            # numpy removes the dimension if indexed with a integer, so remove it from the dimension label dictionary.
            if isinstance(idx[i], int):
                ncoords.pop(k)

            else:
                # reduce the label array for the current axis to match the indexed numpy array.
                # idx has a value for every dimension so we can use i to get the correct index key
                ncoords[k] = v[idx[i]]

        # revert to standard numpy array if we weren't able to keep coords consistent with the numpy array data
        if not check_shapes(obj.shape, ncoords.shape):
            return obj.view(np.ndarray)

        # if dim and the obj shape match, update the dim member of the indexed obj and return
        obj.coords = ncoords
        return obj


    def __setitem__(self, key, value):
        # __setitem__ cannot change the array shape, so we don't need to modify the coordinates. This is only 
        # overloaded to support dictionary indices.
        
        # if the index key is not a dictionary, use the numpy __setitem__
        if not isinstance(key, dict):
            super().__setitem__(key, value)

        # if key is dictionary, convert to standard indices with _coord2idx and set value
        else:
            idx = self._coord2idx(key)
            # call __setitem__ again, but this time with a standard index
            self[idx] = value
        

    def squeeze(self):
        """ Same as numpy.squeeze but also removes the axis labels
        """
        # build full idx key of all the dimensions
        idx = [slice(None) for i in range(len(self.shape))]

        for i, s in enumerate(self.shape):
            # if axis length is 1, replace index key with an integer index. numpy will remove 
            # the axis and the __getitem__ routine will remove the indexed out axis label
            if s <= 1:
                idx[i] = 0

        # call __getitem__ and return
        return self[tuple(idx)]


    def __str__(self):

        s = super().__repr__()

        if self.coords is None:
            return s

        # append coordinates to numpy output
        s+='\nCoordinates: ' + str(self.shape)
        for k, v in self.coords.items():

            if isinstance(v[0], datetime.datetime):
                v = np.array(v).astype('datetime64[m]')

            s+='\n  '+k + ': '+ np.array2string(v, threshold=3, suppress_small=True, edgeitems=2, prefix="  ")
        
        return s + '\n'

    def __repr__(self):
        return str(self)
    

    def _coord2idx(self, dct_idx):
        """ 
        Converts dictionary indices to standard numpy indices.
        """

        # Start with list of slices that index the full range of each dimension. The slices will be updated with the
        # bounds given in the dictionary index
        np_index = [slice(None,None) for i in range(len(self.shape))]
        dim_keys = list(self.coords.keys())
        
        for k, v in dct_idx.items():
            # Return a type error if the dictionary has a key that is not tracked in the dimensional dictionary.
            if k not in dim_keys:
                raise TypeError('Invalid index key: {}'.format(k))

            # get the index of the current dimension key in the array shape. dim_keys is the keys from an
            # Ordered Dictionary so the order will hold.
            np_i = dim_keys.index(k)

            # get values of the dimension labels. This is a 1D numpy array where each value is unique
            coords_k = self.coords[k]

            is_idx_slice = isinstance(v, slice)

            # cast v as a slice if not already, this allows us to use v.start and v.stop below. Both v.start and 
            # v.stop will be the same when a single value is cast to a slice.
            if not is_idx_slice:
                v = slice(v, v, None)
            
            # check if this dimension has a custom handler defined
            if k in self.coords.idx_handlers.keys():
                # get handler from dictionary
                handler = self.coords.idx_handlers[k]

                # call handler for each start, stop and step value
                slc_list = [handler(vv, coords_k) if vv is not None else None for vv in [v.start, v.stop, v.step]]

                # populate numpy index with slice of standard indices
                np_index[np_i] = slice(slc_list[0], slc_list[1] + 1, slc_list[2])

            # The dimensional key will be in the idx_precision dictionary if the labels are floats.
            # Use approximate indexing based on index precision given in the lddim class.
            elif k in self.coords.idx_precision.keys():
                # get dimension precision
                precision = self.coords.idx_precision[k]

                # initialize slice values to None
                s_start, s_stop, s_step = None, None, v.step
                # temporary array to store start and stop indices
                s_temp = []
                # for both start and stop labels, find nearest index if it exists with the given precision
                for v_s in [v.start, v.stop]:
                    # skip if the label is None
                    if v_s == None:
                        s_temp.append(None)
                        continue
                    # subtract start value from label values
                    label_diff = np.abs(v_s - coords_k)

                    # get minimum value and index from difference array
                    lmin = np.min(label_diff)
                    lmin_arg = np.argmin(label_diff)

                    # raise Type error if no label exists within given precision
                    if lmin > precision:
                        raise IndexError(
                            "Coordinate {} is outside precision given for dimension key {}.".format(v.start, k)
                        )
                    # set slice value to index of minimum value if within precision
                    s_temp.append(lmin_arg)
                
                # unpack temporary array
                s_start, s_stop = s_temp
                s_stop = s_stop+1 if s_stop is not None else s_stop
                # populate numpy index with slice of standard indices
                np_index[np_i] = slice(s_start, s_stop, s_step)


            # label index is exact (integer or string) so we use the lookup table
            else:
                # get lookup table for current dimension
                label_lut = self.coords.idx_label_lut[k]

                # convert labels to standard indices and build slice. Slice step is not modified, a step of 2 will 
                # still index every other value.
                s_start = label_lut[v.start] if v.start != None else None
                s_stop = label_lut[v.stop] +1 if v.stop != None else None
                s_step = v.step

                # populate numpy index with slice of standard indices
                np_index[np_i] = slice(s_start, s_stop, s_step)

            # if indexed with single values, convert the index from a slice to a integer
            if not is_idx_slice:
                np_index[np_i] = int(np_index[np_i].start)

        return tuple(np_index)


    def save(self, filepath: str):
        """
        Saves to disk in numpy structured array format (.npy).
        """

        if self.coords is None:
            return np.save(filepath, self)
        
        # initialize value and dtype of structured array
        dim_value = []
        dim_dtype = []

        # build dtype and value from dimension labels
        # dtype for a structured array is a tuple in the format (name, dtype, shape)
        for k,v in self.coords.items():
            dim_value.append(v)
            dim_dtype.append((k, v.dtype, v.shape))

        value = [self, tuple(dim_value)]
        dtype = [('data', self.dtype, self.shape), ('coords', dim_dtype, (1,))]
        
        # create structured array
        structure = np.array([tuple(value)], dtype=dtype)

        # save to file
        np.save(filepath, structure)

    @classmethod
    def load(cls, filepath: str):
        """
        Loads a ldarray from disk. (.npy)
        """
        # load structured array
        structure = np.load(filepath)

        if structure.dtype.names is not None and "coords" not in structure.dtype.names:
            return np.array(structure)
        
        # pull the dimension labels from the array
        coords_s = structure['coords'][0]
        data = structure['data'][0]

        # build coords from dim structure
        coords = Coords(**{k : coords_s[k][0] for k in coords_s.dtype.names})
        
        # return data array
        return ldarray(data, coords=coords)
            

