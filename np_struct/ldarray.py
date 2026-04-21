import numpy as np
import datetime as dt
from scipy import interpolate, ndimage
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


def datetime_idx_handler(v: datetime.datetime, coords: np.ndarray):
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
    else:
        raise NotImplementedError(f"Unsupported datetime index: {type(v)}")

    # get index of the minimum distance to the indexing timestamp
    return np.argmin(np.abs(coords_ts - v_ts))


def float_idx_handler(v: float, coords: np.ndarray, precision: float):
    """
    Convert float type coord to standard index
    """
    # subtract start value from label values
    label_diff = np.abs(v - coords)

    # get minimum value and index from difference array
    lmin = np.min(label_diff)
    lmin_arg = np.argmin(label_diff)

    # raise Type error if no label exists within given precision
    if lmin > precision:
        raise IndexError(
            "Coordinate {} is outside precision given for dimension key ({}).".format(v, precision)
        )
    
    # set slice value to index of minimum value if within precision
    return lmin_arg


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
    
        # cast as list
        v = [v] if isinstance(v, (str, int, float)) else v
        v_1d = np.atleast_1d(v)

        f64 = np.dtype(np.float64)
        f32 = np.dtype(np.float32)

        # Provide default values for index precision if the values are floats
        if v_1d.dtype in [f64, f32]:
            # add entry to the index precision for this dimension if it doesn't exist 
            if k not in self.idx_precision.keys():
                if len(v_1d) == 1:
                    self.idx_precision[k] = 1e-10
                else:
                    self.idx_precision[k] = np.average(np.diff(v_1d))

            super().__setitem__(k, v_1d)

        elif isinstance(v_1d[0], (datetime.datetime, datetime.date)):
            # cast dates (only day/month/year) to more general datetime objects
            if isinstance(v_1d[0], datetime.date):
                v_1d = np.array([dt.datetime(year=d.year, month=d.month, day=d.day) for d in v])

            # use index handler for datetime objects
            self.idx_handlers[k] = datetime_idx_handler
            super().__setitem__(k, v_1d)

        else:
            # add to lookup table
            self.idx_label_lut[k] = {vv:i for i,vv in enumerate(v_1d)}
            super().__setitem__(k, v_1d)


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

    >>> coords = dict(a=[1.2, 2.4, 3.1], b=[4,5])
    >>> ld2 = ldarray([[10, 11],[12, 13],[14, 15]], coords=coords, dtype=np.float64, idx_precision=dict(a=1e-2))

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
            obj = np.asarray(data)
            
            if dtype is not None:
                obj = obj.astype(dtype)
                
            obj = obj.view(cls)

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
    

    def __copy__(self):
        obj = super().__copy__()
        obj.coords = dcopy(self.coords)
        return obj


    def __deepcopy__(self, memo):
        return self.__copy__()


    def sel(self, **keys):
        return self[keys]
    

    def __getitem__(self, key):
        # called whenever array is indexed
        
        # if coords were dropped, use the normal ndarray __getitem__, dictionary coordinates will raise an error here
        if self.coords is None:

            # raise an error if key is a dictionary
            if isinstance(key, dict):
                raise IndexError(f"Unable to use dictionary {key} to index an array without coords.")
            
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
        # this object will have the coords set to None by __array_finalize___
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
        try:
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
                    ncoords[k] = np.array(v)[idx[i]].squeeze()

            # revert to standard numpy array if we weren't able to keep coords consistent with the numpy array data
            if not check_shapes(obj.shape, ncoords.shape):
                return obj.view(np.ndarray)

            # if dim and the obj shape match, update the dim member of the indexed obj and return
            obj.coords = ncoords
            return obj
        
        # if the coords were unable to be indexed, clear the coords and return a unlabeled numpy array.
        except Exception:
            obj.coords = None
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
        
        LEN_THRESHOLD = 7
        s = super().__repr__()

        if self.coords is None:
            return s

        # append coordinates to numpy output
        s+='\nCoordinates: ' + str(self.shape)
        for k, v in self.coords.items():

            if isinstance(v[0], datetime.datetime):
                v = np.array(v).astype('datetime64[m]')
            if isinstance(v, np.ndarray):
                v_str = np.array2string(v, threshold=LEN_THRESHOLD, suppress_small=True, edgeitems=2, prefix="  ")
            else:
                # abbreviate long coordinate lists
                if len(v) > LEN_THRESHOLD:
                    v_start = v[:int(LEN_THRESHOLD / 2)]
                    v_end = v[-int(LEN_THRESHOLD / 2):]
                    v_str = str(v_start)[:-1] + ", ... " + str(v_end)[1:]
                else:
                    v_str = str(v)


            s+='\n  '+k + ': '+ v_str
        
        return s + '\n'

    def __repr__(self):
        return str(self)
    

    def _coord2idx(self, dct_idx: dict):
        """ 
        Converts dictionary indices to standard numpy indices.
        """

        # Start with list of slices that index the full range of each dimension. The slices will be updated with the
        # bounds given in the dictionary index
        np_index = [slice(None, None) for i in range(len(self.shape))]
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

            # get coordinate to index function for the coordinate type
            handler_kwargs = dict()
            if k in self.coords.idx_label_lut.keys():
                lut = self.coords.idx_label_lut[k]
                # convert coord index to string type if the lut keys are string (allow "1" to be 
                # selected with 1)
                is_str_type = isinstance(list(lut.keys())[0], str)
                handler = lambda x, *args, **kwargs: lut[str(x) if is_str_type else x]

            # check if this dimension has a custom handler defined
            elif k in self.coords.idx_handlers.keys():
                # get handler from dictionary
                handler = self.coords.idx_handlers[k]

            elif k in self.coords.idx_precision.keys():
                handler = float_idx_handler
                handler_kwargs["precision"] = self.coords.idx_precision[k]

            else:
                raise ValueError(f"Coordinate type not recognized for dimension {k}")

            # convert coordinate to standard index
            if isinstance(v, (list, tuple, np.ndarray)):
                # get standard indices for each value in list
                np_index[np_i] = [handler(vv, coords_k, **handler_kwargs) for vv in v]
                    
            elif isinstance(v, slice):
                # call handler for each start, stop and step value
                s_start, s_stop = [handler(vv, coords_k, **handler_kwargs) if vv is not None else None for vv in [v.start, v.stop]]
                s_stop = s_stop + 1 if s_stop is not None else s_stop

                # populate numpy index with slice of standard indices
                np_index[np_i] = slice(s_start, s_stop, v.step)
            else:
                # if indexed with single value
                np_index[np_i] = int(handler(v, coords_k, **handler_kwargs))

        # if more than one index is a list or array, numpy does pair-wise indexing. Otherwise, we can return the 
        # indices as is.
        if np.count_nonzero([isinstance(idx, list) for idx in np_index]) <= 1:
            return tuple(np_index)
        
        # create pairwise indices. 
        for i, idx in enumerate(np_index):
            # convert slice indices to a range of indices
            if isinstance(np_index[i], slice):

                start = 0 if idx.step is None else idx.start
                stop = self.shape[i] if idx.stop is None else idx.stop + 1
                step = 1 if idx.step is None else idx.step

                np_index[i] = np.arange(start, stop + 1, step)

            else:
                np_index[i] = np.atleast_1d(idx)

        # return a meshgrid of index values, the resulting array when this index is used will have the same
        # shape as each array in the axis positions. np.ix_ doesn't perform a full meshgrid broadcast, but ensures
        # the shapes are compatible. 
        return np.ix_(*np_index)

    def _interp_idx(self, dct_idx: dict):
        """ 
        Converts dictionary indices to interpolated indices.
        """

        # Start with list of slices that index the full range of each dimension. 
        interp_index = [np.arange(0, self.shape[i]) for i in range(len(self.shape))]
        dim_keys = list(self.coords.keys())
        
        for k, v in dct_idx.items():
            # Return a type error if the dictionary has a key that is not tracked in the dimensional dictionary.
            if k not in dim_keys:
                raise TypeError('Invalid index key: {}'.format(k))
            
            v = np.atleast_1d(v)

            # get the index of the current dimension key in the array shape. 
            np_i = dim_keys.index(k)
            # get values of the dimension labels. This is a 1D numpy array where each value is unique
            coords_k = self.coords[k]

            # check if this dimension has a custom handler defined
            if k in self.coords.idx_handlers.keys():
                # get handler from dictionary
                handler = self.coords.idx_handlers[k]
                interp_index[np_i] = [handler(vv, coords_k) for vv in v]

            # can't interpolate string coordinates, use nearest value
            elif isinstance(v[0], str):
                interp_index[np_i] = [self.coords.idx_label_lut[k][vv] for vv in v]

            # get the floating point "index" by interpolation for each coordinate value.
            else:
                coord_interp = interpolate.CubicSpline(coords_k, np.arange(0, self.shape[np_i]))
                interp_index[np_i] = coord_interp(v)

        # return a meshgrid of index values, the resulting array when this index is used will have the same
        # shape as each array in the axis positions. np.ix_ doesn't perform a full meshgrid broadcast, but ensures
        # the shapes are compatible. 
        return np.ix_(*interp_index)
    

    def save(self, filepath: str):
        """
        Save to disk in numpy structured array format (.npy).

        Parameters
        ----------
        filepath : str | Path
            filepath of .npy file
        """

        if self.coords is None:
            return np.save(filepath, self)
        
        # initialize value and dtype of structured array
        dim_value = []
        dim_dtype = []

        # build dtype and value from dimension labels
        # dtype for a structured array is a tuple in the format (name, dtype, shape)
        for k, v in self.coords.items():
            v = np.atleast_1d(v)
            dim_value.append(v)
            dim_dtype.append((k, v.dtype, v.shape))

        value = [self, tuple(dim_value)]
        dtype = [('data', self.dtype, self.shape), ('coords', dim_dtype, (1,))]
        
        # create structured array
        structure = np.array([tuple(value)], dtype=dtype)

        # save to file
        np.save(filepath, structure)

    def interpolate(
        self, 
        order: int = 3,
        output: np.ndarray = None,
        mode: str = "constant",
        cval: float = 0,
        prefilter: bool = True,
        dtype: np.dtype = None,
        **coords, 
    ):
        """
        Interpolate data at the given coordinates. String value coordinates are not interpolated and must 
        be included in the data coordinates. See scipy.ndimage.map_coordinates().

        Parameters
        ----------
        output : array_like, optional
            The array in which to place the output. By default an array of the same dtype as input will be created.
        order : int, default: 3
            The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
        mode : {"reflect", "grid-mirror", "constant", "grid-constant", "nearest", "mirror", "grid-wrap", "wrap"}
            The mode parameter determines how the input array is extended beyond its boundaries. Default is "constant".
        cval : float, default: 0.0
            Value to fill past edges of input if mode is "constant". Default is 0.0.
        prefilter : bool, default: False
            Determines if the input array is prefiltered with spline_filter before interpolation. 
            The default is False.
        dtype : np.dtype, optional
            The dtype of the returned array. By default, the dtype is the same as the input array, which may lead to 
            unexpected results if interpolating an integer array. 
        **coords
            coordinate values to interpolate at

        Examples
        --------

        >>> from np_struct import ldarray
        >>> import numpy as np
        >>> np.set_printoptions(suppress=True)

        >>> coords = dict(a=[1, 2], b=['data1', 'data2', 'data3'])
        >>> ld = ldarray([[10, 8, 6], [0, 2, 4]], coords=coords)
        >>> ld
        ldarray([[10.,  8.,  6.],
                [ 0.,  2.,  4.]])
        Coordinates: (2, 3)
        a: [1 2]
        b: ['data1' 'data2' 'data3']

        >>> ld.interpolate(a = [1.5, 2], dtype=np.float64)
        ldarray([[ 5.,  5.,  5.],
                [-0.,  2.,  4.]])
        Coordinates: (2, 3)
        a: [1.5 2. ]
        b: ['data1' 'data2' 'data3']

        Returns
        -------
        ldarray

        """

        # get an array of indices for each dimension. Indices may be floating point, in between integer indices. 
        # Each array will be the same number of dimensions but not equal shapes.
        interp_idx = self._interp_idx(coords)
        # shape of the data after interpolation
        result_shape = [m.shape[i] for i, m in enumerate(interp_idx)]
        # map_coordinates doesn't broadcast the indices like numpy does for advanced indexing. Broadcast 
        # index array to the same shape for each dimension.
        map_idx = [np.broadcast_to(m, result_shape) for m in interp_idx]

        if dtype is None:
            dtype = self.dtype

        data = ndimage.map_coordinates(
            self.astype(dtype), map_idx, output=output, order=order, mode=mode, cval=cval, prefilter=prefilter
        )

        # update coordinates to new interpolated ones
        data_coords = dcopy(self.coords)
        for k, v in coords.items():
            data_coords[k] = v

        return ldarray(
            data, coords=data_coords
        )

    @classmethod
    def load(cls, filepath: str, **kwargs):
        """
        Load a ldarray from disk. (.npy)

        Parameters
        ----------
        filepath : str | Path
            filepath of .npy file
        
        **kwargs
            kwargs passed to np.load(). allow_pickle must be set to True if array contains object types,
            or if datetime objects are used as coordinates.
        """
        # load structured array, allow pickled objects to support numpy arrays with object types
        structure = np.load(filepath, **kwargs)

        if structure.dtype.names is not None and "coords" not in structure.dtype.names:
            return np.array(structure)
        
        # pull the dimension labels from the array
        coords_s = structure['coords'][0]
        data = structure['data'][0]

        # build coords from dim structure
        coords = Coords(**{k : coords_s[k][0] for k in coords_s.dtype.names})
        
        # return data array
        return ldarray(data, coords=coords)
    
    def transpose(self, order: tuple):
        """
        Transpose axis by dimension name.
        """

        if self.coords is None:
            return super().transpose(order)
        
        order = list(order)
        dims = list(self.coords.keys())

        # list of dims skipped with ellipsis
        skipped_dims = [d for d in dims if d not in order]

        if Ellipsis in order:
            # index of ellipsis
            idx = order.index(Ellipsis)
            # add missing dimensions to the order in place of the ellipsis
            for i, d in enumerate(skipped_dims):
                order.insert(idx + i, d)

            # remove ellipsis
            order = [d for d in order if d != Ellipsis]

        # convert dimension name list to axis
        order_idx = [dims.index(d) for d in order]

        # reorder coords
        coords = Coords(
            **{d: dcopy(self.coords[d]) for d in order}, 
            idx_precision=self.coords.idx_precision,
            idx_handlers=self.coords.idx_handlers
        )

        return ldarray(super().transpose(order_idx), coords=coords)

            
            

