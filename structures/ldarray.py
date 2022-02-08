import numpy as np
import datetime as dt
from collections import OrderedDict
import time

def round_to_float(value, multiple=1):
    """ Rounds value to nearest multiple. Multiple can be greater or less than 1.

        Example:
            round_to_float(7.77777, 1e-3) --> 7.778

            round_to_float(7.77777, 3)    --> 9.0    
    """

    invmul = 1/multiple

    r1 = value/multiple
    w = r1//1
    w = np.where((r1%1) >= 0.5, w+1, w)
    
    mlog10 = np.log10(multiple)
    
    if mlog10 > 0:
        return w/invmul
    else:
        return np.around(w/invmul, int(np.abs(mlog10)))

class lddim(OrderedDict):
    """ Labeled dimension indices for ldarray. Conditions values to work as indices, but otherwise, same as 
        an Ordered Dictionary. 

        Accepts a key-value pair 'idx_precision' that will not be included in dictionary, 
        but can be optionally used to specify index precision for each dimension. Value of idx_precision is
        a dictionary with keys matching the lddim keys and values that will override the default precision
        for that dimension.
        
        Example:
            dim = lddim(a=[1.2, 2.4, 3.1], b=[4,5,6], idx_precision={'a':1e-6})

            dim = lddim(a=[1.2, 2.4, 3.1], b=[4,5,6], idx_precision={'a':2})

        Index precision can also be modified after the constructor is called:

            dim.set_precision(b=2, a=1e-3)
    """
    DEFAULT_PRECISION = 1e-6

    def __init__(self, **kwargs):
        ## Pop idx_precison from kwargs. Floating point indices default to 6 decimal precision.
        self.idx_precision = kwargs.pop('idx_precision', {})

        ## Call OrderedDict __init__ to create dictionary of values
        super().__init__(**kwargs)

        ## Cast each item in the dictionary to a numpy array. If items are objects, numpy will use the object
        ## dtype for the array. Single items (not arrays) will be cast as single item arrays.
        for k,v in self.items():
            ## Cast single items as numpy arrays with np.array([v])
            self[k] = np.array(v) if isinstance(v, (list, np.ndarray, tuple)) else np.array([v])

            ## Provide default values for index precision if the values are floats
            f64 = np.dtype(np.float64)
            f32 = np.dtype(np.float32)

            if (k not in self.idx_precision) and (self[k].dtype in [f64, f32]):
                self.idx_precision[k] = self.DEFAULT_PRECISION
            
    def set_precision(self, **kwargs):
        """ Sets precision for given dimensional indices. Accpets key value pairs where key is dimensional key
            and value is index precision.

            Example:
                dim = lddim(a=[1.2, 2.4, 3.1], b=[4,5,6])
                dim.set_precision(b=2, a=1e-3)
        """

        ## Update precisions only if the key already exsists in idx_precision.
        ## This ensures only floats have idx_precision specified.
        for k,v in self.idx_precision.items():
            self.idx_precision[k] = kwargs.get(k, self.idx_precision[k])

    @property
    def shape(self):
        """ Returns shape of the ldarray that uses this lddim for it's dimensional indexing.
        """
        return tuple([len(v) for k,v in self.items()])

    def __str__(self):
        ## breaks out each key-value pair into it's own line for easier reading printed dictionary
        s = '{\n'
        for k,v in self.items():
            s += k + ': ' + str(v) + '\n'
        return s+ '}'

    def __repr__(self):
        return self.__str__()

class ldarray(np.ndarray):
    """ Labeled numpy array.
    """
    def __new__(cls, input_=None, dim=None, dtype=None):
        
        if np.all(input_ == None):
            shape = tuple([len(v) for k,v in dim.items()])
            obj = np.zeros(shape, dtype=dtype).view(cls)
        else:   
            obj = np.asarray(input_).view(cls)

        obj.dim = lddim(dim)
        obj.rdim = {}

        for k,v in dim.items():
            obj.rdim[k] = {vv:i for i,vv in enumerate(obj.dim[k])} if obj.idx_types[k] == 'exact' else {}
        
        return obj

    def save(self, file_):
        np.savez(file_, data=self, dim=self.dim, idx_types=self.idx_types)

    @classmethod
    def load(self, file_):
        loadf = np.load(file_.with_suffix(r'.npz'), allow_pickle=True)
        data = loadf['data'][()]
        dim = loadf['dim'][()]
        idx_types = loadf['idx_types'][()]
        return ldarray(data, dim, idx_types)

    def __array_finalize__(self, obj):
        if obj is None: return
        self.dim = getattr(obj, 'dim', lddim())
        self.idx_types = getattr(obj, 'idx_types', {})
        self.rdim = getattr(obj, 'rdim', {})
        
    def __getitem__(self, key):
        ntypes = dict(self.idx_types)
        ndim = lddim(self.dim)

        if isinstance(key, dict):
            idx = self._v2idx(key)
            obj = self[idx]

            for i, (k,v) in enumerate(self.dim.items()):
                ndim[k] = np.array(v)[idx[i]]

                ## squeeze dimensions if indexed by integer
                if isinstance(idx[i], int):
                    ndim.pop(k)
                    ntypes.pop(k)

                if k in self.idx_types.keys():
                    ntypes[k] = self.idx_types[k]

        else:
            ## pass key to the numpy indexing routine
            obj = super(ldarray, self).__getitem__(key)

            if len(obj.shape) > len(self.shape):
                return np.array(obj)


            ## index dimensions
            nkey = tuple(key) if isinstance(key, (tuple, list)) else (key,)
            idx = [slice(None,None) for i in range(len(self.shape))]
            

            ## build full index, supporting Ellipsis
            cur_idx = 0 
            for ii, k in enumerate(nkey):
                if isinstance(k, type(Ellipsis)):
                    cur_idx = len(idx) - (len(nkey) - cur_idx) 
                else:
                    idx[cur_idx] = k
                cur_idx += 1

            ## if shape matches the dimension number, we can attempt to index dimensions,
            for i, (k,v) in enumerate(self.dim.items()):
                if isinstance(idx[i], int):
                    ndim.pop(k)
                    ntypes.pop(k)
                else:
                    ndim[k] = np.array(v)[idx[i]]

        if not self.check_shapes(obj.shape, ndim.shape):
            return np.array(obj)
        elif len(ndim):
            return ldarray(obj, dim=ndim, idx_types=ntypes)
        else:
            return np.array([obj])

    def check_shapes(self, a, b):
        if len(a) != len(b):
            return False
        
        for i in range(len(a)):
            if a[i] != b[i]:
                return False
        
        return True

    def get_axis_num(self, key):
        dim_keys = list(self.dim.keys())
        return dim_keys.index(key)

    def __setitem__(self, key, value):
        if not isinstance(key, dict):
            super(ldarray, self).__setitem__(key, value)
            return

        idx = self._v2idx(key)
        self[idx] = value
        
    def squeeze(self):
        ndim = dict(self.dim)
        ntypes = dict(self.idx_types)
        idx = [slice(None) for i in range(len(self.shape))]
        for i, (k,v) in enumerate(self.dim.items()):
            if (len(v) <= 1):
                ndim.pop(k)
                ntypes.pop(k, None)
        return ldarray(super(ldarray, self).squeeze(), dim=ndim, idx_types=ntypes)

    def _v2idx(self, dct_indx):
        """ Converts dictionary indices to standard numpy indices. Using dictionaries as indices avoids the need
            to remember the dimensions order.
        """
        np_index = [slice(None,None) for i in range(len(self.shape))]
        dim_keys = list(self.dim.keys())
        int_flag = False
        
        for k,v in dct_indx.items():
            if k not in dim_keys:
                raise TypeError('Invalid index key: {}'.format(k))

            np_i = dim_keys.index(k)
            integer_idx = (v.__class__ != slice)

            if integer_idx:
                v = slice(v, v, None)

            if self.idx_types[k] == 'time':
                kvals = self.dim[k]
                v0 = kvals[0]
                vt_s =  np.array([(vt - v0).total_seconds() for vt in kvals])
                
                if v.step is None:
                    istart = abs(abs(v.start - v0).total_seconds() - vt_s).argmin() if v.start is not None else None
                    istop = abs(abs(v.stop - v0).total_seconds() - vt_s).argmin()+1 if v.stop is not None else None
                    np_index[np_i] = slice(istart,istop,None)
                else:
                    cur_vt = v.start
                    step_idx = []
                    while cur_vt <= v.stop:
                        step_idx.append( abs( abs(cur_vt - v0).total_seconds() - vt_s).argmin())
                        cur_vt = cur_vt + v.step
                    np_index[np_i] = list(step_idx)
                
            elif self.idx_types[k] == 'approx':
                kvals = np.array(self.dim[k])
                dstart = (np.abs(v.start-kvals)).argmin() if v.start != None else None
                dstop = (np.abs(v.stop-kvals)).argmin()+1 if v.stop != None else None
                dstep = v.step
                np_index[np_i] = slice(dstart,dstop,dstep)
            else:
                dstart = self.rdim[k][v.start] if v.start != None else None
                dstop = self.rdim[k][v.stop] +1 if v.stop != None else None
                dstep = v.step
                np_index[np_i] = slice(dstart,dstop,dstep)

            ## replacing slices with integers ensures that numpy will squeeze that dimension
            if integer_idx:
                np_index[np_i] = int(np_index[np_i].start)

        return tuple(np_index)

    def run_loop(self, func, index_to=None, dtype='float64', progress_interval=0):
        ## get rid of element_shape, need to have self be the full dimensioned value and index appropriately in the run_loop

        dim_keys = list(self.dim.keys())
        shape = self.shape[:index_to]
        func_run_idx = [np.prod(shape[i+1:]) for i in range(len(shape))]

        if func.__class__ != list:
            func_list = [None]*len(shape)
            func_list[-1] = func
            func = func_list

        vals = [None]*len(self.dim)
        stime = time.time()
        iter_ = np.prod(shape)

        for i in range(iter_):
            idx = np.unravel_index(i, shape)
            for d in range(len(shape)):
                vals[d] = self.dim[dim_keys[d]][idx[d]]
                if ((i % func_run_idx[d]) == 0) and (func[d] != None):
                    self[idx] = func[d](*vals[:d+1], idx=idx[:d+1])
            
            if progress_interval:
                if (i % progress_interval == 0):
                    sys.stdout.write('\r {:.2f}% {}\t\t\t\t'.format(((i+1)/iter_)*100, vals))
        
        if progress_interval:
            print('\nIterations: {}, Timer: {:0.4f}s'.format(self.iter, time.time()-stime))