# np-struct

`np-struct` is a user friendly interface to NumPy structured arrays, with added support for transferring arrays across interfaces.
 
The `Struct` type is designed to mirror the struct typedef in C, but can be used for any complicated data structure. They behave similar to the standard `ndarray`, but support mixed data types, bitfields, labeling, and variable length arrays. Arrays are easily written or loaded from disk in the standard `.npy` binary format.

## Installation

```bash
pip install np-struct
```

## Usage

```python
import np_struct 
from np_struct import Struct
import numpy as np
```

### Structures

Create a C-style structure with Numpy types.

```python
class example(Struct):
    data1 = np.uint32()
    data2 = np.complex128([0]*3)
```

Structures can be initialized with arbitrary shapes by using the `shape` kwarg:
```python
ex = example(shape = (3,), byte_order='>')
ex[0].data2 = 1 + 2j
```
```python
>>> ex
Struct example: (3,)
[
    data1:  uint32[0]
    data2:  complex128[1.+2.j 1.+2.j 1.+2.j]
]
...
[
    data1:  uint32[0]
    data2:  complex128[0.+0.j 0.+0.j 0.+0.j]
]
```

Members can also be initialized by passing in their name to the constructor with an initial value. 
```python
>>> example(data2 = np.zeros(shape=(3,2)))
Struct example: 
    data1:  uint32[0]
    data2:  complex128[[0.+0.j 0.+0.j]
	       [0.+0.j 0.+0.j]
	       [0.+0.j 0.+0.j]]
```

The structure inherits from np.ndarray and supports all math functions that a normal structured array does.
To cast as a standard numpy structured array,

```python
>>>  ex.view(np.ndarray)
array([([0], [0.+0.j, 0.+0.j, 0.+0.j]), ([0], [0.+0.j, 0.+0.j, 0.+0.j]),
       ([0], [0.+0.j, 0.+0.j, 0.+0.j])],
      dtype=[('data1', '<u4', (1,)), ('data2', '<c16', (3,))])
```

Nested structures are also supported,
```python
class nested(Struct):
    field1 = example()
    field2 = example()

n = nested()
n.field1.data2 += 1j*np.pi
```
```bash
>>> n
Struct nested: 
    field1:  Struct example: 
              data1:  uint32[0]
              data2:  complex128[0.+3.14159265j 0.+3.14159265j 0.+3.14159265j]
    field2:  Struct example: 
              data1:  uint32[0]
              data2:  complex128[0.+0.j 0.+0.j 0.+0.j]
```

To save to disk,
```python
n = nested(shape=2)
n[1].field2.data1 = 3
np.save("test.npy", n)
n_disk = nested(np.load("test.npy"))
```
```python
>>> n_disk.field2.data1
array([[[0]],

       [[3]]], dtype=uint32)
```

### Labeled Arrays

`np-struct` implements a very scaled down version of [Xarray](https://docs.xarray.dev/en/stable/) DataArrays. 
`ldarray` behaves exactly the same as standard numpy arrays (no need to use .values to avoid errors), can be
written to disk in the standard `.npy` binary format, and supports indexing with coordinates. 

Math operations that change the coordinates or array shape (i.e. sum or transpose) silently revert the labeled array 
to a standard numpy array without coordinates. `np-struct` leaves it up to the user to re-cast the array as an
`ldarray` with the appropriate coordinates.


```python
>>> from np_struct import ldarray

>>> coords = dict(a=[1,2], b=['data1', 'data2', 'data3'])
>>> ld = ldarray([[10, 11, 12],[13, 14, 15]], coords=coords, dtype=np.float64)
>>> ld
ldarray([[10, 11, 12],
         [13, 14, 15]])
Coordinates: (2, 3)
    a: [1 2]
    b: ['data1' 'data2' 'data3']
```

Arrays can be set or indexed with slices,

```python
>>> coords = dict(a=['data1', 'data2'], b=np.arange(0, 20, 0.2))
>>> data = np.arange(200).reshape(2, 100)
>>> ld = ldarray(data, coords=coords)
>>> ld
ldarray([[  0,   1, ... 98,  99],
         [100, 101, ... 198, 199]])
Coordinates: (2, 100)
  a: ['data1' 'data2']
  b: [ 0.   0.2 ... 19.6 19.8]
```

Coordinate indexing with slices is inclusive on the endpoint:
```python
>>> ld.sel(b=slice(15, 16), a="data1")
ldarray([75, 76, 77, 78, 79, 80])
Coordinates: (6,)
  b: [15.  15.2 ... 15.8 16. ]
```

Setting arrays with coordinates work similar to xarray, where dictionaries can be used as indices
```python
>>>  ld[dict(b = 0.2)] = 77
>>>  ld.sel(b = 0.2)
ldarray([77, 77])
Coordinates: (2,)
  a: ['data1' 'data2']
```

Real or complex-valued arrays can be written to disk uses the normal numpy methods if the coordinates are not needed, 
or, to keep the coords, use `ldarray.save()` and `load()`. Array is stored as a structured array in the usual 
`.npy` binary format.
```python
ld.save("ld_file.npy")
ldarray.load("ld_file.npy")
```

## Examples

[Struct example](./examples/structures.ipynb)  


## License

np-struct is licensed under the MIT License.
