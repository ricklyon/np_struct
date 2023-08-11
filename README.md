# np-struct

`np-struct` extends structured arrays in NumPy to be a bit more user friendly and intuitive, with added support for transferring structured arrays across serial or socket interfaces. 
 
Structured arrays are built to mirror the struct typedef in C/C++, but can be used for any complicated data structure. They behave similar to standard arrays, but support mixed data types, labeling, and unequal length arrays. Arrays are easily written or loaded from disk in the standard `.npy` binary format.

## Installation

```bash
pip install np-struct
```

## Usage

```python
import np_struct 
```

Create a c-style structure with Numpy types.

```python
from np_struct import Struct
import numpy as np

class example(Struct):
    data1 = np.uint32()
    data2 = np.complex128([0]*3)
```

Structures can be initialized with arbitrary shapes by using the `shape` kwarg:
```python
ex = example(shape = (3,), byte_order='>')
ex[0].data2 = 1 + 2j
```
```bash
>>> ex
Struct example: (3,)
[
    psize:  uint16[0]
    data1:  uint32[0]
    data2:  complex128[1.+2.j 1.+2.j 1.+2.j]
]
...
[
    psize:  uint16[0]
    data1:  uint32[0]
    data2:  complex128[0.+0.j 0.+0.j 0.+0.j]
]
```

Members can also be initialized by passing in their name to the constructor with an inital value. 
```bash
>> example(data2 = np.zeros(shape=(3,2)))
Struct example: 
    psize:  uint16[0]
    data1:  uint32[0]
    data2:  complex128[[0.+0.j 0.+0.j]
	       [0.+0.j 0.+0.j]
	       [0.+0.j 0.+0.j]]
```

The structure inherits from np.ndarray and supports all math functions that a normal structured array does.
To cast as a standard numpy array:

```bash
>> ex2.view(np.ndarray)
array([([0], [0], [[0.+0.j, 0.+0.j], [0.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]])],
      dtype=[('psize', '<u2', (1,)), ('data1', '<u4', (1,)), ('data2', '<c16', (3, 2))])
```

Nested structures are also supported:
```python
class nested(Struct):
    field1 = example()
    field2 = example()

n = nested()
n.field1.data2 += j*np.pi
```
```bash
>> n
Struct nested: 
    field1:  Struct example: 
              psize:  uint16[0]
              data1:  uint32[0]
              data2:  complex128[0.+3.14159265j 0.+3.14159265j 0.+3.14159265j]
    field2:  Struct example: 
              psize:  uint16[0]
              data1:  uint32[0]
              data2:  complex128[0.+0.j 0.+0.j 0.+0.j]
```


Examples:

[Transfering structures over an interface](./examples/structures.ipynb)  
[Labeled arrays](./examples/ldarray.ipynb)


## License

np-struct is licensed under the MIT License.
