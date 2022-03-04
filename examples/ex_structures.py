
from numpy import uint8, uint16, float64, int16, uint32, string_
import numpy as np
from structures import Struct, Packet, ldarray, lddim
from enum import Enum
from rfnetwork import Sparam
from pathlib import Path

try:
    dir_ = Path(__file__).parent
except:
    dir_ = Path().cwd()


sp = Sparam(dir_/'data/QPL9057.s2p')

freq, sdata = sp.freq, sp.sdata

class dim(Struct):
    freq = np.zeros(10)
    p_b = np.array([1,2,3])
    p_a = np.array([1,2,3])
    
class sp(Struct):
    dim = dim()
    sdata = np.array([[1,2],[1,2]])

    
d = sp()

class spam(Struct):
    comments = string_()
    freq_ghz = float64()
    data = np.array([0], dtype='complex128')

    
s = spam(freq_ghz=np.array([1,2,3], dtype=np.uint8))

s.freq_ghz = 1

s.comments = ['test', 'test']


dim = sp.dim

value = []
dtype = []

for k,v in dim.items():
    value.append(v)
    dtype.append((k, v.dtype, v.shape))
    
# value.append(sp.sdata)
# dtype.append(('sdata', sp.sdata.dtype, sp.sdata.shape))
 

a = np.array([tuple(value)], dtype=dtype)

dtype.append((k, value.dtype, value.shape))

value = []
for (k, v, _enum) in self._defs_list:
    value.append(self._get_field_value(k,v))