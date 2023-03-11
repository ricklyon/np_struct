
from numpy import uint8, uint16, float64, int16, uint32, string_
import numpy as np
from structures import Struct, ldarray, lddim
from enum import Enum
from rfnetwork import Sparam
from pathlib import Path

try:
    dir_ = Path(__file__).parent
except:
    dir_ = Path().cwd()


class dim(Struct):
    freq = np.zeros(10)
    p_b = np.array([1,2,3])
    p_a = np.array([1,2,3])
    
class sp(Struct):
    dim = dim()
    sdata = np.array([[1,2],[1,2]])

class spam(Struct):
    comments = string_()
    freq_ghz = float64()
    data = np.array([0], dtype='complex128')

d = sp()
s = spam(freq_ghz=np.array([1,2,3], dtype=np.uint8))
