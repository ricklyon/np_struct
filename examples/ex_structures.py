
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

class spam(Struct):
    comments = string_()
    freq_ghz = float64()
    data = np.array([0], dtype='complex128')

    
s = spam(freq_ghz=np.array([1,2,3], dtype=np.uint8))

s.freq_ghz = 1

s.comments = ['test', 'test']
