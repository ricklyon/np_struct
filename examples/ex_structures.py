

import numpy as np
from structures import Struct, ldarray, lddim
from pathlib import Path
from structures.fields import uint16, float32

try:
    dir_ = Path(__file__).parent
except:
    dir_ = Path().cwd()


class spam(Struct):
    a = float32()
    b = uint16()


s = spam(a=np.array([0,1,3,4]))
