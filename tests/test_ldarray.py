import unittest
from structures import ldarray
from pathlib import Path
from time import time
import numpy as np
import matplotlib.pyplot as plt

class TestLdArray(unittest.TestCase):

    def test_simple(self):
        dim = dict(a=np.arange(0, 20), b=['data1', 'data2', 'data3'])
        data = np.arange(60).reshape(20, 3)
        ld = ldarray(data, dim=dim)

        plt.plot(ld.sel(b='data1'))
        plt.show()
        print(ld)

        print(ld.sel(a=slice(3, 6), b=slice('data1', 'data3', 2)))

    # def test_dates(self):

if __name__ == '__main__':
    unittest.main()