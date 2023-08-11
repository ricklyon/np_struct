import unittest
from np_struct import ldarray
import numpy as np
from numpy import testing as npt
import datetime as dt
from dateutil import relativedelta as rdt    

class TestLdArray(unittest.TestCase):

    def test_exact_index(self):
        dim = dict(a=np.arange(0, 20), b=['data1', 'data2', 'data3'])
        data = np.arange(60).reshape(20, 3)
        ld = ldarray(data, dim=dim)

        npt.assert_array_equal(ld.sel(a=slice(3, 6), b='data2'), np.array([10, 13, 16, 19]))

        npt.assert_array_equal(ld.sel(a=19, b=slice('data2', 'data3')), np.array([58, 59]))

        npt.assert_array_equal(ld[1], np.array([3,4,5]))

        ld[dict(a=15, b='data3')] = 3
        npt.assert_array_equal(ld[1], np.array([3,4,5]))
        npt.assert_array_equal(ld[15, 2], 3)

    def test_float_index(self):

        dim = dict(a=['data1', 'data2'], b=np.arange(0, 20, 0.2))
        data = np.arange(200).reshape(2, 100)
        ld = ldarray(data, dim=dim)

        npt.assert_array_equal(ld.sel(b=19.6), np.array([98, 198]))

        ld[dict(b = 0.2)] = 77

        npt.assert_array_equal(ld.sel(b = 0.2), np.array([77, 77]))
        npt.assert_array_equal(ld.sel(b=19.6), np.array([98, 198]))

        self.assertTrue(np.sum(ld, axis=0).__class__ == np.ndarray)
        self.assertTrue(np.sum(ld, axis=0).shape == (100,))

    def test_dates(self):

        start = dt.date(1989, 12, 13)
        dates = [start + rdt.relativedelta(days = i) for i in range(12)]
        ld = ldarray(np.arange(12), dim=dict(dates=dates))

        npt.assert_array_equal(ld.sel(dates = dt.date(1989, 12, 14)), 1)

        ld[dict(dates = dt.date(1989, 12, 14))] = 10

        npt.assert_array_equal(ld.sel(dates = slice(dt.date(1989, 12, 14), dt.date(1989, 12, 16))), [10,2,3])

if __name__ == '__main__':
    unittest.main()