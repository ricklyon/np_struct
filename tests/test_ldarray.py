import unittest
from np_struct import ldarray, Coords
import numpy as np
from numpy import testing as npt
import datetime as dt
from dateutil import relativedelta as rdt    
import os


class TestLdArray(unittest.TestCase):

    def test_exact_index(self):
        coords = dict(a=np.arange(0, 20), b=['data1', 'data2', 'data3'])
        data = np.arange(60).reshape(20, 3)
        ld = ldarray(data, coords=coords)

        npt.assert_array_equal(ld.sel(a=slice(3, 6), b='data2'), np.array([10, 13, 16, 19]))
        npt.assert_array_equal(ld.sel(a=slice(3, 6), b='data1').coords["a"], [3, 4, 5, 6])

        npt.assert_array_equal(ld.sel(a=19, b=slice('data2', 'data3')), np.array([58, 59]))

        npt.assert_array_equal(ld[1], np.array([3,4,5]))

        ld[dict(a=15, b='data3')] = 3
        npt.assert_array_equal(ld[1], np.array([3,4,5]))
        npt.assert_array_equal(ld[15, 2], 3)

    def test_advanced_indexing(self):
        coords = dict(a=np.arange(0, 20), b=['data1', 'data2', 'data3'])
        data = np.arange(60).reshape(20, 3)
        ld = ldarray(data, coords=coords)

        ld_1 = ld[:, np.array([1, 0])]
        npt.assert_array_equal(ld_1, data[:, np.array([1, 0])])
        npt.assert_array_equal(ld_1.coords["b"], ["data2", "data1"])
        npt.assert_array_equal(ld_1.coords["a"], np.arange(0, 20))

        ld_2 = ld[np.array([1, 0]), 0]
        npt.assert_array_equal(ld_2, data[np.array([1, 0]), 0])
        npt.assert_array_equal(ld_2.coords["a"], [1, 0])

    def test_float_index(self):

        coords = dict(a=['data1', 'data2'], b=np.arange(0, 20, 0.2))
        data = np.arange(200).reshape(2, 100)
        ld = ldarray(data, coords=coords)

        npt.assert_array_equal(ld.sel(b=19.6), np.array([98, 198]))
        npt.assert_array_equal(ld.sel(b=slice(15, 16), a="data1"), [75, 76, 77, 78, 79, 80])
        npt.assert_array_almost_equal(ld.sel(b=slice(15, 16), a="data1").coords["b"], np.arange(15, 16.2, 0.2))

        ld[dict(b = 0.2)] = 77
        ld.sel(b = 0.2)

        npt.assert_array_equal(ld.sel(b = 0.2), np.array([77, 77]))
        npt.assert_array_equal(ld.sel(b = 0.2).coords["a"], ['data1', 'data2'])
        npt.assert_array_equal(list(ld.sel(b = 0.2).coords.keys()), ["a"])

        npt.assert_array_equal(ld.sel(b=19.6), np.array([98, 198]))

        self.assertTrue(isinstance(np.sum(ld, axis=0), np.ndarray))
        self.assertTrue(np.sum(ld, axis=0).shape == (100,))

    def test_dates(self):

        start = dt.date(2014, 12, 13)
        date = [start + rdt.relativedelta(days = i) for i in range(12)]
        ld = ldarray(np.arange(12), coords=dict(date=date))

        npt.assert_array_equal(ld.sel(date = dt.date(2014, 12, 14)), 1)
        npt.assert_array_equal(ld.sel(date = "2014-12-23T00:00"), 10)

        ld[dict(date = dt.date(2014, 12, 14))] = 10

        ld_slc = ld.sel(date = slice(dt.date(2014, 12, 14), dt.date(2014, 12, 16)))
        npt.assert_array_equal(ld_slc, [10, 2, 3])
        npt.assert_array_equal([d.day for d in ld_slc.coords["date"]], [d.day for d in date[1:4]])

    def test_drop_coords_math(self):

        ld = ldarray(np.ones((12, 12)), coords=dict(a=np.arange(12), b=np.ones(12)))

        self.assertTrue(ld.T.coords is None)
        self.assertTrue(ld.flatten().coords is None)
        self.assertTrue(ld.reshape(-1, 2).coords is None)

    def test_index_precision(self):

        coords = Coords(a=[1.2, 2.4, 3.1], b=[4,5], idx_precision=dict(a=1e-2))
        ld = ldarray([[10, 11],[12, 13],[14, 15]], coords=coords, dtype=np.float64)

        with self.assertRaises(IndexError):
            ld[dict(a=1.21)]

        npt.assert_array_equal(ld[dict(a=1.201)], [10, 11])

    def test_interpolation_real(self):
        
        # avoid interpolating at the endpoints, it's close to the right value but hard to test exactly
        t = np.linspace(0, 2 * np.pi, 21)
        t_int = np.linspace(0.5, 5.5, 61)

        data = np.array([np.sin(t), np.cos(t)])

        ld = ldarray(data, coords = dict(a=["sin", "cos"], t=t))

        ld_int = ld.interpolate(t=t_int)

        # interpolate each separately to make sure the result doesn't change
        ld_int_sin = ld.interpolate(t=t_int, a="sin")
        ld_int_cos = ld.interpolate(t=t_int, a="cos")

        # import matplotlib.pyplot as plt
        # plt.plot(t_int, ld_int.T)
        # plt.plot(t, ld.T, marker=".", linestyle="")

        # plt.plot(t_int, ld_int_sin.T)
        # plt.plot(t, ld[0].T, marker=".", linestyle="")
        # plt.plot(t_int, ld_int_cos.T)
        # plt.plot(t, ld[1].T, marker=".", linestyle="")

        np.testing.assert_array_almost_equal(ld_int.sel(a="sin"), np.sin(t_int), decimal=2)
        np.testing.assert_array_almost_equal(ld_int.sel(a="cos"), np.cos(t_int), decimal=2)
        np.testing.assert_array_almost_equal(ld_int_sin.squeeze(), np.sin(t_int), decimal=2)
        np.testing.assert_array_almost_equal(ld_int_cos.squeeze(), np.cos(t_int), decimal=2)

    def test_interpolation_complex(self):
        
        # avoid interpolating at the endpoints, it's close to the right value but hard to test exactly
        t = np.linspace(0, 2 * np.pi, 21)
        t_int = np.linspace(0.5, 5.5, 61)

        data = np.array([np.exp(1j * t), np.exp(0.5 * 1j * t)])

        ld = ldarray(data, coords = dict(a=["exp(t)", "exp(0.5t)"], t=t))

        ld_int = ld.interpolate(t=t_int)

        # import matplotlib.pyplot as plt
        # plt.plot(ld_int[0].real, ld_int[0].imag)
        # plt.plot(ld[0].real, ld[0].imag, marker=".", linestyle="")
        # plt.gca().set_aspect("equal")

        # plt.plot(ld_int[1].real, ld_int[1].imag)
        # plt.plot(ld[1].real, ld[1].imag, marker=".", linestyle="")

        np.testing.assert_array_almost_equal(ld_int.sel(a="exp(t)"), np.exp(1j * t_int), decimal=2)
        np.testing.assert_array_almost_equal(ld_int.sel(a="exp(0.5t)"), np.exp(0.5 * 1j * t_int), decimal=2)



    def test_save(self):
        
        coords = dict(a=['data1', 'data2'], b=np.arange(0, 20, 0.2))
        data = np.arange(200).reshape(2, 100)
        ld = ldarray(data, coords=coords)

        ld.save("ld_temp_file.npy")
        ld_load = ldarray.load("ld_temp_file.npy")

        np.testing.assert_array_equal(ld_load, ld)

        os.remove("ld_temp_file.npy")


if __name__ == '__main__':
    unittest.main()