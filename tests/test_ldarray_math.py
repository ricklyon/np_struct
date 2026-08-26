import unittest
from np_struct import ldarray, Coords
import numpy as np
from numpy import testing as npt
import datetime as dt
from dateutil import relativedelta as rdt    
import os

np.printoptions(suppress=True)


def assert_coords_equal(c1: dict, c2: dict):
    """
    Check that two coordinates are identical
    """
    assert tuple(c1.keys()) == tuple(c2.keys()), f"Coord keys are different: {c1.keys()} vs {c2.keys()}"

    for k in c1.keys():
        np.testing.assert_array_almost_equal(c1[k], c2[k])


class TestLdArrayMath(unittest.TestCase):

    def test_expand_ends(self):
        # verify dimensions are expanded at the ends of the array during math operations

        theta = np.arange(90, -100, -10)

        # 2x20 array
        data1 = np.array([1 * np.exp(1j * np.deg2rad(theta)), 2 * np.exp(1j * np.deg2rad(theta))])
        # length 20 vector
        data2 = np.array([1j, 1])
        # length 20 vector
        data3 = np.exp(2j * np.deg2rad(theta))

        ld_data1 = ldarray(data1, coords=dict(a=[1, 2], theta=theta))
        ld_data2 = ldarray(data2, coords=dict(a=[1, 2]))
        ld_data3 = ldarray(data3, coords=dict(theta=theta))

        # test expanding dimension at the end
        result = ld_data1 * ld_data2
        np.testing.assert_array_almost_equal(data1 * data2[..., None], result)
        assert_coords_equal(result.coords, ld_data1.coords)
        # test expanding dimension at the beginning'
        result = ld_data1 * ld_data3
        np.testing.assert_array_almost_equal(data1 * data3, result)
        assert_coords_equal(result.coords, ld_data1.coords)


    def test_expand_middle(self):
        # verify dimensions are expanded in the middle of the array during math operations

        data1 = np.arange(24).reshape(3, 2, 4)
        data2 = np.arange(12).reshape(3, 4)

        ld_data1 = ldarray(data1, coords=dict(a=[1, 2, 3], b=[1, 2], c=[4, 5, 6, 7]))
        ld_data2 = ldarray(data2, coords=dict(a=[1, 2, 3], c=[4, 5, 6, 7]))

        result = ld_data1 * ld_data2[:, None, :]
        np.testing.assert_array_almost_equal(data1 * data2[:, None, :], result)
        assert_coords_equal(result.coords, ld_data1.coords)

    def test_broadcast(self):
        # verify two arrays with different dimensions can be broadcast together correctly

        data1 = np.arange(5)
        data2 = np.arange(5)

        ld_data1 = ldarray(data1, coords=dict(b=[1, 2, 3, 4, 5]))
        ld_data2 = ldarray(data2, coords=dict(c=[4, 5, 6, 7, 8]))

        result = ld_data1 * ld_data2
        np.testing.assert_array_equal(result, np.outer(data1, data2))

        assert_coords_equal(result.coords, dict(b=[1, 2, 3, 4, 5], c=[4, 5, 6, 7, 8]))

    def test_sum(self):

        data1 = np.arange(24).reshape(3, 2, 4)
        ld_data1 = ldarray(data1, coords=dict(a=[1, 2, 3], b=[1, 2], c=[4, 5, 6, 7]))

        np.testing.assert_array_equal(np.sum(data1, axis=1), np.sum(ld_data1, axis=1))
        np.testing.assert_array_equal(np.sum(data1, axis=1), np.sum(ld_data1, axis="b"))
        np.testing.assert_array_equal(np.sum(data1, axis=(0, 1)), np.sum(ld_data1, axis=("a", "b")))
        np.testing.assert_array_equal(np.sum(data1, axis=(0, -1)), np.sum(ld_data1, axis=("a", -1)))

    def test_mismatching_coords(self):
        # check that numpy array is returned instead of ldarray if coords do not match in the two math operands

        data1 = np.arange(5)
        data2 = np.arange(5)

        ld_data1 = ldarray(data1, coords=dict(c=[1, 2, 3, 4, 5]))
        ld_data2 = ldarray(data2, coords=dict(c=[4, 5, 6, 7, 8]))

        result = ld_data2 * ld_data1

        np.testing.assert_array_equal(result, data1 * data2)
        self.assertTrue(isinstance(result, np.ndarray))

    def test_fft(self):

        data = np.arange(80).reshape(8, 10)
        data_ld = ldarray(data, coords=dict(a=np.arange(8), b=np.arange(10)))

        result = np.fft.fft(data_ld, axis="b")

        np.testing.assert_array_equal(data, data_ld)
        self.assertEqual(data_ld.coords, result.coords)

    def test_average(self):

        data = np.arange(80).reshape(8, 5, 2)
        data_ld = ldarray(data, coords=dict(a=np.arange(8), b=np.arange(5), c=["m", "n"]))

        result = np.average(data_ld, axis="b")

        np.testing.assert_array_equal(np.average(data.view(np.ndarray), axis=1), result)
        np.testing.assert_array_equal(data_ld.a, result.a)
        np.testing.assert_array_equal(data_ld.c, result.c)

if __name__ == '__main__':
    unittest.main()