import os

from pathlib import Path
import numpy as np
import matlab

from quara.utils.matlab_utils import to_np


def test_to_matlab_real_number():
    np_data = np.array([[1, 2, 3], [4, 5, 6]])
    matlab_data = matlab.int64(np_data.tolist())

    actual_data = to_np(matlab_data)
    expected_data = np_data
    assert np.array_equal(actual_data, expected_data)

    np_data = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])
    matlab_data = matlab.double(np_data.tolist())

    actual_data = to_np(matlab_data)
    expected_data = np_data
    assert np.array_equal(actual_data, expected_data)


def test_to_matlab_complex_number():
    this_pypath = os.path.abspath(__file__)
    test_data_path = (
        Path(this_pypath).parent.parent.parent / "data/tester_1qubit_state.csv"
    )
    np_data = np.loadtxt(test_data_path, delimiter=",", dtype=np.complex128)
    matlab_data = matlab.double(np_data.tolist(), is_complex=True)

    actual_data = to_np(matlab_data)
    expected_data = np_data
    assert np.array_equal(actual_data, expected_data)


def test_to_matlab_3d():
    np_data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    matlab_data = matlab.int64(np_data.tolist())

    actual_data = to_np(matlab_data)
    expected_data = np_data
    assert np.array_equal(actual_data, expected_data)
