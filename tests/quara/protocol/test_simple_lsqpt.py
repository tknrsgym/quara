import math
import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

import quara.protocol.lsqpt as lsqpt


def test_check_natural_number():
    parameter_name = "dim"
    # valid
    target = 1
    lsqpt.check_natural_number(target, parameter_name)

    # invalid
    target = 0

    with pytest.raises(ValueError):
        lsqpt.check_natural_number(target, parameter_name)


def test_load_matL0_invalid_rows():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/matL0_1qubit_X90.csv"
    matL0_np = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    dim = 1

    with pytest.raises(ValueError):
        _ = lsqpt.load_matL0(path, dim)


def test_load_matL0():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/matL0_1qubit_X90.csv"
    matL0_np = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    dim = int(math.sqrt(matL0_np.shape[0]))

    expected_data = np.array(
        [
            [
                0.000000000000000e00,
                0.000000000000000e00 + 7.853981633974483e-01j,
                0.000000000000000e00 - 7.853981633974483e-01j,
                0.000000000000000e00,
            ],
            [
                0.000000000000000e00 + 7.853981633974483e-01j,
                0.000000000000000e00,
                0.000000000000000e00,
                0.000000000000000e00 - 7.853981633974483e-01j,
            ],
            [
                0.000000000000000e00 - 7.853981633974483e-01j,
                0.000000000000000e00,
                0.000000000000000e00,
                0.000000000000000e00 + 7.853981633974483e-01j,
            ],
            [
                0.000000000000000e00,
                0.000000000000000e00 - 7.853981633974483e-01j,
                0.000000000000000e00 + 7.853981633974483e-01j,
                0.000000000000000e00,
            ],
        ]
    )

    actual_data = lsqpt.load_matL0(path, dim)
    assert np.array_equal(actual_data, expected_data)
