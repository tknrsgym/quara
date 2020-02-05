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
    lsqpt.load_matL0(target, parameter_name)

    # invalid
    target = 0

    with pytest.raises(ValueError):
        lsqpt.load_matL0(target, parameter_name)


def test_load_matL0_invalid_rows():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/matL0_1qubit.csv"
    matL0_np = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    dim = 1

    with pytest.raises(ValueError):
        _ = lsqpt.load_matL0(path, dim)


def test_load_matL0():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/matL0_1qubit.csv"
    matL0_np = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    dim = int(math.sqrt(matL0_np.shape[0]))

    expected_data = np.array(
        [
            [1.1 + 0.1j, 1.2 + 0.1j, 1.3 + 0.1j, 1.4 + 0.1j],
            [2.1 + 0.2j, 2.2 + 0.2j, 2.3 + 0.2j, 2.4 + 0.2j],
            [3.1 + 0.3j, 3.2 + 0.3j, 3.3 + 0.3j, 3.4 + 0.3j],
            [4.1 + 0.4j, 4.2 + 0.4j, 4.3 + 0.4j, 4.4 + 0.4j],
        ]
    )

    actual_data = lsqpt.load_matL0(path, dim)
    assert np.array_equal(actual_data, expected_data)
