import os
from pathlib import Path

import numpy as np
import pytest

import quara.protocol.simple_qpt as s_qpt


def test_check_file_extension():
    # valid
    valid_path = "hoge.csv"
    s_qpt.check_file_extension(valid_path)

    # invalid
    invalid_path = "hoge.tsv"

    with pytest.raises(ValueError):
        s_qpt.check_file_extension(invalid_path)


def test_load_state_list_invalid_dim():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/tester_1qubit_state.csv"
    dim = 2 ** 1
    invalid_dim = 2 ** 1 + 1  # make invalid data
    state_np = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    num_state = state_np.shape[0] // dim

    with pytest.raises(ValueError):
        _ = s_qpt.load_state_list(path, invalid_dim, num_state)


def test_load_state_list_invalid_rows():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/tester_1qubit_state.csv"
    dim = 2 ** 1
    state_np = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    invalid_num_state = state_np.shape[0]  # make invalid data

    with pytest.raises(ValueError):
        _ = s_qpt.load_state_list(path, dim, invalid_num_state)


def test_load_state_list():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/tester_1qubit_state.csv"
    dim = 2 ** 1
    state_np = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    num_state = state_np.shape[0] // dim

    expected_data = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5j, 0.5j, 0.5],
        ]
    )

    actual_data = s_qpt.load_state_list(path, dim, num_state)
    assert np.array_equal(actual_data, expected_data)
