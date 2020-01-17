import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
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


def test_load_povm_list_invalid_dim():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/tester_1qubit_povm.csv"
    povm_np = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    invalid_dim = povm_np.shape[1] + 1  # make invalid data
    num_povm, num_outcome = 3, 2

    with pytest.raises(ValueError):
        _ = s_qpt.load_povm_list(path, invalid_dim, num_povm, num_outcome)


def test_load_povm_list_invalid_rows():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/tester_1qubit_povm.csv"
    povm_np = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    dim = povm_np.shape[1]
    num_povm, num_outcome = 3, 2
    invalid_num_povm, invalid_num_outcome = 4, 3

    with pytest.raises(ValueError):
        _ = s_qpt.load_povm_list(path, dim, invalid_num_povm, num_outcome)

    with pytest.raises(ValueError):
        _ = s_qpt.load_povm_list(path, dim, num_povm, invalid_num_outcome)


def test_load_povm_list():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/tester_1qubit_povm.csv"
    povm_np = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    dim = povm_np.shape[1]
    num_povm, num_outcome = 3, 2

    expected_data = np.array(
        [
            [[0.5, 0.5, 0.5, 0.5], [0.5, -0.5, -0.5, 0.5]],
            [[0.5, -0.5j, 0.5j, 0.5 + 0], [0.5, 0.5j, -0.5j, 0.5]],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ]
    )

    actual_data = s_qpt.load_povm_list(path, dim, num_povm, num_outcome)

    assert np.array_equal(actual_data, expected_data)


def test_load_weight_list_invalid_num_outcome():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/weight_2valued_uniform.csv"
    state_np = np.loadtxt(path, delimiter=",", dtype=np.float64)
    num_outcome = state_np.shape[1]
    invalid_num_outcome = num_outcome + 1  # make invalid data
    num_schedule = state_np.shape[0] // num_outcome

    with pytest.raises(ValueError):
        _ = s_qpt.load_weight_list(path, num_schedule, invalid_num_outcome)


def test_load_weight_list_invalid_num_schedule():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/weight_2valued_uniform.csv"
    state_np = np.loadtxt(path, delimiter=",", dtype=np.float64)
    num_outcome = state_np.shape[1]
    num_schedule = state_np.shape[0] // num_outcome
    invalid_num_schedule = num_schedule + 1  # make invalid data

    with pytest.raises(ValueError):
        _ = s_qpt.load_weight_list(path, invalid_num_schedule, num_outcome)


def test_load_weight_list():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/weight_2valued_uniform.csv"
    state_np = np.loadtxt(path, delimiter=",", dtype=np.float64)
    num_outcome = state_np.shape[1]
    num_schedule = state_np.shape[0] // num_outcome

    expected_data = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0],],
            [[1.0, 0.0], [0.0, 1.0],],
            [[1.0, 0.0], [0.0, 1.0],],
            [[1.0, 0.0], [0.0, 1.0],],
            [[1.0, 0.0], [0.0, 1.0],],
            [[1.0, 0.0], [0.0, 1.0],],
            [[1.0, 0.0], [0.0, 1.0],],
            [[1.0, 0.0], [0.0, 1.0],],
            [[1.0, 0.0], [0.0, 1.0],],
            [[1.0, 0.0], [0.0, 1.0],],
            [[1.0, 0.0], [0.0, 1.0],],
            [[1.0, 0.0], [0.0, 1.0],],
        ]
    )

    actual_data = s_qpt.load_weight_list(path, num_schedule, num_outcome)
    assert np.array_equal(actual_data, expected_data)


@pytest.mark.call_matlab
def test_execute():
    # load test data
    dim = 2 ** 1  # 2**qubits
    num_state = 4
    num_povm = 3
    num_outcome = 2

    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    data_dir = test_root_dir / "data"
    states = s_qpt.load_state_list(
        data_dir / "tester_1qubit_state.csv", dim=dim, num_state=num_state
    )
    povms = s_qpt.load_povm_list(
        data_dir / "tester_1qubit_povm.csv",
        dim=dim,
        num_povm=num_povm,
        num_outcome=num_outcome,
    )
    num_schedule, schedule = s_qpt.load_schedule(
        data_dir / "schedule_1qubit_start_from_0.csv",
        num_state=num_state,
        num_povm=num_povm,
    )
    empis = s_qpt.load_empi_list(
        data_dir / "listEmpiDist_2valued.csv",
        num_schedule=num_schedule,
        num_outcome=num_outcome,
    )
    weights = s_qpt.load_weight_list(
        data_dir / "weight_2valued_uniform.csv",
        num_schedule=num_schedule,
        num_outcome=num_outcome,
    )

    # Expected data (MATLAB output)
    path = Path(os.path.dirname(__file__)) / "data/expected_simple_qpt_1qubit.csv"
    expected_choi = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    expected_obj_value = 5.484953853954200e-13

    # Confirm that it is the same as the output in MATLAB.\
    actual_data = s_qpt.execute(
        dim=dim,
        state_list=states,
        povm_list=povms,
        schedule=schedule,
        weight_list=weights,
        empi_list=empis,
    )
    actual_choi, actual_obj_value = actual_data

    npt.assert_almost_equal(actual_choi, expected_choi, decimal=15)
    npt.assert_almost_equal(actual_obj_value, expected_obj_value)

