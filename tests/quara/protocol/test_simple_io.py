import os
from pathlib import Path

import numpy as np
import pytest

from quara.protocol import simple_io as s_io


def test_load_state_list_invalid_dim():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/tester_1qubit_state.csv"
    dim = 2 ** 1
    invalid_dim = 2 ** 1 + 1  # make invalid data
    state_np = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    num_state = state_np.shape[0] // dim

    with pytest.raises(ValueError):
        _ = s_io.load_state_list(path, invalid_dim, num_state)


def test_load_state_list_invalid_rows():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/tester_1qubit_state.csv"
    dim = 2 ** 1
    state_np = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    invalid_num_state = state_np.shape[0]  # make invalid data

    with pytest.raises(ValueError):
        _ = s_io.load_state_list(path, dim, invalid_num_state)


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

    actual_data = s_io.load_state_list(path, dim, num_state)
    assert np.array_equal(actual_data, expected_data)


def test_load_povm_list_invalid_dim():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/tester_1qubit_povm.csv"
    povm_np = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    invalid_dim = povm_np.shape[1] + 1  # make invalid data
    num_povm, num_outcome = 3, 2

    with pytest.raises(ValueError):
        _ = s_io.load_povm_list(path, invalid_dim, num_povm, num_outcome)


def test_load_povm_list_invalid_rows():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/tester_1qubit_povm.csv"
    povm_np = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    dim = povm_np.shape[1]
    num_povm, num_outcome = 3, 2
    invalid_num_povm, invalid_num_outcome = 4, 3

    with pytest.raises(ValueError):
        _ = s_io.load_povm_list(path, dim, invalid_num_povm, num_outcome)

    with pytest.raises(ValueError):
        _ = s_io.load_povm_list(path, dim, num_povm, invalid_num_outcome)


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

    actual_data = s_io.load_povm_list(path, dim, num_povm, num_outcome)

    assert np.array_equal(actual_data, expected_data)


def test_load_schedule_invalid_num_column():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/schedule_1qubit_invalid_num_column.csv"
    schedule_np = np.loadtxt(path, delimiter=",", dtype=np.uint16)
    num_state = np.unique(schedule_np[:, 0]).size
    num_povm = np.unique(schedule_np[:, 1]).size

    with pytest.raises(ValueError):
        _, _ = s_io.load_schedule(path, num_state, num_povm)


def test_load_schedule_invalid_rows():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/schedule_1qubit.csv"
    schedule_np = np.loadtxt(path, delimiter=",", dtype=np.uint16)
    num_state = np.unique(schedule_np[:, 0]).size
    num_povm = np.unique(schedule_np[:, 1]).size
    invalid_num_state = num_state - 1
    invalid_num_povm = num_povm - 1

    with pytest.raises(ValueError):
        _, _ = s_io.load_schedule(path, invalid_num_state, num_povm)

    with pytest.raises(ValueError):
        _, _ = s_io.load_schedule(path, num_state, invalid_num_povm)


def test_load_schedule_invalid_state_less():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/schedule_1qubit_invalid_state_less.csv"
    schedule_np = np.loadtxt(path, delimiter=",", dtype=np.uint16)
    num_state = np.unique(schedule_np[:, 0]).size
    num_povm = np.unique(schedule_np[:, 1]).size

    with pytest.raises(ValueError):
        _, _ = s_io.load_schedule(path, num_state, num_povm)


def test_load_schedule_invalid_povm_less():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/schedule_1qubit_invalid_povm_less.csv"
    schedule_np = np.loadtxt(path, delimiter=",", dtype=np.uint16)
    num_state = np.unique(schedule_np[:, 0]).size
    num_povm = np.unique(schedule_np[:, 1]).size

    with pytest.raises(ValueError):
        _, _ = s_io.load_schedule(path, num_state, num_povm)


def test_load_schedule():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/schedule_1qubit_start_from_0.csv"
    schedule_np = np.loadtxt(path, delimiter=",", dtype=np.uint16)
    num_state = np.unique(schedule_np[:, 0]).size
    num_povm = np.unique(schedule_np[:, 1]).size

    expected_num_schedule = schedule_np.shape[0]
    expected_data = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 0],
            [2, 1],
            [2, 2],
            [3, 0],
            [3, 1],
            [3, 2],
        ]
    )

    actual_num_schedule, actual_data = s_io.load_schedule(path, num_state, num_povm)
    assert np.array_equal(actual_num_schedule, expected_num_schedule)
    assert np.array_equal(actual_data, expected_data)


def test_load_empi_list_invalid_rows():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/listEmpiDist_2valued.csv"
    empi_list_np = np.loadtxt(path, delimiter=",", dtype=np.float64)
    num_schedule = empi_list_np.shape[0]
    num_outcome = empi_list_np.shape[1]
    invalid_num_schedule = num_schedule - 1
    invalid_num_outcome = num_outcome - 1

    with pytest.raises(ValueError):
        _ = s_io.load_empi_list(path, invalid_num_schedule, num_outcome)

    with pytest.raises(ValueError):
        _ = s_io.load_empi_list(path, num_schedule, invalid_num_outcome)


def test_load_empi_list_invalid_vlaue_negative():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/listEmpiDist_2valued_invalid_vlaue_negative.csv"
    empi_list_np = np.loadtxt(path, delimiter=",", dtype=np.float64)
    num_schedule = empi_list_np.shape[0]
    num_outcome = empi_list_np.shape[1]

    with pytest.raises(ValueError):
        _ = s_io.load_empi_list(path, num_schedule, num_outcome)


def test_load_empi_list_invalid_sum():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/listEmpiDist_2valued_invalid_sum.csv"
    empi_list_np = np.loadtxt(path, delimiter=",", dtype=np.float64)
    num_schedule = empi_list_np.shape[0]
    num_outcome = empi_list_np.shape[1]

    with pytest.raises(ValueError):
        _ = s_io.load_empi_list(path, num_schedule, num_outcome)


def test_load_empi_list():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/listEmpiDist_2valued.csv"
    empi_list_np = np.loadtxt(path, delimiter=",", dtype=np.float64)
    num_schedule = empi_list_np.shape[0]
    num_outcome = empi_list_np.shape[1]

    expected_data = np.array(
        [
            [5.043978292038679e-01, 4.956021707961321e-01],
            [5.142542524664510e-01, 4.857457475335489e-01],
            [9.997774258456620e-01, 2.225741543378976e-04],
            [4.956021707961321e-01, 5.043978292038679e-01],
            [4.857457475335489e-01, 5.142542524664510e-01],
            [2.225741543378976e-04, 9.997774258456620e-01],
            [9.778858482455595e-01, 2.211415175444042e-02],
            [3.529451941440284e-01, 6.470548058559715e-01],
            [4.999889870767086e-01, 5.000110129232912e-01],
            [6.469890306958841e-01, 3.530109693041158e-01],
            [9.776732150343842e-01, 2.232678496561569e-02],
            [4.850827450984023e-01, 5.149172549015976e-01],
        ]
    )

    actual_data = s_io.load_empi_list(path, num_schedule, num_outcome)
    assert np.array_equal(actual_data, expected_data)


def test_load_weight_list_invalid_num_outcome():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/weight_2valued_uniform.csv"
    state_np = np.loadtxt(path, delimiter=",", dtype=np.float64)
    num_outcome = state_np.shape[1]
    invalid_num_outcome = num_outcome + 1  # make invalid data
    num_schedule = state_np.shape[0] // num_outcome

    with pytest.raises(ValueError):
        _ = s_io.load_weight_list(path, num_schedule, invalid_num_outcome)


def test_load_weight_list_invalid_num_schedule():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/weight_2valued_uniform.csv"
    state_np = np.loadtxt(path, delimiter=",", dtype=np.float64)
    num_outcome = state_np.shape[1]
    num_schedule = state_np.shape[0] // num_outcome
    invalid_num_schedule = num_schedule + 1  # make invalid data

    with pytest.raises(ValueError):
        _ = s_io.load_weight_list(path, invalid_num_schedule, num_outcome)


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

    actual_data = s_io.load_weight_list(path, num_schedule, num_outcome)
    assert np.array_equal(actual_data, expected_data)
