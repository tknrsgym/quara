import math
import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

import quara.protocol.simple_lsqpt as s_lsqpt


def test_check_natural_number():
    parameter_name = "dim"
    # valid
    target = 1
    s_lsqpt.check_natural_number(target, parameter_name)

    # invalid
    target = 0

    with pytest.raises(ValueError):
        s_lsqpt.check_natural_number(target, parameter_name)


def test_load_matL0_invalid_rows():
    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    path = test_root_dir / "data/matL0_1qubit_X90.csv"
    matL0_np = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    dim = 1

    with pytest.raises(ValueError):
        _ = s_lsqpt.load_matL0(path, dim)


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

    actual_data = s_lsqpt.load_matL0(path, dim)
    assert np.array_equal(actual_data, expected_data)


@pytest.mark.call_matlab
def test_execute_1qubit():
    # load test data
    dim = 2 ** 1  # 2**qubits
    num_state = 4
    num_povm = 3
    num_outcome = 2

    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    data_dir = test_root_dir / "data"
    states = s_lsqpt.load_state_list(
        data_dir / "tester_1qubit_state.csv", dim=dim, num_state=num_state
    )
    povms = s_lsqpt.load_povm_list(
        data_dir / "tester_1qubit_povm.csv",
        dim=dim,
        num_povm=num_povm,
        num_outcome=num_outcome,
    )
    num_schedule, schedule = s_lsqpt.load_schedule(
        data_dir / "schedule_1qubit_start_from_0.csv",
        num_state=num_state,
        num_povm=num_povm,
    )
    empis = s_lsqpt.load_empi_list(
        data_dir / "listEmpiDist_2valued_k3.csv",
        num_schedule=num_schedule,
        num_outcome=num_outcome,
    )
    weights = s_lsqpt.load_weight_list(
        data_dir / "weight_2valued_uniform.csv",
        num_schedule=num_schedule,
        num_outcome=num_outcome,
    )
    k = 3
    matL0 = s_lsqpt.load_matL0(data_dir / "matL0_1qubit_X90.csv", dim=dim,)
    eps_logmat = 10e-10

    # Expected data (MATLAB output)
    # output of test_qpt_1qubit.m
    path = Path(os.path.dirname(__file__)) / "data/expected_simple_lsqpt_1qubit.csv"
    expected_choi = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    expected_obj_value = 2.83106871279414917808e-15

    # Confirm that it is the same as the output in MATLAB.\
    actual_data = s_lsqpt.execute(
        dim=dim,
        state_list=states,
        povm_list=povms,
        schedule=schedule,
        weight_list=weights,
        empi_list=empis,
        k=3,
        matL0=matL0,
        eps_logmat=eps_logmat,
    )
    actual_choi, actual_obj_value = actual_data

    # NOTICE: the decimal that tests can pass depends on the execution machine
    npt.assert_almost_equal(actual_choi, expected_choi, decimal=8)
    npt.assert_almost_equal(actual_obj_value, expected_obj_value, decimal=14)


@pytest.mark.call_matlab
def test_execute_2qubit():
    # load test data
    dim = 2 ** 2  # 2**qubits
    num_state = 16
    num_povm = 9
    num_outcome = 4

    test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    data_dir = test_root_dir / "data"
    states = s_lsqpt.load_state_list(
        data_dir / "tester_2qubit_state.csv", dim=dim, num_state=num_state
    )
    povms = s_lsqpt.load_povm_list(
        data_dir / "tester_2qubit_povm.csv",
        dim=dim,
        num_povm=num_povm,
        num_outcome=num_outcome,
    )
    num_schedule, schedule = s_lsqpt.load_schedule(
        data_dir / "schedule_2qubit_start_from_0.csv",
        num_state=num_state,
        num_povm=num_povm,
    )
    empis = s_lsqpt.load_empi_list(
        data_dir / "listEmpiDist_4valued_k3.csv",
        num_schedule=num_schedule,
        num_outcome=num_outcome,
    )
    weights = s_lsqpt.load_weight_list(
        data_dir / "weight_4valued_uniform.csv",
        num_schedule=num_schedule,
        num_outcome=num_outcome,
    )
    k = 3
    matL0 = s_lsqpt.load_matL0(data_dir / "matL0_2qubit_ZX90.csv", dim=dim,)
    eps_logmat = 10e-10

    # Expected data (MATLAB output)
    # output of test_qpt_1qubit.m
    path = Path(os.path.dirname(__file__)) / "data/expected_simple_lsqpt_2qubit.csv"
    expected_choi = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    expected_obj_value = 6.51636522519538630149e-11

    # Confirm that it is the same as the output in MATLAB.\
    actual_data = s_lsqpt.execute(
        dim=dim,
        state_list=states,
        povm_list=povms,
        schedule=schedule,
        weight_list=weights,
        empi_list=empis,
        k=3,
        matL0=matL0,
        eps_logmat=eps_logmat,
    )
    actual_choi, actual_obj_value = actual_data

    # NOTICE: the decimal that tests can pass depends on the execution machine
    npt.assert_almost_equal(actual_choi, expected_choi, decimal=10)
    npt.assert_almost_equal(actual_obj_value, expected_obj_value, decimal=12)
