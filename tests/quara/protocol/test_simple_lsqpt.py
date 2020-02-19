import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from quara.protocol import simple_io as s_io
import quara.utils.matrix_util as util


@pytest.mark.matlab_dependent
class TestSimplelsQptMatlab:
    def test_execute_1qubit(self):
        import quara.protocol.simple_lsqpt as s_lsqpt

        # load test data
        dim = 2 ** 1  # 2**qubits
        num_state = 4
        num_povm = 3
        num_outcome = 2

        test_root_dir = Path(os.path.dirname(__file__)).parent.parent
        data_dir = test_root_dir / "data"
        states = s_io.load_state_list(
            data_dir / "tester_1qubit_state.csv", dim=dim, num_state=num_state
        )
        povms = s_io.load_povm_list(
            data_dir / "tester_1qubit_povm.csv",
            dim=dim,
            num_povm=num_povm,
            num_outcome=num_outcome,
        )
        num_schedule, schedule = s_io.load_schedule(
            data_dir / "schedule_1qubit_start_from_0.csv",
            num_state=num_state,
            num_povm=num_povm,
        )
        empis = s_io.load_empi_list(
            data_dir / "listEmpiDist_2valued_k3.csv",
            num_schedule=num_schedule,
            num_outcome=num_outcome,
        )
        weights = s_io.load_weight_list(
            data_dir / "weight_2valued_uniform.csv",
            num_schedule=num_schedule,
            num_outcome=num_outcome,
        )
        k = 3
        matL0 = s_io.load_matL0(data_dir / "matL0_1qubit_X90.csv", dim=dim,)
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

        assert util.is_hermitian(actual_choi, atol=1e-15) == True
        assert util.is_positive_semidefinite(actual_choi, atol=1e-15) == True
        assert util.is_tp(actual_choi, 2, atol=1e-15) == True

        assert util.is_hermitian(actual_choi) == True
        assert util.is_positive_semidefinite(actual_choi) == True
        assert util.is_tp(actual_choi, 2) == True

    def test_execute_2qubit(self):
        import quara.protocol.simple_lsqpt as s_lsqpt

        # load test data
        dim = 2 ** 2  # 2**qubits
        num_state = 16
        num_povm = 9
        num_outcome = 4

        test_root_dir = Path(os.path.dirname(__file__)).parent.parent
        data_dir = test_root_dir / "data"
        states = s_io.load_state_list(
            data_dir / "tester_2qubit_state.csv", dim=dim, num_state=num_state
        )
        povms = s_io.load_povm_list(
            data_dir / "tester_2qubit_povm.csv",
            dim=dim,
            num_povm=num_povm,
            num_outcome=num_outcome,
        )
        num_schedule, schedule = s_io.load_schedule(
            data_dir / "schedule_2qubit_start_from_0.csv",
            num_state=num_state,
            num_povm=num_povm,
        )
        empis = s_io.load_empi_list(
            data_dir / "listEmpiDist_4valued_k3.csv",
            num_schedule=num_schedule,
            num_outcome=num_outcome,
        )
        weights = s_io.load_weight_list(
            data_dir / "weight_4valued_uniform.csv",
            num_schedule=num_schedule,
            num_outcome=num_outcome,
        )
        k = 3
        matL0 = s_io.load_matL0(data_dir / "matL0_2qubit_ZX90.csv", dim=dim,)
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

        assert util.is_hermitian(actual_choi, atol=1e-14) == True
        assert util.is_positive_semidefinite(actual_choi, atol=1e-14) == True
        assert util.is_tp(actual_choi, 4, atol=1e-13) == True

        assert util.is_hermitian(actual_choi) == True
        assert util.is_positive_semidefinite(actual_choi) == True
        assert util.is_tp(actual_choi, 4) == True
