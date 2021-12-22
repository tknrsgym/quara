import pickle
from pathlib import Path
from quara.protocol.qtomography.standard.standard_qmpt import StandardQmpt
import shutil
import os
import glob

import numpy as np
import numpy.testing as npt
import pytest

from quara.simulation import standard_qtomography_simulation as sim
from quara.simulation.standard_qtomography_simulation_flow import (
    re_estimate_test_settings,
)
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate
from quara.objects.mprocess import MProcess

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.qoperation_typical import generate_qoperation
from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
from quara.simulation.standard_qtomography_simulation import (
    StandardQTomographySimulationSetting,
)
from quara.objects.qoperation_typical import generate_qoperation_object
from quara.objects.composite_system_typical import generate_composite_system

from quara.simulation.standard_qtomography_simulation import (
    EstimatorTestSetting,
    NoiseSetting,
)

from tests.quara.simulation import random_test


def assert_equal_estimation_result(result_source, result_target):
    source_qoperations = result_source.estimated_qoperation_sequence
    target_qoperations = result_target.estimated_qoperation_sequence

    assert len(source_qoperations) == len(target_qoperations)
    for s, t in zip(source_qoperations, target_qoperations):
        assert_equal_qoperation(s, t)


def assert_equal_qoperation(source, target):
    assert type(source) == type(target)
    if type(source) == State:
        npt.assert_almost_equal(source.vec, target.vec, decimal=15)
    elif type(source) == Povm:
        assert len(source.vecs) == len(target.vecs)
        for vec_a, vec_b in zip(source.vecs, target.vecs):
            npt.assert_almost_equal(vec_a, vec_b, decimal=15)
    elif type(source) == Gate:
        npt.assert_almost_equal(source.hs, target.hs, decimal=15)
    else:
        raise NotImplementedError()


def make_test_data(test_data_dir):
    setting = {
        "mode": "qubit",
        "n_qubit": 1,
        "tomography_type": "state",
        "true_objects": ["z0", "z1"],
        "tester_names": [("povm", name) for name in ["x", "y", "z"]],
        "noise_method": "random_effective_lindbladian",
        "noise_para": {
            "lindbladian_base": "identity",
            "strength_h_part": 0.1,
            "strength_k_part": 0.1,
        },
        "n_sample": 2,
        "n_rep": 3,
        "num_data": [10, 100],
        "seed_qoperation": 888,
        "seed_data": 777,
        "output_root_dir": test_data_dir,
    }
    random_test.execute(**setting)

    return test_data_dir


@pytest.fixture(scope="class")
def execute_simulation_fixture():
    # setup
    test_data_dir = Path(os.path.dirname(__file__)) / "data/re_simulation_qst/source"
    make_test_data(test_data_dir)

    # execute test
    yield {"test_data_root_dir": test_data_dir}

    # remove
    shutil.rmtree(test_data_dir.parent)


@pytest.mark.usefixtures("execute_simulation_fixture")
class TestReEstimate:
    @pytest.mark.parametrize(
        ("case_file_name"),
        [f"case_{i}_result.pickle" for i in range(4)],
    )
    def test_re_estimate_sequence(self, execute_simulation_fixture, case_file_name):
        # Arrange
        input_root_dir = execute_simulation_fixture["test_data_root_dir"]
        result_path = Path(input_root_dir) / "0" / "0" / case_file_name
        test_setting_path = Path(input_root_dir) / "0" / "test_setting.pickle"

        with open(result_path, "rb") as f:
            source_result = pickle.load(f)

        with open(test_setting_path, "rb") as f:
            source_test_setting = pickle.load(f)

        # Act
        actual_results = sim.re_estimate_sequence(source_test_setting, source_result)

        # Assert
        expected_results = source_result.estimation_results
        assert len(actual_results) == len(expected_results)
        for actual, expected in zip(actual_results, expected_results):
            assert_equal_estimation_result(actual, expected)

    def test_re_estimate_sequence_from_path(self, execute_simulation_fixture):
        input_root_dir = execute_simulation_fixture["test_data_root_dir"]

        result_path = Path(input_root_dir) / "0" / "1" / "case_2_result.pickle"
        test_setting_path = Path(input_root_dir) / "1" / "test_setting.pickle"

        # Act
        actual_results = sim.re_estimate_sequence_from_path(
            test_setting_path, result_path
        )

        # Assert
        with open(result_path, "rb") as f:
            source_result = pickle.load(f)
        expected_results = source_result.estimation_results
        assert len(actual_results) == len(expected_results)
        for actual, expected in zip(actual_results, expected_results):
            assert_equal_estimation_result(actual, expected)

    def test_re_estimate_sequence_from_index(self, execute_simulation_fixture):
        # Arrange
        input_root_dir = execute_simulation_fixture["test_data_root_dir"]

        # Act
        actual_results = sim.re_estimate_sequence_from_index(input_root_dir, 1, 0, 3)

        # Assert
        result_path = Path(input_root_dir) / "1" / "0" / "case_3_result.pickle"
        with open(result_path, "rb") as f:
            source_result = pickle.load(f)
        # Assert
        expected_results = source_result.estimation_results
        assert len(actual_results) == len(expected_results)
        for actual, expected in zip(actual_results, expected_results):
            assert_equal_estimation_result(actual, expected)

    def test_re_estimate_flow(self, execute_simulation_fixture):
        # Arrange
        input_root_dir = execute_simulation_fixture["test_data_root_dir"]
        output_root_dir = input_root_dir.parent / "target_re_estimate_flow"

        # Act
        re_estimated_all_results = re_estimate_test_settings(
            input_root_dir=input_root_dir,
            output_root_dir=output_root_dir,
            pdf_mode="all",
        )

        # Assert
        source_paths = glob.glob(f"{str(input_root_dir)}/*/*/case_*_result.pickle")
        source_all_results = []
        for source_path in sorted(source_paths):
            with open(source_path, "rb") as f:
                source_result = pickle.load(f)
            source_all_results.append(source_result)

        assert len(source_all_results) == len(re_estimated_all_results)
        for expected_sim_result, actual_sim_result in zip(
            source_all_results, re_estimated_all_results
        ):
            assert len(expected_sim_result.estimation_results) == len(
                actual_sim_result.estimation_results
            )
            for expected, actual in zip(
                expected_sim_result.estimation_results,
                actual_sim_result.estimation_results,
            ):
                assert_equal_estimation_result(expected, actual)

    def test_re_estimate_with_exec_check(self, execute_simulation_fixture):
        # Arrange
        input_root_dir = execute_simulation_fixture["test_data_root_dir"]
        output_root_dir = input_root_dir.parent / "target_re_estimate_flow"

        # Case 1:
        # Act
        source_exec_sim_check = {
            "consistency": False,
        }
        re_estimated_all_results = re_estimate_test_settings(
            input_root_dir=input_root_dir,
            output_root_dir=output_root_dir,
            pdf_mode="all",
            exec_sim_check=source_exec_sim_check,
        )
        actual = set(
            r["name"] for r in re_estimated_all_results[0].check_result["results"]
        )

        # Assert
        expected = {
            "MSE of Empirical Distributions",
            "MSE of estimators",
            "Physicality Violation",
        }
        assert actual == expected

        # Case 2:
        # Act
        source_exec_sim_check = {
            "consistency": False,
            "mse_of_estimators": False,
            "mse_of_empi_dists": False,
            "physicality_violation": False,
        }
        re_estimated_all_results = re_estimate_test_settings(
            input_root_dir=input_root_dir,
            output_root_dir=output_root_dir,
            pdf_mode="all",
            exec_sim_check=source_exec_sim_check,
        )
        actual = set(
            r["name"] for r in re_estimated_all_results[0].check_result["results"]
        )

        # Assert
        expected = set()
        assert actual == expected

        # Case3: invalid input
        source_exec_sim_check = {
            "invalid": True,
        }

        with pytest.raises(KeyError):
            # ValueError: The key 'invalid' of the argument 'exec_check' is invalid. 'exec_check' can be used with the following keys: ['consistency', 'mse_of_estimators', 'mse_of_empi_dists', 'physicality_violation']
            _ = re_estimate_test_settings(
                input_root_dir=input_root_dir,
                output_root_dir=output_root_dir,
                pdf_mode="all",
                exec_sim_check=source_exec_sim_check,
            )


def is_same_dist(a_dist: tuple, b_dist: tuple):
    for a, b in zip(a_dist, b_dist):
        if a[0] != b[0]:
            return False
        if not np.allclose(a[1], b[1]):
            return False
    return True


def test_execute_simulation_with_seed_or_generator():
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    true_object = generate_qoperation(mode="state", name="y0", c_sys=c_sys)

    povm_x = generate_qoperation(mode="povm", name="x", c_sys=c_sys)
    povm_y = generate_qoperation(mode="povm", name="y", c_sys=c_sys)
    povm_z = generate_qoperation(mode="povm", name="z", c_sys=c_sys)
    tester_objects = [povm_x, povm_y, povm_z]

    sim_setting = StandardQTomographySimulationSetting(
        name="dummy name",
        true_object=true_object,
        tester_objects=tester_objects,
        estimator=LinearEstimator(),
        seed_data=777,
        n_rep=5,
        num_data=[10],
        schedules=None,
        eps_proj_physical=1e-13,
        eps_truncate_imaginary_part=1e-13,
    )

    qtomography = sim.generate_qtomography(sim_setting, para=True, init_with_seed=False)

    random_gen = np.random.Generator(np.random.MT19937(sim_setting.seed_data))

    # Execute
    sim_result = sim.execute_simulation(
        qtomography=qtomography,
        simulation_setting=sim_setting,
        seed_or_generator=random_gen,
    )
    # Assert
    # Verify that it is random.
    a_list = sim_result.empi_dists_sequences[0]

    for b_list in sim_result.empi_dists_sequences[1:]:
        assert is_same_dist(a_list[0], b_list[0]) is False

    # Execute
    random_gen = np.random.Generator(np.random.MT19937(sim_setting.seed_data))
    sim_result_1 = sim.execute_simulation(
        qtomography=qtomography,
        simulation_setting=sim_setting,
        seed_or_generator=random_gen,
    )

    # Assert
    # Confirmation of reproducibility
    expected = sim_result.empi_dists_sequences
    actual = sim_result_1.empi_dists_sequences
    for a, e in zip(actual, expected):
        assert is_same_dist(a[0], e[0])


def test_generate_qtomography_with_qmpt():
    c_sys = generate_composite_system(mode="qubit", num=1, ids_esys=[1])

    # Tester Object
    state_names = ["x0", "y0", "z0", "z1"]
    povm_names = ["x", "y", "z"]

    tester_states = [
        generate_qoperation_object(
            mode="state", object_name="state", name=name, c_sys=c_sys
        )
        for name in state_names
    ]
    tester_povms = [
        generate_qoperation_object(
            mode="povm", object_name="povm", name=name, c_sys=c_sys
        )
        for name in povm_names
    ]
    tester_objects = tester_states + tester_povms

    # True Object
    true_object_name = "x-type1"
    on_para_eq_constraint = True
    true_object = generate_qoperation_object(
        mode="mprocess", object_name="mprocess", name=true_object_name, c_sys=c_sys
    )
    if on_para_eq_constraint is False:
        true_object = MProcess(
            hss=true_object.hss, on_para_eq_constraint=False, c_sys=c_sys
        )

    sim_setting = StandardQTomographySimulationSetting(
        name="dummy name",
        true_object=true_object,
        tester_objects=tester_objects,
        estimator=LinearEstimator(),
        seed_data=888,
        n_rep=5,
        num_data=[10],
        schedules=None,
        eps_proj_physical=1e-13,
        eps_truncate_imaginary_part=1e-13,
    )

    actual = sim.generate_qtomography(
        sim_setting, para=true_object.on_para_eq_constraint
    )

    assert type(actual) == StandardQmpt
    assert actual.on_para_eq_constraint == on_para_eq_constraint


class TestEstimatorTestSetting:
    def _make_test_setting(self, generation_setting_is_physicality_required: bool):
        c_sys = generate_composite_system(mode="qubit", num=1)
        setting = {
            "mode": "qubit",
            "n_qubit": 1,
            "tomography_type": "state",
            "true_objects": ["z0"],
            "tester_names": [("povm", name) for name in ["x", "y", "z"]],
            "noise_method": "random_effective_lindbladian",
            "noise_para": {
                "lindbladian_base": "identity",
                "strength_h_part": 0.1,
                "strength_k_part": 0.1,
            },
            "n_sample": 1,
            "n_rep": 1,
            "num_data": [1000, 10000],
            "seed_qoperation": 888,
            "seed_data": 777,
        }

        true_object_noise_setting = NoiseSetting(
            qoperation_base=(setting["tomography_type"], setting["true_objects"][0]),
            method=setting["noise_method"],
            para=setting["noise_para"],
            ids=None,
        )

        tester_object_noise_settings = [
            NoiseSetting(
                qoperation_base=name,
                method=setting["noise_method"],
                para=setting["noise_para"],
                ids=None,
            )
            for name in setting["tester_names"]
        ]

        if generation_setting_is_physicality_required is None:
            test_setting = EstimatorTestSetting(
                true_object=true_object_noise_setting,
                tester_objects=tester_object_noise_settings,
                seed_qoperation=setting["seed_qoperation"],
                seed_data=setting["seed_data"],
                n_sample=setting["n_sample"],
                n_rep=setting["n_rep"],
                num_data=setting["num_data"],
                schedules="all",
                case_names=["Linear (True)"],
                estimators=[LinearEstimator()],
                eps_proj_physical_list=[1e-5],
                eps_truncate_imaginary_part_list=[1e-5],
                algo_list=[(None, None)],
                loss_list=[(None, None)],
                parametrizations=[True],
                c_sys=c_sys,
            )
        else:
            test_setting = EstimatorTestSetting(
                true_object=true_object_noise_setting,
                tester_objects=tester_object_noise_settings,
                seed_qoperation=setting["seed_qoperation"],
                seed_data=setting["seed_data"],
                n_sample=setting["n_sample"],
                n_rep=setting["n_rep"],
                num_data=setting["num_data"],
                schedules="all",
                case_names=["Linear (True)"],
                estimators=[LinearEstimator()],
                eps_proj_physical_list=[1e-5],
                eps_truncate_imaginary_part_list=[1e-5],
                algo_list=[(None, None)],
                loss_list=[(None, None)],
                parametrizations=[True],
                c_sys=c_sys,
                generation_setting_is_physicality_required=generation_setting_is_physicality_required,
            )

        return test_setting

    def test_to_generation_settings_is_physilicaty_required(self):
        # Case 1:
        # Arrange
        source_setting = self._make_test_setting(True)
        # Act
        actual = source_setting.to_generation_settings()

        # Assert
        expected = True
        assert actual.true_setting.qoperation_base.is_physicality_required is expected

        for a in actual.tester_settings:
            assert a.qoperation_base.is_physicality_required is expected

        # Case 2:
        # Arrange
        source_setting = self._make_test_setting(False)
        # Act
        actual = source_setting.to_generation_settings()
        expected = False

        # Assert
        assert actual.true_setting.qoperation_base.is_physicality_required is expected

        for a in actual.tester_settings:
            assert a.qoperation_base.is_physicality_required is expected

        # Case 1:
        # Arrange
        source_setting = self._make_test_setting(None)
        # Act
        actual = source_setting.to_generation_settings()

        # Assert
        expected = True
        assert actual.true_setting.qoperation_base.is_physicality_required is expected

        for a in actual.tester_settings:
            assert a.qoperation_base.is_physicality_required is expected
