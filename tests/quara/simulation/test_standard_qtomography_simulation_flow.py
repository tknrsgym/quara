from pathlib import Path
from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
import shutil
import os
from collections import defaultdict
import itertools

import numpy as np
import pytest

from quara.simulation.standard_qtomography_simulation import (
    SimulationResult,
    EstimatorTestSetting,
    NoiseSetting,
)
from quara.simulation.standard_qtomography_simulation_flow import (
    _print_summary,
    execute_simulation_test_settings,
)

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects import matrix_basis

from tests.quara.simulation import random_test


def test_print_summary(capfd):
    # Arrange
    check_result = {
        "name": "dummy case 0",
        "total_result": True,
        "results": [
            {"name": "MSE of Empirical Distributions", "result": True, "detail": None},
            {
                "name": "Consistency",
                "result": True,
                "detail": {"possibly_ok": True, "to_be_checked": False},
            },
            {"name": "MSE of estimators", "result": True, "detail": None},
            {"name": "Physicality Violation", "result": True, "detail": None},
        ],
    }
    result_0 = SimulationResult(
        result_index=None,
        simulation_setting=None,
        estimation_results=None,
        qtomography=None,
        empi_dists_sequences=None,
        check_result=check_result,
    )

    check_result = {
        "name": "dummy case 1",
        "total_result": False,
        "results": [
            {"name": "MSE of Empirical Distributions", "result": True, "detail": None},
            {
                "name": "Consistency",
                "result": False,
                "detail": {"possibly_ok": False, "to_be_checked": False},
            },
            {"name": "MSE of estimators", "result": True, "detail": None},
            {"name": "Physicality Violation", "result": False, "detail": None},
        ],
    }
    result_1 = SimulationResult(
        result_index=None,
        simulation_setting=None,
        estimation_results=None,
        qtomography=None,
        empi_dists_sequences=None,
        check_result=check_result,
    )

    check_result = {
        "name": "dummy case 2",
        "total_result": False,
        "results": [
            {"name": "MSE of Empirical Distributions", "result": True, "detail": None},
            {
                "name": "Consistency",
                "result": False,
                "detail": {"possibly_ok": False, "to_be_checked": True},
            },
            {"name": "MSE of estimators", "result": False, "detail": None},
            {"name": "Physicality Violation", "result": False, "detail": None},
        ],
    }
    result_2 = SimulationResult(
        result_index=None,
        simulation_setting=None,
        estimation_results=None,
        qtomography=None,
        empi_dists_sequences=None,
        check_result=check_result,
    )

    check_result = {
        "name": "dummy case 3",
        "total_result": False,
        "results": [
            {"name": "MSE of Empirical Distributions", "result": False, "detail": None},
            {
                "name": "Consistency",
                "result": True,
                "detail": {"possibly_ok": True, "to_be_checked": False},
            },
            {"name": "MSE of estimators", "result": False, "detail": None},
            {"name": "Physicality Violation", "result": False, "detail": None},
        ],
    }
    result_3 = SimulationResult(
        result_index=None,
        simulation_setting=None,
        estimation_results=None,
        qtomography=None,
        empi_dists_sequences=None,
        check_result=check_result,
    )
    dummy_results = [result_0, result_1, result_2, result_3]

    # Act
    _print_summary(results=dummy_results, elapsed_time=500)
    out, _ = capfd.readouterr()

    # Assert
    start_red = "\033[31m"
    start_green = "\033[32m"
    start_yellow = "\033[33m"
    end_color = "\033[0m"
    expected = f"""{start_yellow}=============== Summary ================={end_color}
MSE of Empirical Distributions:
{start_green}OK: 3 cases{end_color}, {start_red}NG: 1 cases{end_color}

Consistency:
{start_green}OK: 2 cases{end_color}, {start_red}NG: 2 cases{end_color}
You need to check report: 1 cases

MSE of estimators:
{start_green}OK: 2 cases{end_color}, {start_red}NG: 2 cases{end_color}

Physicality Violation:
{start_green}OK: 1 cases{end_color}, {start_red}NG: 3 cases{end_color}

Time:
500.0s (0:08:20)
{start_yellow}========================================={end_color}\n"""

    assert out == expected


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
def tmp_out_dir_fixture():
    # setup
    tmp_out_dir = Path(os.path.dirname(__file__)) / "data/tmp_out_dir"

    # execute test
    yield {"tmp_out_dir": tmp_out_dir}

    # remove
    shutil.rmtree(tmp_out_dir)


@pytest.mark.usefixtures("tmp_out_dir_fixture")
class TestSimulationCheckOnOff:
    def test_execute_simulation_flow_with_exec_check(self, tmp_out_dir_fixture):
        tmp_out_dir = tmp_out_dir_fixture["tmp_out_dir"]

        noise_para = {"error_rate": 0.01}
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        test_setting = EstimatorTestSetting(
            true_object=NoiseSetting(
                qoperation_base=("state", "x0"), method="depolarized", para=noise_para
            ),
            tester_objects=[
                NoiseSetting(
                    qoperation_base=("povm", "x"), method="depolarized", para=noise_para
                ),
                NoiseSetting(
                    qoperation_base=("povm", "y"), method="depolarized", para=noise_para
                ),
                NoiseSetting(
                    qoperation_base=("povm", "z"), method="depolarized", para=noise_para
                ),
            ],
            seed_qoperation=888,
            seed_data=777,
            n_sample=1,
            n_rep=3,
            num_data=[5],
            schedules="all",
            case_names=["dummy_name"],
            estimators=[LinearEstimator()],
            eps_proj_physical_list=[1e-5],
            algo_list=[(None, None)],
            loss_list=[(None, None)],
            parametrizations=[True],
            c_sys=c_sys,
        )
        test_settings = [test_setting]

        # Case 1:
        # Act
        all_results = execute_simulation_test_settings(
            test_settings, tmp_out_dir, pdf_mode="all"
        )
        actual = set(r["name"] for r in all_results[0].check_result["results"])

        # Assert
        expected = {
            "MSE of Empirical Distributions",
            "Consistency",
            "MSE of estimators",
            "Physicality Violation",
        }
        assert actual == expected

        # Case 2:
        # Act
        source_exec_check = {
            "consistency": False,
            "mse_of_estimators": True,
            "mse_of_empi_dists": False,
            "physicality_violation": True,
        }
        all_results = execute_simulation_test_settings(
            test_settings,
            tmp_out_dir,
            pdf_mode="all",
            exec_sim_check=source_exec_check,
        )
        actual = set(r["name"] for r in all_results[0].check_result["results"])
        # Assert
        expected = {"MSE of estimators", "Physicality Violation"}
        assert actual == expected

        # Case3:
        source_exec_check = {
            "consistency": False,
            "mse_of_estimators": False,
            "mse_of_empi_dists": False,
            "physicality_violation": False,
        }
        all_results = execute_simulation_test_settings(
            test_settings,
            tmp_out_dir,
            pdf_mode="all",
            exec_sim_check=source_exec_check,
        )
        # Act
        actual = set(r["name"] for r in all_results[0].check_result["results"])
        # Assert
        expected = set()
        assert actual == expected

        # Case4: invalid input
        source_exec_check = {
            "invalid": True,
        }

        with pytest.raises(KeyError):
            # ValueError: The key 'invalid' of the argument 'exec_check' is invalid. 'exec_check' can be used with the following keys: ['consistency', 'mse_of_estimators', 'mse_of_empi_dists', 'physicality_violation']
            _ = execute_simulation_test_settings(
                test_settings,
                tmp_out_dir,
                pdf_mode="all",
                exec_sim_check=source_exec_check,
            )


@pytest.mark.usefixtures("tmp_out_dir_fixture")
class TestRandomStateQoperation:
    def convert_results_to_dict(self, sim_results):
        sim_results_dict = defaultdict(
            lambda: defaultdict(lambda: dict(true_object=[], tester_objects=[]))
        )

        for sim_result in sim_results:
            test_setting_index = sim_result.result_index["test_setting_index"]
            sample_index = sim_result.result_index["sample_index"]
            sim_results_dict[test_setting_index][sample_index]["true_object"].append(
                sim_result.simulation_setting.true_object
            )
            sim_results_dict[test_setting_index][sample_index]["tester_objects"].append(
                sim_result.simulation_setting.tester_objects
            )
        return sim_results_dict

    def is_same_vecs(self, a_vecs, b_vecs):
        for a, b in zip(a_vecs, b_vecs):
            if not np.allclose(a, b):
                return False
        return True

    def test_random_state_qoperation(self, tmp_out_dir_fixture):
        tmp_out_dir = tmp_out_dir_fixture["tmp_out_dir"]
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
            "n_sample": 3,
            "n_rep": 3,
            "num_data": [10, 50],
            "seed_qoperation": 888,
            "seed_data": 777,
            "output_root_dir": Path(tmp_out_dir)
            / random_test.get_current_time_string(),
        }

        all_results = random_test.execute(**setting)
        sim_results_dict_0 = self.convert_results_to_dict(all_results)

        # (1) Check if true object is randomly generated between samples.
        sample_true_objects_0 = []
        for _, sample_unit in sim_results_dict_0[0].items():
            # Check by looking at the first case only.
            sample_true_object = sample_unit["true_object"][0]  # case_index=0
            sample_true_objects_0.append(sample_true_object)

        # Assert
        # Check all combinations to see if they are random values.
        for a, b in itertools.combinations(sample_true_objects_0, 2):
            assert not np.allclose(a.vec, b.vec)

        # (2) Check if tester objects are randomly generated between samples.
        sample_tester_objects_0 = {0: [], 1: [], 2: []}
        for _, v in sim_results_dict_0[0].items():
            sample_objects = v["tester_objects"][0]  # case_index = 0
            for vecs_index, sample_object in enumerate(sample_objects):
                sample_tester_objects_0[vecs_index].append(sample_object)

        # Assert
        # Check all combinations to see if they are random values.
        for a, b in itertools.combinations(sample_tester_objects_0[0], 2):
            assert self.is_same_vecs(a.vecs, b.vecs) is False

        # (3) Make sure that the true object is reproducible.
        # re-estimate
        all_results = random_test.execute(**setting)
        sim_results_dict_1 = self.convert_results_to_dict(all_results)

        sample_true_objects_1 = []
        for _, sample_unit in sim_results_dict_1[0].items():
            # Check by looking at the first case only.
            sample_true_object = sample_unit["true_object"][0]  # case_index=0
            sample_true_objects_1.append(sample_true_object)

        # Assert
        assert len(sample_true_objects_0) == len(sample_true_objects_1)
        for a, b in zip(sample_true_objects_0, sample_true_objects_1):
            assert np.allclose(a.vec, b.vec)

        # (4) Make sure that the tester objects are reproducible.
        sample_tester_objects_1 = {0: [], 1: [], 2: []}
        for _, v in sim_results_dict_1[0].items():
            sample_objects = v["tester_objects"][0]  # case_index = 0
            for vecs_index, sample_object in enumerate(sample_objects):
                sample_tester_objects_1[vecs_index].append(sample_object)

        # Assert
        assert len(sample_tester_objects_0) == len(sample_tester_objects_1)
        for key in sample_tester_objects_0.keys():
            assert len(sample_tester_objects_0[key]) == len(
                sample_tester_objects_1[key]
            )
            for a, b in zip(sample_tester_objects_0[key], sample_tester_objects_1[key]):
                assert self.is_same_vecs(a.vecs, b.vecs)
