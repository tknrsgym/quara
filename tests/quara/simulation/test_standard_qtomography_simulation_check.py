from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
import pytest

from quara.simulation.standard_qtomography_simulation_check import (
    StandardQTomographySimulationCheck,
)
from quara.simulation.standard_qtomography_simulation import (
    StandardQTomographySimulationSetting,
    SimulationResult,
    execute_simulation,
    generate_qtomography,
)

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects import matrix_basis
from quara.objects.qoperation_typical import generate_qoperation


class TestStandardQTomographySimulationCheck:
    def test_print_summary(self, capfd):
        # Arrange
        source_dummy_setting = StandardQTomographySimulationSetting(
            name="dummy name",
            true_object=None,
            tester_objects=[],
            estimator=None,
            seed_data=None,
            n_rep=None,
            num_data=[],
            schedules=None,
            eps_proj_physical=1e-13,
        )
        source_dummy_results = None

        sim_check = StandardQTomographySimulationCheck(
            SimulationResult(
                simulation_setting=source_dummy_setting,
                estimation_results=source_dummy_results,
                empi_dists_sequences=[],  # Dummy
                qtomography=None,  # Dummy
            )
        )

        source_results = [
            {"name": "MSE of Empirical Distributions", "result": True, "detail": None},
            {
                "name": "Consistency",
                "result": True,
                "detail": {"possibly_ok": True, "to_be_checked": False},
            },
            {"name": "MSE of estimators", "result": True, "detail": None},
            {"name": "Physicality Violation", "result": True, "detail": None},
        ]
        # Act
        sim_check._print_summary(source_results)
        out, _ = capfd.readouterr()
        # Assert
        start_red = "\033[31m"
        start_green = "\033[32m"
        end_color = "\033[0m"
        expected = f"""========== Summary ============
Name: dummy name
MSE of Empirical Distributions: {start_green}OK{end_color}
Consistency: {start_green}OK{end_color}
MSE of estimators: {start_green}OK{end_color}
Physicality Violation: {start_green}OK{end_color}
===============================
"""
        # Case 2:
        # Arrange
        source_results = [
            {"name": "MSE of Empirical Distributions", "result": False, "detail": None},
            {
                "name": "Consistency",
                "result": False,
                "detail": {"possibly_ok": False, "to_be_checked": False},
            },
            {"name": "MSE of estimators", "result": True, "detail": None},
            {"name": "Physicality Violation", "result": False, "detail": None},
        ]
        # Act
        sim_check._print_summary(source_results)
        out, _ = capfd.readouterr()
        # Assert
        expected = f"""========== Summary ============
Name: dummy name
MSE of Empirical Distributions: {start_red}NG{end_color}
Consistency: {start_red}NG{end_color}, but you may not need to check report, because on_para_eq_constraint is False.
MSE of estimators: {start_green}OK{end_color}
Physicality Violation: {start_red}NG{end_color}
===============================
"""
        assert out == expected

        # Case 3:
        # Arrange
        source_results = [
            {"name": "MSE of Empirical Distributions", "result": True, "detail": None},
            {
                "name": "Consistency",
                "result": False,
                "detail": {"possibly_ok": False, "to_be_checked": True},
            },
            {"name": "MSE of estimators", "result": False, "detail": None},
            {"name": "Physicality Violation", "result": True, "detail": None},
        ]
        # Act
        sim_check._print_summary(source_results)
        out, _ = capfd.readouterr()
        # Assert
        expected = f"""========== Summary ============
Name: dummy name
MSE of Empirical Distributions: {start_green}OK{end_color}
Consistency: {start_red}NG{end_color}, and you need to check report.
MSE of estimators: {start_red}NG{end_color}
Physicality Violation: {start_green}OK{end_color}
===============================
"""

    def test_execute_all_with_exec_check(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        true_object = generate_qoperation(mode="state", name="x0", c_sys=c_sys)
        tester_objects = [
            generate_qoperation(mode="povm", name=name, c_sys=c_sys)
            for name in ["x", "y", "z"]
        ]
        sim_setting = StandardQTomographySimulationSetting(
            name="dummy name",
            true_object=true_object,
            tester_objects=tester_objects,
            estimator=LinearEstimator(),
            seed_data=777,
            n_rep=1,
            num_data=[10],
            schedules=None,
            eps_proj_physical=1e-13,
        )
        qtomography = generate_qtomography(
            sim_setting,
            para=True,
            init_with_seed=False,
        )

        sim_result = execute_simulation(
            qtomography=qtomography,
            simulation_setting=sim_setting,
        )
        sim_check = StandardQTomographySimulationCheck(sim_result)

        # Case 1:
        # Act
        check_result = sim_check.execute_all(with_detail=True)
        actual = set(r["name"] for r in check_result["results"])

        # Assert
        expected = {
            "Consistency",
            "MSE of Empirical Distributions",
            "MSE of estimators",
            "Physicality Violation",
        }
        assert actual == expected

        # Case 2:
        source_exec_check = {"consistency": False}
        check_result = sim_check.execute_all(
            with_detail=True, exec_check=source_exec_check
        )
        actual = set(r["name"] for r in check_result["results"])

        # Assert
        expected = {
            "MSE of Empirical Distributions",
            "MSE of estimators",
            "Physicality Violation",
        }
        assert actual == expected

        # Case 3:
        source_exec_check = {"mse_of_empi_dists": False}
        check_result = sim_check.execute_all(
            with_detail=True, exec_check=source_exec_check
        )
        actual = set(r["name"] for r in check_result["results"])

        # Assert
        expected = {
            "Consistency",
            "MSE of estimators",
            "Physicality Violation",
        }
        assert actual == expected

        # Case 4:
        source_exec_check = {"mse_of_estimators": False}
        check_result = sim_check.execute_all(
            with_detail=True, exec_check=source_exec_check
        )
        actual = set(r["name"] for r in check_result["results"])

        # Assert
        expected = {
            "Consistency",
            "MSE of Empirical Distributions",
            "Physicality Violation",
        }
        assert actual == expected

        # Case 5:
        source_exec_check = {"physicality_violation": False}
        check_result = sim_check.execute_all(
            with_detail=True, exec_check=source_exec_check
        )
        actual = set(r["name"] for r in check_result["results"])

        # Assert
        expected = {
            "Consistency",
            "MSE of Empirical Distributions",
            "MSE of estimators",
        }
        assert actual == expected

        # Case 6:
        source_exec_check = {
            "consistency": True,
            "mse_of_estimators": True,
            "mse_of_empi_dists": False,
            "physicality_violation": False,
        }
        check_result = sim_check.execute_all(
            with_detail=True, exec_check=source_exec_check
        )
        actual = set(r["name"] for r in check_result["results"])

        # Assert
        expected = {"Consistency", "MSE of estimators"}
        assert actual == expected

        # Case7: All False
        source_exec_check = {
            "consistency": False,
            "mse_of_estimators": False,
            "mse_of_empi_dists": False,
            "physicality_violation": False,
        }
        check_result = sim_check.execute_all(
            with_detail=True, exec_check=source_exec_check
        )
        actual = set(r["name"] for r in check_result["results"])

        # Assert
        expected = set()
        assert actual == expected

        # Case8: invalid input
        source_exec_check = {
            "invalid": True,
        }

        with pytest.raises(KeyError):
            # ValueError: The key 'invalid' of the argument 'exec_check' is invalid. 'exec_check' can be used with the following keys: ['consistency', 'mse_of_estimators', 'mse_of_empi_dists', 'physicality_violation']
            _ = sim_check.execute_all(with_detail=True, exec_check=source_exec_check)
