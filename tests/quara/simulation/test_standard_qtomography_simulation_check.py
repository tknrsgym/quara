import pytest

from quara.simulation.standard_qtomography_simulation_check import (
    StandardQTomographySimulationCheck,
)
from quara.simulation.standard_qtomography_simulation import (
    StandardQTomographySimulationSetting,
    SimulationResult,
)


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
