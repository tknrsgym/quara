from typing import List
import warnings

from quara.data_analysis import simulation
from quara.data_analysis import physicality_violation_check

from quara.data_analysis.simulation import StandardQTomographySimulationSetting
from quara.data_analysis.consistency_check import execute_consistency_check
from quara.data_analysis.mean_squared_error import (
    check_mse_of_empirical_distributions,
    check_mse_of_estimators,
)
from quara.data_analysis.physicality_violation_check import calc_unphysical_qobjects_n

from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
from quara.protocol.qtomography.standard.projected_linear_estimator import (
    ProjectedLinearEstimator,
)
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
)


class StandardQTomographySimulationCheck:
    def __init__(
        self,
        simulation_setting: StandardQTomographySimulationSetting,
        estimation_results: List["EstimationResult"],
    ) -> None:
        self.simulation_setting = simulation_setting
        self.estimation_results = estimation_results

    def execute_all(
        self,
        consistency_check_eps: float = None,
        show_summary: bool = True,
        show_detail: bool = True,
    ) -> bool:
        results = []
        test_names = []

        # MSE of Empirical Distributions
        test_names.append("MSE of Empirical Distributions")
        result = self.execute_mse_of_empirical_distribution_check(
            show_detail=show_detail
        )
        results.append(result)

        # Consistency
        test_names.append("Consistency")
        result = self.execute_consistency_check(
            eps=consistency_check_eps, show_detail=show_detail
        )
        results.append(result)

        # MSE of estimators
        test_names.append("MSE of estimators")
        result = self.execute_mse_of_estimators_check(show_detail=show_detail)
        results.append(result)

        # Pysicality Violation
        test_names.append("Physicality Violation")
        result = self.execute_physicality_violation_check()
        results.append(result)

        # Show summary
        if show_summary:
            lines = [
                f"{name}: {'OK' if r else 'NG'}" for name, r in zip(test_names, results)
            ]
            summary = "========== Summary ============\n"
            summary += "\n".join(lines)
            print(summary)

        if False in results:
            return False
        else:
            return True

    def execute_mse_of_empirical_distribution_check(
        self, show_detail: bool = True
    ) -> bool:
        result = check_mse_of_empirical_distributions(
            simulation_setting=self.simulation_setting,
            estimation_results=self.estimation_results,
            show_detail=show_detail,
        )
        return result

    def execute_consistency_check(
        self, eps: float = None, show_detail: bool = True
    ) -> bool:
        result = execute_consistency_check(
            simulation_setting=self.simulation_setting,
            estimation_results=self.estimation_results,
            eps=eps,
            show_detail=show_detail,
        )
        return result

    def execute_mse_of_estimators_check(self, show_detail: bool = True):
        try:
            result = check_mse_of_estimators(
                simulation_setting=self.simulation_setting,
                estimation_results=self.estimation_results,
                show_detail=show_detail,
            )
        except TypeError as e:
            import traceback

            t = traceback.format_exception_only(type(e), e)
            if "Estimator must be LinearEstimator, " in t[0]:

                warnings.warn(
                    "Estimator MSE is not checked except for LinearEstimator, ProjectedLinearEstimator, Maximum-likelihood."
                )
                return True
            else:
                raise

        return result

    def execute_physicality_violation_check(self, show_detail: bool = True) -> bool:
        if type(self.simulation_setting.estimator) == ProjectedLinearEstimator:
            return physicality_violation_check.is_physical_qobjects_all(
                self.estimation_results, show_detail
            )
        elif type(self.simulation_setting.estimator) == LinearEstimator:
            para = self.estimation_results[0].estimated_qoperation.on_para_eq_constraint
            if para:
                return physicality_violation_check.is_eq_constraint_satisfied_all(
                    self.estimation_results, show_detail=show_detail
                )
            else:
                warnings.warn(
                    "If on_para_eq_constraint is False in Linear Estimator, nothing is checked"
                )
                return True
        elif type(self.simulation_setting.estimator) == LossMinimizationEstimator:
            if self.simulation_setting.algo_option:
                on_algo_eq_constraint = (
                    self.simulation_setting.algo_option.on_algo_eq_constraint
                )
                on_algo_ineq_constraint = (
                    self.simulation_setting.algo_option.on_algo_ineq_constraint
                )

                results = []
                if on_algo_eq_constraint:
                    result = physicality_violation_check.is_eq_constraint_satisfied_all(
                        self.estimation_results, show_detail=show_detail
                    )
                    results.append(result)
                    if show_detail:
                        message = f"[{'OK' if result else 'NG'}] is_eq_constraint_satisfied_all"
                        print(message)

                if on_algo_ineq_constraint:
                    result = physicality_violation_check.is_ineq_constraint_satisfied_all(
                        self.estimation_results, show_detail=show_detail
                    )
                    results.append(result)
                    if show_detail:
                        message = f"[{'OK' if result else 'NG'}] is_ineq_constraint_satisfied_all"
                        print(message)

                if not results:
                    warnings.warn(
                        "If both on_algo_eq_constraint and on_algo_ineq_constraint of algo_option are False in LossMinimizationEstimator, nothing is checked"
                    )
                if False in results:
                    return False
                else:
                    return True
            else:
                warnings.warn(
                    "If algo_option is None in LossMinimizationEstimator, nothing is checked"
                )
                return True
        else:
            warnings.warn(
                "Check nothing except LinearEstimator, ProjectedLinearEstimator and LossMinimizationEstimator."
            )
            return True

