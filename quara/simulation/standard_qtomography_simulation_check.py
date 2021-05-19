from typing import List, Union
import warnings

import numpy as np

from quara.data_analysis import physicality_violation_check
from quara.data_analysis.physicality_violation_check import calc_unphysical_qobjects_n
from quara.loss_function.mean_squared_error import (
    check_mse_of_empirical_distributions,
    check_mse_of_estimators,
)
from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
from quara.protocol.qtomography.standard.projected_linear_estimator import (
    ProjectedLinearEstimator,
)
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
)
from quara.simulation.consistency_check import execute_consistency_check
from quara.simulation.standard_qtomography_simulation import (
    StandardQTomographySimulationSetting,
)


class StandardQTomographySimulationCheck:
    def __init__(
        self,
        simulation_setting: StandardQTomographySimulationSetting,
        estimation_results: List["EstimationResult"],
    ) -> None:
        self.simulation_setting = simulation_setting
        self.estimation_results = estimation_results

    def _print_summary(self, results) -> None:
        start_red = "\033[31m"
        start_green = "\033[32m"
        start_yellow_bg = "\033[43m"
        end_color = "\033[0m"
        ok_text = f"{start_green}OK{end_color}"
        ng_text = f"{start_red}NG{end_color}"

        text_lines = ""
        for result_dict in results:
            name = result_dict["name"]
            result = result_dict["result"]

            if name == "Consistency":
                detail = result_dict["detail"]
                text = f"{ok_text if detail['possibly_ok'] else ng_text}"
                if detail["possibly_ok"]:
                    text = ok_text
                else:
                    if detail["to_be_checked"]:
                        to_be_checked_text = f"and you need to check report."
                    else:
                        to_be_checked_text = f"but you may not need to check report, because on_para_eq_constraint is False."
                    text = f"{ng_text}, {to_be_checked_text}"
                text_lines += f"{name}: {text}\n"
            else:
                text_lines += f"{name}: {ok_text if result else ng_text}\n"

        summary = "========== Summary ============\n"
        summary += f"Name: {self.simulation_setting.name}\n"
        summary += text_lines
        summary += "==============================="
        print(summary)

    def execute_all(
        self,
        consistency_check_eps: float = None,
        show_summary: bool = True,
        show_detail: bool = True,
        with_detail: bool = False,
    ) -> Union[bool, dict]:
        results = []

        def _to_result_dict(name: str, result: bool, detail: dict = None) -> dict:
            result_dict = {}
            result_dict["name"] = name
            result_dict["result"] = result
            result_dict["detail"] = detail
            return result_dict

        # MSE of Empirical Distributions
        name = "MSE of Empirical Distributions"
        result = self.execute_mse_of_empirical_distribution_check(
            show_detail=show_detail
        )
        results.append(_to_result_dict(name, result))

        # Consistency
        name = "Consistency"
        result = self.execute_consistency_check(
            eps=consistency_check_eps, show_detail=show_detail
        )
        detail = result
        result = detail["possibly_ok"]
        results.append(_to_result_dict(name, result, detail))

        # MSE of estimators
        name = "MSE of estimators"
        result = self.execute_mse_of_estimators_check(show_detail=show_detail)
        results.append(_to_result_dict(name, result))

        # Pysicality Violation
        name = "Physicality Violation"
        result = self.execute_physicality_violation_check(show_detail=show_detail)
        results.append(_to_result_dict(name, result))

        total_result = np.all([r["result"] for r in results])
        # numpy.bool_ -> bool to serialize to json
        total_result = bool(total_result)
        all_result_dict = {
            "name": self.simulation_setting.name,
            "total_result": total_result,
            "results": results,
        }

        # Show summary
        if show_summary:
            self._print_summary(results)

        if with_detail:
            return all_result_dict
        else:
            return all_result_dict["total_result"]

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
    ) -> dict:
        result_dict = execute_consistency_check(
            simulation_setting=self.simulation_setting,
            estimation_results=self.estimation_results,
            eps=eps,
            show_detail=show_detail,
        )
        return result_dict

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
                if show_detail:
                    print(
                        "[Skipped] Physicality Violation Check \nIf on_para_eq_constraint is False in Linear Estimator, nothing is checked."
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

                if on_algo_ineq_constraint:
                    result = (
                        physicality_violation_check.is_ineq_constraint_satisfied_all(
                            self.estimation_results, show_detail=show_detail
                        )
                    )
                    results.append(result)

                if not results:
                    print(
                        "[Skipped] Physicality Violation Check \nIf both on_algo_eq_constraint and on_algo_ineq_constraint of algo_option are False in LossMinimizationEstimator, nothing is checked."
                    )
                if False in results:
                    return False
                else:
                    return True
            else:
                print(
                    "[Skipped] Physicality Violation Check \nIf algo_option is None in LossMinimizationEstimator, nothing is checked."
                )
                return True
        else:
            print(
                "[Skipped] Physicality Violation Check \nCheck nothing except LinearEstimator, ProjectedLinearEstimator and LossMinimizationEstimator."
            )
            return True
