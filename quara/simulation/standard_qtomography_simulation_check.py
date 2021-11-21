from quara import simulation
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
from quara.simulation.standard_qtomography_simulation import SimulationResult


class StandardQTomographySimulationCheck:
    def __init__(self, simulation_result: SimulationResult) -> None:
        self.simulation_result = simulation_result

    def _print_summary(self, results) -> None:
        start_red = "\033[31m"
        start_green = "\033[32m"
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
        summary += f"Name: {self.simulation_result.simulation_setting.name}\n"
        summary += text_lines
        summary += "==============================="
        print(summary)

    def execute_all(
        self,
        consistency_check_eps: float = None,
        show_summary: bool = True,
        show_detail: bool = True,
        with_detail: bool = False,
        exec_check: dict = None,
    ) -> Union[bool, dict]:
        check_items = [
            "consistency",
            "mse_of_estimators",
            "mse_of_empi_dists",
            "physicality_violation",
        ]
        if not exec_check:
            exec_check = {item: True for item in check_items}
        else:
            for item in exec_check:
                if item not in check_items:
                    error_message = f"The key '{item}' of the argument 'exec_check' is invalid. 'exec_check' can be used with the following keys: {check_items}"
                    raise KeyError(error_message)

            for item in check_items:
                if item not in exec_check.keys():
                    exec_check[item] = True

        results = []

        def _to_result_dict(name: str, result: bool, detail: dict = None) -> dict:
            result_dict = {}
            result_dict["name"] = name
            result_dict["result"] = result
            result_dict["detail"] = detail
            return result_dict

        # MSE of Empirical Distributions
        if exec_check["mse_of_empi_dists"]:
            name = "MSE of Empirical Distributions"
            result = self.execute_mse_of_empirical_distribution_check(
                show_detail=show_detail
            )
            results.append(_to_result_dict(name, result))

        # Consistency
        if exec_check["consistency"]:
            name = "Consistency"
            result = self.execute_consistency_check(
                eps=consistency_check_eps, show_detail=show_detail
            )
            detail = result
            result = detail["possibly_ok"]
            results.append(_to_result_dict(name, result, detail))

        # MSE of estimators
        if exec_check["mse_of_estimators"]:
            name = "MSE of estimators"
            result = self.execute_mse_of_estimators_check(show_detail=show_detail)
            results.append(_to_result_dict(name, result))

        # Pysicality Violation
        if exec_check["physicality_violation"]:
            name = "Physicality Violation"
            result = self.execute_physicality_violation_check(show_detail=show_detail)
            results.append(_to_result_dict(name, result))

        total_result = np.all([r["result"] for r in results])
        # numpy.bool_ -> bool to serialize to json
        total_result = bool(total_result)
        all_result_dict = {
            "name": self.simulation_result.simulation_setting.name,
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
        # result = check_mse_of_empirical_distributions(
        #     simulation_setting=self.simulation_setting,
        #     estimation_results=self.estimation_results,
        #     show_detail=show_detail,
        # )
        result = check_mse_of_empirical_distributions(
            simulation_result=self.simulation_result,
            show_detail=show_detail,
        )
        return result

    def execute_consistency_check(
        self, eps: float = None, show_detail: bool = True
    ) -> dict:
        result_dict = execute_consistency_check(
            simulation_setting=self.simulation_result.simulation_setting,
            estimation_results=self.simulation_result.estimation_results,
            qtomography=self.simulation_result.qtomography,
            eps=eps,
            show_detail=show_detail,
        )
        return result_dict

    def execute_mse_of_estimators_check(self, show_detail: bool = True):
        try:
            result = check_mse_of_estimators(
                simulation_setting=self.simulation_result.simulation_setting,
                estimation_results=self.simulation_result.estimation_results,
                qtomography=self.simulation_result.qtomography,
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
        if (
            type(self.simulation_result.simulation_setting.estimator)
            == ProjectedLinearEstimator
        ):
            return physicality_violation_check.is_physical_qobjects_all(
                self.simulation_result.estimation_results,
                num_data=self.simulation_result.simulation_setting.num_data,
                show_detail=show_detail,
            )
        elif (
            type(self.simulation_result.simulation_setting.estimator) == LinearEstimator
        ):
            para = self.simulation_result.estimation_results[
                0
            ].estimated_qoperation.on_para_eq_constraint
            if para:
                return physicality_violation_check.is_eq_constraint_satisfied_all(
                    self.simulation_result.estimation_results,
                    self.simulation_result.simulation_setting.num_data,
                    show_detail=show_detail,
                )
            else:
                if show_detail:
                    print(
                        "[Skipped] Physicality Violation Check \nIf on_para_eq_constraint is False in Linear Estimator, nothing is checked."
                    )
                return True
        elif (
            type(self.simulation_result.simulation_setting.estimator)
            == LossMinimizationEstimator
        ):
            if self.simulation_result.simulation_setting.algo_option:
                on_algo_eq_constraint = (
                    self.simulation_result.simulation_setting.algo_option.on_algo_eq_constraint
                )
                on_algo_ineq_constraint = (
                    self.simulation_result.simulation_setting.algo_option.on_algo_ineq_constraint
                )

                results = []
                if on_algo_eq_constraint:
                    result = physicality_violation_check.is_eq_constraint_satisfied_all(
                        self.simulation_result.estimation_results,
                        self.simulation_result.simulation_setting.num_data,
                        show_detail=show_detail,
                    )
                    results.append(result)

                if on_algo_ineq_constraint:
                    result = (
                        physicality_violation_check.is_ineq_constraint_satisfied_all(
                            self.simulation_result.estimation_results,
                            self.simulation_result.simulation_setting.num_data,
                            show_detail=show_detail,
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
