from typing import List
from quara.data_analysis import simulation

from quara.data_analysis.simulation import StandardQTomographySimulationSetting
from quara.data_analysis.consistency_check import execute_consistency_check
from quara.data_analysis.mean_squared_error import (
    check_mse_of_empirical_distributions,
    check_mse_of_estimators,
)
from quara.data_analysis.physicality_violation_check import physicality_violation_check

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
        result = check_mse_of_estimators(
            simulation_setting=self.simulation_setting,
            estimation_results=self.estimation_results,
            eps=eps,
            show_detail=show_detail,
        )
        return result

    def execute_physicality_violation_check(self):
        if type(self.simulation_setting.estimator) == ProjectedLinearEstimator:
            n = physicality_violation_check.calc_unphysical_qobjects_n(
                self.estimation_results
            )
            return n == 0
        elif type(self.simulation_setting.estimator) == LinearEstimator:
            para == self.estimation_results[0].qtomography.on_para_eq_constraint
            # TODO:
            raise NotImplementedError()
        elif type(self.simulation_setting.estimator) == LossMinimizationEstimator:
            raise NotImplementedError()
        else:
            # TODO: message
            raise TypeError()
