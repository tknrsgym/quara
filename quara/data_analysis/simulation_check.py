from typing import List

from quara.data_analysis.simulation import StandardQTomographySimulationSetting
from quara.data_analysis.consistency_check import execute_consistency_check
from quara.data_analysis.mean_squared_error import check_mse_of_empirical_distributions

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

    def execute_mse_of_estimators_check(self):
        pass

    def execute_physicality_violation_check(self):
        pass
