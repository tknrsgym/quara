from typing import List, Tuple

import numpy as np

from quara.data_analysis.probability_based_loss_function import (
    ProbabilityBasedLossFunction,
    ProbabilityBasedLossFunctionOption,
)
from quara.data_analysis.minimization_algorithm import (
    MinimizationAlgorithm,
    MinimizationAlgorithmOption,
)
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.protocol.qtomography.standard.standard_qtomography_estimator import (
    StandardQTomographyEstimator,
    StandardQTomographyEstimationResult,
)


class LossMinimizationEstimationResult(StandardQTomographyEstimationResult):
    def __init__(
        self,
        qtomography: StandardQTomography,
        data,
        estimated_var_sequence: List[np.array],
        computation_times: List[float],
    ):
        super().__init__(qtomography, data, estimated_var_sequence, computation_times)


class LossMinimizationEstimator(StandardQTomographyEstimator):
    def __init__(self):
        super().__init__()

    def calc_estimate(
        self,
        qtomography: StandardQTomography,
        empi_dists: List[Tuple[int, np.array]],
        loss: ProbabilityBasedLossFunction,
        loss_option: ProbabilityBasedLossFunctionOption,
        algo: MinimizationAlgorithm,
        algo_option: MinimizationAlgorithmOption,
        is_computation_time_required: bool = False,
    ) -> StandardQTomographyEstimationResult:
        # TODO write 'this function changes loss, algo properties'
        # TODO is_computation_time_required

        # set loss settings
        loss.set_func_prob_dists_from_standard_qt(qtomography)
        loss.set_func_gradient_prob_dists_from_standard_qt(qtomography)
        loss.set_func_hessian_prob_dists_from_standard_qt(qtomography)
        empi_dists_tmp = [empi_dist[1] for empi_dist in empi_dists]
        loss.set_prob_dists_q(empi_dists_tmp)
        loss.set_from_option(loss_option)

        # set algorithm settings
        algo.set_from_loss(loss)
        algo.set_from_option(algo_option)

        # TODO validate error messages
        # validate
        if loss.is_option_sufficient() == False:
            raise ValueError(
                "loss.is_option_sufficient() must return True. But returns False"
            )
        if algo.is_loss_sufficient() == False:
            raise ValueError(
                "algo.is_loss_sufficient() must return True. But returns False"
            )
        if algo.is_option_sufficient() == False:
            raise ValueError(
                "algo.is_option_sufficient() must return True. But returns False"
            )
        if algo.is_loss_and_option_sufficient() == False:
            raise ValueError(
                "algo.is_loss_and_option_sufficient() must return True. But returns False"
            )

        # optimize
        algo_result = algo.optimize(loss, loss_option, algo_option)
        result = LossMinimizationEstimationResult(
            qtomography, empi_dists, [algo_result.value], [algo_result.computation_time]
        )

        # TODO post-processing
        return result
