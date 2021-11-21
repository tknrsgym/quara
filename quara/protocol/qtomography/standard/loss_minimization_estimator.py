from quara.objects.qoperation import QOperation
import time
from typing import List, Tuple

import numpy as np

from quara.loss_function.probability_based_loss_function import (
    ProbabilityBasedLossFunction,
    ProbabilityBasedLossFunctionOption,
)
from quara.minimization_algorithm.minimization_algorithm import (
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
        estimated_var_sequence: List[np.ndarray],
        computation_times: List[float],
        template_qoperation: QOperation,
    ):
        super().__init__(estimated_var_sequence, computation_times, template_qoperation)


class LossMinimizationEstimator(StandardQTomographyEstimator):
    def __init__(self):
        super().__init__()

    def calc_estimate(
        self,
        qtomography: StandardQTomography,
        empi_dists: List[Tuple[int, np.ndarray]],
        loss: ProbabilityBasedLossFunction,
        loss_option: ProbabilityBasedLossFunctionOption,
        algo: MinimizationAlgorithm,
        algo_option: MinimizationAlgorithmOption,
        is_computation_time_required: bool = False,
    ) -> StandardQTomographyEstimationResult:
        """calculates estimate variables.

        Notice: this function updates ``loss`` and ``algo`` properties.

        Parameters
        ----------
        qtomography : StandardQTomography
            StandardQTomography to calculates estimate variables.
        empi_dists : List[Tuple[int, np.ndarray]]
            empirical distributions to calculates estimate variables.
        loss : ProbabilityBasedLossFunction
            ProbabilityBasedLossFunction to calculates estimate variables.
        loss_option : ProbabilityBasedLossFunctionOption
            ProbabilityBasedLossFunctionOption to calculates estimate variables.
        algo : MinimizationAlgorithm
            MinimizationAlgorithm to calculates estimate variables.
        algo_option : MinimizationAlgorithmOption
            MinimizationAlgorithmOption to calculates estimate variables.
        is_computation_time_required : bool, optional
            whether to include computation time in the return value or not, by default False.

        Returns
        -------
        StandardQTomographyEstimationResult
            estimation result.

        Raises
        ------
        ValueError
            loss.is_option_sufficient() returns False.
        ValueError
            algo.is_loss_sufficient() returns False.
        ValueError
            algo.is_option_sufficient() returns False.
        ValueError
            algo.is_loss_and_option_sufficient() returns False.
        """
        result = self.calc_estimate_sequence(
            qtomography,
            [empi_dists],
            loss,
            loss_option,
            algo,
            algo_option,
            is_computation_time_required=is_computation_time_required,
        )

        return result

    def calc_estimate_sequence(
        self,
        qtomography: StandardQTomography,
        empi_dists_sequence: List[List[Tuple[int, np.ndarray]]],
        loss: ProbabilityBasedLossFunction,
        loss_option: ProbabilityBasedLossFunctionOption,
        algo: MinimizationAlgorithm,
        algo_option: MinimizationAlgorithmOption,
        is_computation_time_required: bool = False,
    ) -> StandardQTomographyEstimationResult:
        """calculates sequence of estimate variables.

        Notice: this function updates ``loss`` and ``algo`` properties.

        Parameters
        ----------
        qtomography : StandardQTomography
            StandardQTomography to calculates estimate variables.
        empi_dists_sequence : List[List[Tuple[int, np.ndarray]]]
            sequence of empirical distributions to calculates estimate variables.
        loss : ProbabilityBasedLossFunction
            ProbabilityBasedLossFunction to calculates estimate variables.
        loss_option : ProbabilityBasedLossFunctionOption
            ProbabilityBasedLossFunctionOption to calculates estimate variables.
        algo : MinimizationAlgorithm
            MinimizationAlgorithm to calculates estimate variables.
        algo_option : MinimizationAlgorithmOption
            MinimizationAlgorithmOption to calculates estimate variables.
        is_computation_time_required : bool, optional
            whether to include computation time in the return value or not, by default False.

        Returns
        -------
        StandardQTomographyEstimationResult
            estimation result.

        Raises
        ------
        ValueError
            loss.is_option_sufficient() returns False.
        ValueError
            algo.is_loss_sufficient() returns False.
        ValueError
            algo.is_option_sufficient() returns False.
        ValueError
            algo.is_loss_and_option_sufficient() returns False.
        """
        estimated_var_sequence = []
        computation_times = [] if is_computation_time_required else None

        for empi_dists in empi_dists_sequence:
            if is_computation_time_required:
                start_time = time.time()

            # set loss settings
            loss.set_from_standard_qtomography_option_data(
                qtomography,
                loss_option,
                empi_dists,
                algo.is_gradient_required,
                algo.is_hessian_required,
            )

            # set algorithm settings
            algo.set_from_option(algo_option)
            algo.set_constraint_from_standard_qt_and_option(qtomography, algo_option)
            algo.set_from_loss(loss)

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

            if is_computation_time_required:
                prepare_time = time.time() - start_time

            # optimize
            algo_result = algo.optimize(
                loss,
                loss_option,
                algo_option,
                on_iteration_history=is_computation_time_required,
            )

            # post-processing
            estimated_var_sequence.append(algo_result.value)
            if is_computation_time_required:
                computation_times.append(prepare_time + algo_result.computation_time)

        result = LossMinimizationEstimationResult(
            estimated_var_sequence, computation_times, qtomography._template_qoperation
        )
        return result
