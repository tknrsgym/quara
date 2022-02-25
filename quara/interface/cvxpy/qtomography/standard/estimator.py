from typing import List, Tuple
import time

import numpy as np
import scipy as sp
import cvxpy as cp

from quara.objects.qoperation import QOperation
from quara.protocol.qtomography.standard.standard_qtomography import (
    StandardQTomography,
)
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
    LossMinimizationEstimationResult,
)
from quara.protocol.qtomography.standard.standard_qtomography_estimator import (
    StandardQTomographyEstimator,
    StandardQTomographyEstimationResult,
)
from quara.interface.cvxpy.qtomography.standard.loss_function import (
    CvxpyLossFunction,
    CvxpyLossFunctionOption,
)
from quara.interface.cvxpy.qtomography.standard.minimization_algorithm import (
    CvxpyMinimizationAlgorithm,
    CvxpyMinimizationAlgorithmOption,
    CvxpyMinimizationResult,
)


class CvxpyLossMinimizationEstimationResult(LossMinimizationEstimationResult):
    def __init__(
        self,
        estimated_var_sequence: List[np.ndarray],
        computation_times: List[float],
        template_qoperation: QOperation,
        estimated_loss_sequence: List[float] = None,
    ):
        super().__init__(estimated_var_sequence, computation_times, template_qoperation)
        self._estimated_loss_sequence: List[float] = estimated_loss_sequence

    @property
    def estimated_loss_sequence(self) -> List[float]:
        """returns sequence of estimated loss.

        Returns
        -------
        List[float]
            sequence of estimated loss.
        """
        return self._estimated_loss_sequence


class CvxpyLossMinimizationEstimator(LossMinimizationEstimator):
    def __init__(self):
        super().__init__()

    def calc_estimate(
        self,
        qtomography: StandardQTomography,
        empi_dists: List[Tuple[int, np.ndarray]],
        loss: CvxpyLossFunction,
        loss_option: CvxpyLossFunctionOption,
        algo: CvxpyMinimizationAlgorithm,
        algo_option: CvxpyMinimizationAlgorithmOption,
        is_computation_time_required: bool = False,
    ) -> CvxpyLossMinimizationEstimationResult:
        """calculates estimate variables.
        Notice: this function updates ``loss`` and ``algo`` properties.

        Parameters
        ----------
        qtomography : StandardQTomography
            StandardQTomography to calculates estimate variables.
        empi_dists : List[Tuple[int, np.ndarray]]
            empirical distributions to calculates estimate variables.
        loss : CvxpyLossFunction
            ProbabilityBasedLossFunction to calculates estimate variables.
        loss_option : CvxpyLossFunctionOption
            ProbabilityBasedLossFunctionOption to calculates estimate variables.
        algo : CvxpyMinimizationAlgorithm
            MinimizationAlgorithm to calculates estimate variables.
        algo_option : CvxpyMinimizationAlgorithmOption
            MinimizationAlgorithmOption to calculates estimate variables.
        is_computation_time_required : bool, optional
            whether to include computation time in the return value or not, by    default False.

        Returns
        -------
        CvxpyLossMinimizationEstimationResult
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
        loss: CvxpyLossFunction,
        loss_option: CvxpyLossFunctionOption,
        algo: CvxpyMinimizationAlgorithm,
        algo_option: CvxpyMinimizationAlgorithmOption,
        is_computation_time_required: bool = False,
    ) -> CvxpyLossMinimizationEstimationResult:
        """calculates sequence of estimate variables.
        Notice: this function updates ``loss`` and ``algo`` properties.

        Parameters
        ----------
        qtomography : StandardQTomography
            StandardQTomography to calculates estimate variables.
        empi_dists_sequence : List[List[Tuple[int, np.ndarray]]]
            sequence of empirical distributions to calculates estimate variables.
        loss : CvxpyLossFunction
            ProbabilityBasedLossFunction to calculates estimate variables.
        loss_option : CvxpyLossFunctionOption
            ProbabilityBasedLossFunctionOption to calculates estimate variables.
        algo : CvxpyMinimizationAlgorithm
            MinimizationAlgorithm to calculates estimate variables.
        algo_option : CvxpyMinimizationAlgorithmOption
            MinimizationAlgorithmOption to calculates estimate variables.
        is_computation_time_required : bool, optional
            whether to include computation time in the return value or not, by default False.

        Returns
        -------
        CvxpyLossMinimizationEstimationResult
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
        loss_value_sequence = []
        computation_times = [] if is_computation_time_required else None

        loss.set_standard_qtomography(sqt=qtomography)
        loss.set_from_option(loss_option)
        algo.set_from_option(algo_option)
        for empi_dists in empi_dists_sequence:
            if is_computation_time_required:
                start_time = time.time()

            # set loss settings
            loss.set_prob_dists_data_from_empi_dists(empi_dists)

            # set algorithm settings
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
            algo_result = algo.optimize()

            # post-processing
            estimated_var_sequence.append(algo_result.variable_value)
            loss_value_sequence.append(algo_result.loss_value)
            if is_computation_time_required:
                computation_times.append(prepare_time + algo_result.computation_time)

        result = CvxpyLossMinimizationEstimationResult(
            estimated_var_sequence,
            computation_times=computation_times,
            template_qoperation=qtomography._template_qoperation,
            estimated_loss_sequence=loss_value_sequence,
        )
        return result
