import time
from typing import List, Tuple

import numpy as np

from quara.data_analysis.probability_based_loss_function import (
    ProbabilityBasedLossFunction,
    ProbabilityBasedLossFunctionOption,
)
from quara.data_analysis.weighted_probability_based_squared_error import (
    WeightedProbabilityBasedSquaredError,
    WeightedProbabilityBasedSquaredErrorOption,
)
from quara.data_analysis.minimization_algorithm import (
    MinimizationAlgorithm,
    MinimizationAlgorithmOption,
)
from quara.objects.qoperation import QOperation
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
    LossMinimizationEstimationResult,
)
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.protocol.qtomography.standard.standard_qtomography_estimator import (
    StandardQTomographyEstimator,
    StandardQTomographyEstimationResult,
)
from quara.utils import matrix_util


class WeightedLeastSquaresEstimationResult(LossMinimizationEstimationResult):
    def __init__(
        self,
        qtomography: StandardQTomography,
        data,
        estimated_var_sequence: List[np.array],
        computation_times: List[float],
    ):
        super().__init__(qtomography, data, estimated_var_sequence, computation_times)


class WeightedLeastSquaresEstimator(LossMinimizationEstimator):
    def __init__(self, num_var: int):
        super().__init__()
        self._num_var = num_var

    def calc_estimate(
        self,
        qtomography: StandardQTomography,
        empi_dists: List[Tuple[int, np.array]],
        algo: MinimizationAlgorithm,
        algo_option: MinimizationAlgorithmOption,
        mode_covariance: str,
        mode_inverse: str,
        is_computation_time_required: bool = False,
    ) -> StandardQTomographyEstimationResult:
        result = self.calc_estimate_sequence(
            qtomography,
            [empi_dists],
            algo,
            algo_option,
            mode_covariance,
            mode_inverse,
            is_computation_time_required=is_computation_time_required,
        )

        return result

    def calc_estimate_sequence(
        self,
        qtomography: StandardQTomography,
        empi_dists_sequence: List[List[Tuple[int, np.array]]],
        algo: MinimizationAlgorithm,
        algo_option: MinimizationAlgorithmOption,
        mode_covariance: str,
        mode_inverse: str,
        is_computation_time_required: bool = False,
    ) -> StandardQTomographyEstimationResult:

        estimated_var_sequence = []
        computation_times = []

        for empi_dists in empi_dists_sequence:
            # weight_matrices
            weight_matrices = []
            for (num_data, empi_dist) in empi_dists:
                empi_dist = matrix_util.replace_prob_dist(empi_dist)

                # calc covariance matrix
                if mode_covariance == "scm":
                    covariance_mat = matrix_util.calc_covariance_mat(
                        empi_dist, num_data
                    )
                elif mode_covariance == "ucm":
                    covariance_mat = matrix_util.calc_covariance_mat(
                        empi_dist, num_data - 1
                    )
                else:
                    raise ValueError(f"unsupported mode_covariance={mode_covariance}")

                # calc inverse of covariance matrix
                if mode_inverse == "extraction":
                    weight_matrix = np.zeros(covariance_mat.shape)
                    (row, col) = covariance_mat.shape
                    extracted_mat = covariance_mat[:-1, :-1] + np.eye(row - 1) / (
                        num_data ** (3 / 2)
                    )

                    extracted_mat_inv = np.linalg.inv(extracted_mat)
                    if row == 2 and col == 2:
                        weight_matrix[0, 0] = extracted_mat_inv[0, 0]
                    else:
                        weight_matrix[:row, :col] = extracted_mat_inv
                else:
                    raise ValueError(f"unsupported mode_inverse={mode_inverse}")
                weight_matrices.append(weight_matrix)

            # lossFunction
            loss = WeightedProbabilityBasedSquaredError(
                self._num_var, weight_matrices=weight_matrices
            )
            loss_option = WeightedProbabilityBasedSquaredErrorOption()

            # calc estimate
            tmp_result = super().calc_estimate_sequence(
                qtomography,
                [empi_dists],
                loss,
                loss_option,
                algo,
                algo_option,
                is_computation_time_required=is_computation_time_required,
            )
            estimated_var_sequence.append(tmp_result.estimated_var)
            computation_times.append(tmp_result.computation_time)

        result = LossMinimizationEstimationResult(
            qtomography, empi_dists_sequence, estimated_var_sequence, computation_times
        )
        return result
