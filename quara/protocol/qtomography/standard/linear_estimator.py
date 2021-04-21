import time
from typing import List, Tuple

import numpy as np

from quara.objects.qoperation import QOperation
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.protocol.qtomography.standard.standard_qtomography_estimator import (
    StandardQTomographyEstimator,
    StandardQTomographyEstimationResult,
)


class LinearEstimationResult(StandardQTomographyEstimationResult):
    def __init__(
        self,
        qtomography: StandardQTomography,
        data,
        estimated_var_sequence: List[np.ndarray],
        computation_times: List[float],
    ):
        super().__init__(qtomography, data, estimated_var_sequence, computation_times)


class LinearEstimator(StandardQTomographyEstimator):
    def __init__(self):
        super().__init__()

    def calc_estimate(
        self,
        qtomography: StandardQTomography,
        empi_dists: List[Tuple[int, np.ndarray]],
        is_computation_time_required: bool = False,
    ) -> LinearEstimationResult:
        """calculates estimate variables.

        see :func:`~quara.protocol.qtomography.standard.standard_qtomography_estimator.StandardQTomographyEstimator.calc_estimate`
        """
        result = self.calc_estimate_sequence(
            qtomography,
            [empi_dists],
            is_computation_time_required=is_computation_time_required,
        )

        return result

    def calc_estimate_sequence(
        self,
        qtomography: StandardQTomography,
        empi_dists_sequence: List[List[Tuple[int, np.ndarray]]],
        is_computation_time_required: bool = False,
    ) -> LinearEstimationResult:
        """calculates sequence of estimate variables.

        see :func:`~quara.protocol.qtomography.standard.standard_qtomography_estimator.StandardQTomographyEstimator.calc_estimate_sequence`
        """
        if not qtomography.is_fullrank_matA():
            raise Exception

        A = qtomography.calc_matA()
        b = qtomography.calc_vecB()

        A_ddag = np.linalg.inv(A.T @ A) @ A.T

        estimate_sequence = []
        comp_time_sequence = [] if is_computation_time_required else None
        for empi_dists in empi_dists_sequence:
            if is_computation_time_required:
                start_time = time.time()

            empi_dists_tmp = [empi_dist[1] for empi_dist in empi_dists]
            f = np.vstack(empi_dists_tmp).flatten()
            v = A_ddag @ (f - b)
            # -------
            estimate_sequence.append(v)

            if is_computation_time_required:
                comp_time = time.time() - start_time
                comp_time_sequence.append(comp_time)

        result = LinearEstimationResult(
            qtomography, empi_dists_sequence, estimate_sequence, comp_time_sequence,
        )
        return result
