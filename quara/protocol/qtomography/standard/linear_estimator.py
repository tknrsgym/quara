import itertools
import time
from typing import Dict, List, Tuple

import numpy as np

from quara.objects.qoperation import QOperation
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.protocol.qtomography.standard.standard_qtomography_estimator import (
    StandardQTomographyEstimator,
)


class LinearEstimator(StandardQTomographyEstimator):
    def __init__(self):
        super().__init__()

    def calc_estimate_var(
        self,
        qtomography: StandardQTomography,
        empi_dists: List[Tuple[int, np.array]],
        is_computation_time_required: bool = False,
    ) -> Dict:
        """calculates estimate variables.

        see :func:`~quara.protocol.qtomography.standard.standard_qtomography_estimator.StandardQTomographyEstimator.calc_estimate_var`
        """
        estimate = self.calc_estimate_sequence_var(
            qtomography,
            [empi_dists],
            is_computation_time_required=is_computation_time_required,
        )

        value = {"estimate": estimate["estimate"][0]}
        if is_computation_time_required:
            value["computation_time"] = estimate["computation_time"][0]

        return value

    def calc_estimate_sequence_var(
        self,
        qtomography: StandardQTomography,
        empi_dists_sequence: List[List[Tuple[int, np.array]]],
        is_computation_time_required: bool = False,
    ) -> Dict:
        """calculates sequence of estimate variables.

        see :func:`~quara.protocol.qtomography.standard.standard_qtomography_estimator.StandardQTomographyEstimator.calc_estimate_sequence_var`
        """
        if not qtomography.is_fullrank_matA():
            raise Exception

        A = qtomography.calc_matA()
        b = qtomography.calc_vecB()
        A_ddag = np.linalg.pinv(A.T @ A) @ A.T

        estimate_sequence = []
        comp_time_sequence = []
        for empi_dists in empi_dists_sequence:
            start_time = time.time()

            empi_dists_tmp = [empi_dist[1] for empi_dist in empi_dists]
            f = np.vstack(empi_dists_tmp).flatten()
            v = A_ddag @ (f - b)
            estimate_sequence.append(v)

            comp_time = time.time() - start_time
            comp_time_sequence.append(comp_time)

        value = {"estimate": estimate_sequence}
        if is_computation_time_required:
            value["computation_time"] = comp_time_sequence

        return value
