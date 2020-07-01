import itertools
from typing import List, Tuple

import numpy as np

from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.protocol.qtomography.standard.standard_qtomography_estimator import (
    StandardQTomographyEstimator,
)


class LinearEstimator(StandardQTomographyEstimator):
    def __init__(self):
        super().__init__()

    def calc_estimate_var(
        self, qtomography: StandardQTomography, empi_dists: List[Tuple[int, np.array]],
    ) -> np.array:
        estimate = self.calc_estimate_sequence_var(qtomography, [empi_dists])
        return estimate[0]

    def calc_estimate_sequence_var(
        self,
        qtomography: StandardQTomography,
        empi_dists_sequence: List[List[Tuple[int, np.array]]],
    ) -> np.array:
        if not qtomography.is_fullrank_matA():
            raise Exception

        A = qtomography.calc_matA()
        b = qtomography.calc_vecB()
        A_ddag = np.linalg.pinv(A.T @ A) @ A.T

        estimate_sequence = []
        for empi_dists in empi_dists_sequence:
            empi_dists_tmp = [empi_dist[1] for empi_dist in empi_dists]
            f = np.vstack(empi_dists_tmp).flatten()
            v = A_ddag @ (f - b)
            estimate_sequence.append(v)

        return estimate_sequence
