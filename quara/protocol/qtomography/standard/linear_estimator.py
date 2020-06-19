import itertools
from typing import List

import numpy as np

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
        empi_dists: List[List[float]],
        on_para_eq_constraint: bool = True,
    ) -> List[np.array]:
        pass

    def calc_estimates_sequence_var(
        self,
        qtomography: StandardQTomography,
        empi_dists_sequence: List[List[float]],
        on_para_eq_constraint: bool = True,
    ) -> List[np.array]:
        stacked_empi_dists_seq = list(
            itertools.chain.from_iterable(empi_dists_sequence)
        )
        print(stacked_empi_dists_seq)

