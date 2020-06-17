import numpy as np

from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.protocol.qtomography.standard.standard_qtomography_estimator import (
    StandardQTomographyEstimator,
)


class LinearEstimator(StandardQTomographyEstimator):
    def __init__(self):
        super().__init__()

    def calc_estimate_var(
        self, qtomography: StandardQTomography, empi_dists: List[List[float]]
    ) -> np.array:
        pass
