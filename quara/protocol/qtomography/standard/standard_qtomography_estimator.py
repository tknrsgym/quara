from abc import abstractmethod
from typing import List

import numpy as np

from quara.objects.qoperation import QOperation
from quara.protocol.qtomography.estimator import Estimator
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography


class StandardQTomographyEstimator(Estimator):
    def __init__(self):
        super().__init__()

    def calc_estimate_qoperation(
        self, qtomography: StandardQTomography, empi_dists: List[List[float]]
    ) -> QOperation:
        var = self.calc_estimate_var(qtomography, empi_dists)
        qope = qtomography.convert_var_to_qoperation(var)
        return qope

    @abstractmethod
    def calc_estimate_var(
        self, qtomography: StandardQTomography, empi_dists: List[List[float]]
    ) -> np.array:
        raise NotImplementedError()
