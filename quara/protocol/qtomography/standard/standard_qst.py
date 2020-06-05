import numpy as np

from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.qcircuit.experiment import Experiment


class StandardQst(StandardQTomography):
    def __init__(self, experiment: Experiment):
        super().__init__(experiment)

    def calc_coeff0(self, i: int, x: int) -> np.array:
        pass

    def calc_coeff1(self, i: int, x: int) -> np.array:
        pass
