from typing import List

import numpy as np

from quara.protocol.qtomography.qtomography import QTomography
from quara.qcircuit.experiment import Experiment


class StandardQTomography(QTomography):
    def __init__(self, experiment: Experiment):
        super().__init__(experiment)
        self._set_coeff0s()
        self._set_coeff1s()

    def _set_coeff0s(self):
        pass

    def _set_coeff1s(self):
        pass

    def calc_coeff0(self, i: int, x: int) -> np.array:
        pass

    def calc_coeff1(self, i: int, x: int) -> np.array:
        pass

    def calc_matA(self) -> np.array:
        pass

    def calc_vecB(self) -> np.array:
        pass

    def is_standard(self) -> bool:
        pass

    def convert_var_to_qoperation(self, var: np.array):
        pass
