from abc import abstractmethod
from typing import List

import numpy as np

from quara.objects.qoperation import QOperation
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

    @abstractmethod
    def calc_coeff0(self, i: int, x: int) -> np.array:
        raise NotImplementedError()

    @abstractmethod
    def calc_coeff1(self, i: int, x: int) -> np.array:
        raise NotImplementedError()

    @abstractmethod
    def calc_matA(self) -> np.array:
        raise NotImplementedError()

    @abstractmethod
    def calc_vecB(self) -> np.array:
        raise NotImplementedError()

    @abstractmethod
    def is_standard(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def convert_var_to_qoperation(self, var: np.array) -> QOperation:
        raise NotImplementedError()

    def is_fullrank_matA(self) -> bool:
        matA = self.calc_matA()
        row = matA.shape[0]
        rank = np.linalg.matrix_rank(matA)
        return row == rank
