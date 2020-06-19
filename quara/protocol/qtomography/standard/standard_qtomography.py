from abc import abstractmethod
from typing import List

import numpy as np

from quara.objects.qoperation import QOperation
from quara.objects.qoperations import SetQOperations
from quara.protocol.qtomography.qtomography import QTomography
from quara.qcircuit.experiment import Experiment


class StandardQTomography(QTomography):
    def __init__(
        self, experiment: Experiment, set_qoperations: SetQOperations,
    ):
        super().__init__(experiment, set_qoperations)
        self._coeffs_0th = None
        self._coeffs_1st = None

    def get_coeffs_0th(self, schedule_index: int, x: int) -> np.float64:
        return self._coeffs_0th[(schedule_index, x)]

    def get_coeffs_1st(self, schedule_index: int, x: int) -> np.array:
        return self._coeffs_1st[(schedule_index, x)]

    def calc_matA(self) -> np.array:
        sorted_coeffs_1st = sorted(self._coeffs_1st.items())
        sorted_values = [k[1] for k in sorted_coeffs_1st]
        matA = np.vstack(sorted_values)
        return matA

    def calc_vecB(self) -> np.array:
        sorted_coeffs_0th = sorted(self._coeffs_0th.items())
        sorted_values = [k[1] for k in sorted_coeffs_0th]
        vecB = np.vstack(sorted_values)
        return vecB

    @abstractmethod
    def convert_var_to_qoperation(self, var: np.array) -> QOperation:
        raise NotImplementedError()

    def is_fullrank_matA(self) -> bool:
        matA = self.calc_matA()
        row = matA.shape[0]
        rank = np.linalg.matrix_rank(matA)
        return row == rank
