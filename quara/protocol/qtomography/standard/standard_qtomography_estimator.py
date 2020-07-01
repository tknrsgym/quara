from abc import abstractmethod
from typing import List, Tuple

import numpy as np

from quara.objects.qoperation import QOperation
from quara.protocol.qtomography.estimator import Estimator
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography


class StandardQTomographyEstimator(Estimator):
    def __init__(self):
        super().__init__()

    def calc_estimate_qoperation(
        self, qtomography: StandardQTomography, empi_dists: List[Tuple[int, np.array]],
    ) -> QOperation:
        """calculates estimate QOperation.

        Parameters
        ----------
        qtomography : StandardQTomography
            StandardQTomography to calculates estimate QOperation. 
        empi_dists : List[Tuple[int, np.array]]
            empirical distributions to calculates estimate QOperation. 

        Returns
        -------
        QOperation
            estimate QOperation.
        """
        var = self.calc_estimate_var(qtomography, empi_dists)
        qope = qtomography.convert_var_to_qoperation(var)
        return qope

    def calc_estimate_sequence_qoperation(
        self,
        qtomography: StandardQTomography,
        empi_dists_sequence: List[List[Tuple[int, np.array]]],
    ) -> List[QOperation]:
        """calculates estimate QOperations.

        Parameters
        ----------
        qtomography : StandardQTomography
            StandardQTomography to calculates estimate QOperation. 
        empi_dists_sequence : List[List[Tuple[int, np.array]]]
            sequence of empirical distributions to calculates estimate QOperations. 

        Returns
        -------
        List[QOperation]
            estimate QOperations.
        """
        vars = self.calc_estimate_sequence_var(qtomography, empi_dists_sequence)
        qope_seq = []
        for var in vars:
            qope = qtomography.convert_var_to_qoperation(var)
            qope_seq.append(qope)
        return qope_seq

    @abstractmethod
    def calc_estimate_var(
        self, qtomography: StandardQTomography, empi_dists: List[Tuple[int, np.array]],
    ) -> np.array:
        """calculates estimate variables.

        this function must be implemented in the subclass.

        Parameters
        ----------
        qtomography : StandardQTomography
            StandardQTomography to calculates estimate variables. 
        empi_dists : List[Tuple[int, np.array]]
            empirical distributions to calculates estimate variables. 

        Returns
        -------
        np.array
            estimate variables.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def calc_estimate_sequence_var(
        self,
        qtomography: StandardQTomography,
        empi_dists_sequence: List[List[Tuple[int, np.array]]],
    ) -> np.array:
        """calculates sequence of estimate variables.

        this function must be implemented in the subclass.

        Parameters
        ----------
        qtomography : StandardQTomography
            StandardQTomography to calculates estimate variables. 
        empi_dists_sequence : List[List[Tuple[int, np.array]]]
            sequence of empirical distributions to calculates estimate variables. 

        Returns
        -------
        np.array
            sequence of empirical distributions.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()
