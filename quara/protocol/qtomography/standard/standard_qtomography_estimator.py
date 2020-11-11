from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from quara.objects.qoperation import QOperation
from quara.protocol.qtomography.qtomography_estimator import (
    QTomographyEstimator,
    QTomographyEstimationResult,
)
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography


class StandardQTomographyEstimationResult(QTomographyEstimationResult):
    def __init__(
        self,
        qtomography: StandardQTomography,
        data,
        estimated_var_sequence: List[np.array],
        computation_times: List[float],
    ):
        super().__init__(computation_times)

        self._qtomography: StandardQTomography = qtomography
        self._data = data
        self._estimated_var_sequence: List[np.array] = estimated_var_sequence

    @property
    def data(self):
        return self._data

    @property
    def estimated_var(self) -> np.array:
        """returns the estimate. the type of each estimate is ``np.array``.

        Returns
        -------
        np.array
            the estimate.
        """
        return self._estimated_var_sequence[0]

    @property
    def estimated_var_sequence(self) -> List[np.array]:
        """returns the estimate sequence. the type of each estimate is ``np.array``.

        Returns
        -------
        List[np.array]
            the estimate sequence.
        """
        return self._estimated_var_sequence

    @property
    def estimated_qoperation(self) -> QOperation:
        """returns the estimate. the type of each estimate is ``QOperation``.

        Returns
        -------
        QOperation
            the estimate.
        """
        qoperation = self._qtomography.convert_var_to_qoperation(
            self._estimated_var_sequence[0]
        )
        return qoperation

    @property
    def estimated_qoperation_sequence(self) -> List[QOperation]:
        """returns the estimate sequence. the type of each estimate is ``QOperation``.

        Returns
        -------
        List[QOperation]
            the estimate sequence.
        """
        qoperations = [
            self._qtomography.convert_var_to_qoperation(var)
            for var in self._estimated_var_sequence
        ]
        return qoperations

    @property
    def data(self) -> List[Tuple[int, np.array]]:
        return self._data

    @property
    def num_data(self) -> List[int]:
        num_data = [data[0][0] for data in self._data]
        return num_data


class StandardQTomographyEstimator(QTomographyEstimator):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def calc_estimate(
        self,
        qtomography: StandardQTomography,
        empi_dists: List[Tuple[int, np.array]],
        is_computation_time_required: bool = False,
    ) -> StandardQTomographyEstimationResult:
        """calculates estimate variables.

        this function must be implemented in the subclass.

        Parameters
        ----------
        qtomography : StandardQTomography
            StandardQTomography to calculates estimate variables. 
        empi_dists : List[Tuple[int, np.array]]
            empirical distributions to calculates estimate variables. 
        is_computation_time_required : bool, optional
            whether to include computation time in the return value or not, by default False.

        Returns
        -------
        Dict
            the return value forms the following dict:
            {
                "estimate": estimate variables(type=np.array),
                "computation_time": computation time(seconds, type=float),
            }

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def calc_estimate_sequence(
        self,
        qtomography: StandardQTomography,
        empi_dists_sequence: List[List[Tuple[int, np.array]]],
        is_computation_time_required: bool = False,
    ) -> StandardQTomographyEstimationResult:
        """calculates sequence of estimate variables.

        this function must be implemented in the subclass.

        Parameters
        ----------
        qtomography : StandardQTomography
            StandardQTomography to calculates estimate variables. 
        empi_dists_sequence : List[List[Tuple[int, np.array]]]
            sequence of empirical distributions to calculates estimate variables. 
        is_computation_time_required : bool, optional
            whether to include computation time in the return value or not, by default False.

        Returns
        -------
        Dict
            the return value forms the following dict:
            {
                "estimate": sequence of estimate variables(type=List[np.array]),
                "computation_time": computation time(seconds, type=List[float]),
            }
        np.array
            sequence of empirical distributions.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()
