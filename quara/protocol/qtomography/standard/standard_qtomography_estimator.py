from abc import abstractmethod
from typing import Any, Dict, List, Tuple

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
        estimated_var_sequence: List[np.ndarray],
        computation_times: List[float],
    ):
        super().__init__(computation_times)

        self._qtomography: StandardQTomography = qtomography
        self._data = data
        self._estimated_var_sequence: List[np.ndarray] = estimated_var_sequence

    @property
    def qtomography(self) -> StandardQTomography:
        """returns the StandardQTomography used for estimation.

        Returns
        -------
        StandardQTomography
            the StandardQTomography used for estimation.
        """
        return self._qtomography

    @property
    def data(self) -> Any:
        """returns the data used for estimation.

        Returns
        -------
        Any
            returns the data used for estimation.
        """
        return self._data

    @property
    def estimated_var(self) -> np.ndarray:
        """returns the estimate. the type of each estimate is ``np.ndarray``.

        Returns
        -------
        np.ndarray
            the estimate.
        """
        return self._estimated_var_sequence[0]

    @property
    def estimated_var_sequence(self) -> List[np.ndarray]:
        """returns the estimate sequence. the type of each estimate is ``np.ndarray``.

        Returns
        -------
        List[np.ndarray]
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
        empi_dists: List[Tuple[int, np.ndarray]],
        is_computation_time_required: bool = False,
    ) -> StandardQTomographyEstimationResult:
        """calculates estimate variables.

        this function must be implemented in the subclass.

        Parameters
        ----------
        qtomography : StandardQTomography
            StandardQTomography to calculates estimate variables.
        empi_dists : List[Tuple[int, np.ndarray]]
            empirical distributions to calculates estimate variables.
        is_computation_time_required : bool, optional
            whether to include computation time in the return value or not, by default False.

        Returns
        -------
        StandardQTomographyEstimationResult
            estimation result.

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
        empi_dists_sequence: List[List[Tuple[int, np.ndarray]]],
        is_computation_time_required: bool = False,
    ) -> StandardQTomographyEstimationResult:
        """calculates sequence of estimate variables.

        this function must be implemented in the subclass.

        Parameters
        ----------
        qtomography : StandardQTomography
            StandardQTomography to calculates estimate variables.
        empi_dists_sequence : List[List[Tuple[int, np.ndarray]]]
            sequence of empirical distributions to calculates estimate variables.
        is_computation_time_required : bool, optional
            whether to include computation time in the return value or not, by default False.

        Returns
        -------
        StandardQTomographyEstimationResult
            estimation result.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

