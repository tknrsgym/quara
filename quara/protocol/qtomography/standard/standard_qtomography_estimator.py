from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
from quara.objects import qoperation

from quara.objects.qoperation import QOperation
from quara.protocol.qtomography.qtomography_estimator import (
    QTomographyEstimator,
    QTomographyEstimationResult,
)
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography


class StandardQTomographyEstimationResult(QTomographyEstimationResult):
    def __init__(
        self,
        estimated_var_sequence: List[np.ndarray],
        computation_times: List[float],
        template_qoperation: QOperation,
    ):
        super().__init__(computation_times)

        self._estimated_var_sequence: List[np.ndarray] = estimated_var_sequence
        self._template_qoperation: QOperation = template_qoperation

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
        var = self._estimated_var_sequence[0]
        qoperation = self._template_qoperation.generate_from_var(var)
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
            self._template_qoperation.generate_from_var(var)
            for var in self._estimated_var_sequence
        ]
        return qoperations


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
