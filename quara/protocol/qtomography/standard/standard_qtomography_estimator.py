from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from quara.objects.qoperation import QOperation
from quara.protocol.qtomography.estimator import Estimator, EstimationResult
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography


class StandardQTomographyEstimationResult(EstimationResult):
    def __init__(
        self,
        qtomography: StandardQTomography,
        data,
        computation_times: List[float],
        estimates: List,
    ):
        # TODO
        pass


class StandardQTomographyEstimator(Estimator):
    def __init__(self):
        super().__init__()

    def calc_estimate_qoperation(
        self,
        qtomography: StandardQTomography,
        empi_dists: List[Tuple[int, np.array]],
        is_computation_time_required: bool = False,
    ) -> Dict:
        """calculates estimate QOperation.

        Parameters
        ----------
        qtomography : StandardQTomography
            StandardQTomography to calculates estimate QOperation. 
        empi_dists : List[Tuple[int, np.array]]
            empirical distributions to calculates estimate QOperation. 
        is_computation_time_required : bool, optional
            whether to include computation time in the return value or not, by default False.

        Returns
        -------
        Dict
            the return value forms the following dict:
            {
                "estimate": estimate QOperation(type=QOperation),
                "computation_time": computation time(seconds, type=float),
            }
        """
        value = self.calc_estimate_var(
            qtomography,
            empi_dists,
            is_computation_time_required=is_computation_time_required,
        )
        value["estimate"] = qtomography.convert_var_to_qoperation(value["estimate"])
        return value

    def calc_estimate_sequence_qoperation(
        self,
        qtomography: StandardQTomography,
        empi_dists_sequence: List[List[Tuple[int, np.array]]],
        is_computation_time_required: bool = False,
    ) -> Dict:
        """calculates estimate QOperations.

        Parameters
        ----------
        qtomography : StandardQTomography
            StandardQTomography to calculates estimate QOperation. 
        empi_dists_sequence : List[List[Tuple[int, np.array]]]
            sequence of empirical distributions to calculates estimate QOperations. 
        is_computation_time_required : bool, optional
            whether to include computation time in the return value or not, by default False.

        Returns
        -------
        Dict
            the return value forms the following dict:
            {
                "estimate": sequence of estimate QOperations(type=List[QOperation]),
                "computation_time": sequence of computation times(seconds, type=List[float]),
            }
        """
        value = self.calc_estimate_sequence_var(
            qtomography,
            empi_dists_sequence,
            is_computation_time_required=is_computation_time_required,
        )
        qope_seq = []
        for var in value["estimate"]:
            qope = qtomography.convert_var_to_qoperation(var)
            qope_seq.append(qope)

        value["estimate"] = qope_seq
        return value

    @abstractmethod
    def calc_estimate_var(
        self,
        qtomography: StandardQTomography,
        empi_dists: List[Tuple[int, np.array]],
        is_computation_time_required: bool = False,
    ) -> Dict:
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
    def calc_estimate_sequence_var(
        self,
        qtomography: StandardQTomography,
        empi_dists_sequence: List[List[Tuple[int, np.array]]],
        is_computation_time_required: bool = False,
    ) -> Dict:
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
