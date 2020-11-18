from abc import abstractmethod, abstractproperty
from typing import List, Tuple

import numpy as np

from quara.objects.qoperation import QOperation
from quara.objects.qoperations import SetQOperations
from quara.protocol.qtomography.qtomography import QTomography
from quara.qcircuit.experiment import Experiment
from quara.utils import matrix_util


class StandardQTomography(QTomography):
    def __init__(
        self, experiment: Experiment, set_qoperations: SetQOperations,
    ):
        """initialize standard quantum tomography class.

        To inherit from this class, set the following instance variables in the constructor of the subclass.

        - ``_coeffs_0th``: return value of ``get_coeffs_0th`` function.
        - ``_coeffs_1st``: return value of ``get_coeffs_1st`` function.
        - ``_map_experiment_to_setqoperations``: a map from indices of Experiment to indices of SetQOperations. 
            if you map the 0th state to the 1st state, set ``{("state", 0): ("state", 1)}``.
        - ``_map_setqoperations_to_experiment``: a map from indices of SetQOperations to indices of Experiment. 

        Parameters
        ----------
        experiment : Experiment
            Experiment class used in quantum tomography.
        set_qoperations : SetQOperations
            SetQOperations class used in quantum tomography.
        """
        super().__init__(experiment, set_qoperations)
        self._coeffs_0th = None
        self._coeffs_1st = None

    def get_coeffs_0th(self, schedule_index: int, x: int) -> np.float64:
        """returns 0th coefficients specified by schedule index and povm vecs index

        Parameters
        ----------
        schedule_index : int
            schedule index.
        x : int
            povm vecs index.

        Returns
        -------
        np.float64
            0th coefficients.
        """
        return self._coeffs_0th[(schedule_index, x)]

    def get_coeffs_1st(self, schedule_index: int, x: int) -> np.array:
        """returns 1st coefficients specified by schedule index and povm vecs index

        Parameters
        ----------
        schedule_index : int
            schedule index.
        x : int
            povm vecs index.

        Returns
        -------
        np.array
            1st coefficients.
        """
        return self._coeffs_1st[(schedule_index, x)]

    def calc_matA(self) -> np.array:
        """returns the matrix A.

        the matrix A is a stack of 1st coefficients.

        Returns
        -------
        np.array
            the matrix A.
        """
        sorted_coeffs_1st = sorted(self._coeffs_1st.items())
        sorted_values = [k[1] for k in sorted_coeffs_1st]
        matA = np.vstack(sorted_values)
        return matA

    def calc_vecB(self) -> np.array:
        """returns the vector B.

        the vector B is a stack of 0th coefficients.

        Returns
        -------
        np.array
            the vector B.
        """
        sorted_coeffs_0th = sorted(self._coeffs_0th.items())
        sorted_values = [k[1] for k in sorted_coeffs_0th]
        vecB = np.vstack(sorted_values).flatten()
        return vecB

    def is_fullrank_matA(self) -> bool:
        """returns whether matrix A is full rank.

        Returns
        -------
        bool
            True where matrix A is full rank, False otherwise.
        """
        matA = self.calc_matA()
        rank = np.linalg.matrix_rank(matA)
        size = min(matA.shape)
        return size == rank

    @abstractmethod
    def convert_var_to_qoperation(self, var: np.array) -> QOperation:
        """converts variable to QOperation.

        this function must be implemented in the subclass.

        Parameters
        ----------
        var : np.array
            variables.

        Returns
        -------
        QOperation
            converted QOperation.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def generate_empty_estimation_obj_with_setting_info(self) -> QOperation:
        """generates the empty estimation object with setting information.

        Returns
        -------
        QOperation
            the empty estimation object(QOperation) with setting information.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    def calc_prob_dist(self, qope: QOperation, schedule_index: int) -> List[float]:
        """calculates a probability distribution.
        
        see :func:`~quara.protocol.qtomography.qtomography.QTomography.calc_prob_dist`
        """
        prob_dists = self.calc_prob_dists(qope)
        return prob_dists[schedule_index]

    def calc_prob_dists(self, qope: QOperation) -> List[List[float]]:
        """calculates probability distributions.
        
        see :func:`~quara.protocol.qtomography.qtomography.QTomography.calc_prob_dists`
        """
        tmp_prob_dists = self.calc_matA() @ qope.to_var() + self.calc_vecB()
        prob_dists = tmp_prob_dists.reshape((self.num_schedules, -1))

        return prob_dists

    def calc_covariance_mat_single(
        self, qope: QOperation, schedule_index: int, data_num: int
    ) -> np.array:
        """calculates covariance matrix of single probability distribution.

        Parameters
        ----------
        qope : QOperation
            QOperation to calculate covariance matrix of single probability distribution.
        schedule_index : int
            schedule index.
        data_num : int
            number of data.

        Returns
        -------
        np.array
            covariance matrix of single probability distribution.
        """
        prob_dist = self.calc_prob_dist(qope, schedule_index)
        val = matrix_util.calc_covariance_mat(prob_dist, data_num)
        return val

    def calc_covariance_mat_total(
        self, qope: QOperation, data_num_list: List[int]
    ) -> np.array:
        """calculates covariance matrix of total probability distributions.

        Parameters
        ----------
        qope : QOperation
            QOperation to calculate covariance matrix of total probability distributions.
        data_num_list : List[int]
            list of number of data.

        Returns
        -------
        np.array
            covariance matrix of total probability distributions.
        """
        matrices = []
        for schedule_index in range(self.num_schedules):
            mat_single = self.calc_covariance_mat_single(
                qope, schedule_index, data_num_list[schedule_index]
            )
            matrices.append(mat_single)

        val = matrix_util.calc_direct_sum(matrices)
        return val

    def calc_covariance_linear_mat_total(
        self, qope: QOperation, data_num_list: List[int]
    ) -> np.array:
        """calculates covariance matrix of linear estimate of probability distributions.

        Parameters
        ----------
        qope : QOperation
            QOperation to calculate covariance matrix of linear estimate of probability distributions.
        data_num_list : List[int]
            list of number of data.

        Returns
        -------
        np.array
            covariance matrix of linear estimate of probability distributions.
        """
        A_inv = matrix_util.calc_left_inv(self.calc_matA())
        val = matrix_util.calc_conjugate(
            A_inv, self.calc_covariance_mat_total(qope, data_num_list)
        )
        return val

    def calc_mse_linear_analytical(
        self, qope: QOperation, data_num_list: List[int]
    ) -> np.float64:
        """calculates mean squared error of linear estimate of probability distributions.

        Parameters
        ----------
        qope : QOperation
            QOperation to calculate mean squared error of linear estimate of probability distributions.
        data_num_list : List[int]
            list of number of data.

        Returns
        -------
        np.float64
            mean squared error of linear estimate of probability distributions.
        """
        val = np.trace(self.calc_covariance_linear_mat_total(qope, data_num_list))
        return val

    def calc_mse_empi_dists_analytical(
        self, qope: QOperation, data_num_list: List[int]
    ) -> np.float64:
        """calculates analytical solution of mean squared error of empirical distributions.

        Parameters
        ----------
        qope : QOperation
            QOperation to calculate analytical solution of mean squared error of empirical distributions.
        data_num_list : List[int]
            list of number of data.

        Returns
        -------
        np.float64
            analytical solution of mean squared error of empirical distributions.
        """

        mse_total = 0.0
        for schedule_index, data_num in enumerate(data_num_list):
            mse_total += np.trace(
                self.calc_covariance_mat_single(qope, schedule_index, data_num)
            )
        return mse_total

    @abstractmethod
    def _get_target_index(self, experiment: Experiment, schedule_index: int) -> int:
        raise NotImplementedError()

    @abstractproperty
    def _estimated_qoperation_type(cls):
        raise NotImplementedError()

    def generate_prob_dists_sequence(
        self, true_object: QOperation
    ) -> List[List[Tuple[int, np.array]]]:
        tmp_experiment = self._experiment.copy()
        attribute_name = (
            self.__class__._estimated_qoperation_type.__name__.lower() + "s"
        )
        for schedule_index in range(len(tmp_experiment.schedules)):
            target_index = self._get_target_index(tmp_experiment, schedule_index)
            getattr(tmp_experiment, attribute_name)[target_index] = true_object

        prob_dists_sequence_tmp = tmp_experiment.calc_prob_dists()

        return prob_dists_sequence_tmp
