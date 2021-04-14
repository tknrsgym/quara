from abc import abstractmethod, abstractproperty
from typing import List, Tuple, Union

import numpy as np

from quara.objects.qoperation import QOperation
from quara.objects.qoperations import SetQOperations
from quara.protocol.qtomography.qtomography import QTomography
from quara.qcircuit.experiment import Experiment
from quara.utils import matrix_util


class StandardQTomography(QTomography):
    def __init__(
        self,
        experiment: Experiment,
        set_qoperations: SetQOperations,
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
        """returns 0th coefficients specified by schedule index and measurement outcome index

        Parameters
        ----------
        schedule_index : int
            schedule index.
        x : int
            measurement outcome index.

        Returns
        -------
        np.float64
            0th coefficients.
        """
        return self._coeffs_0th[(schedule_index, x)]

    def get_coeffs_0th_vec(self, schedule_index: int) -> np.ndarray:
        """returns 0th coefficient vector specified by schedule index.

        Parameters
        ----------
        schedule_index : int
            schedule index.

        Returns
        -------
        np.ndarray( , dtype=np.float64)
            0th coefficients vector
        """
        l = []
        xs = [key[1] for key in self._coeffs_0th.keys() if key[0] == schedule_index]
        for x in xs:
            coeffs_0th = self.get_coeffs_0th(schedule_index, x)
            l.append(coeffs_0th)
        return np.array(l, dtype=np.float64)

    def get_coeffs_1st(self, schedule_index: int, x: int) -> np.ndarray:
        """returns 1st coefficients specified by schedule index and measurement outcome index

        Parameters
        ----------
        schedule_index : int
            schedule index.
        x : int
            measurement outcome index.

        Returns
        -------
        np.ndarray
            1st coefficients.
        """
        return self._coeffs_1st[(schedule_index, x)]

    def get_coeffs_1st_mat(self, schedule_index: int) -> np.ndarray:
        """returns 1st coefficient matrix specified by schedule index.

        Parameters
        ----------
        schedule_index : int
            schedule index.

        Returns
        -------
        np.ndarray
            1st coefficient matrix.
        """
        ll = []
        xs = [key[1] for key in self._coeffs_0th.keys() if key[0] == schedule_index]
        for x in xs:
            coeffs_1st = self.get_coeffs_1st(schedule_index, x)
            ll.append(coeffs_1st)
        return np.stack(ll)

    def calc_matA(self) -> np.ndarray:
        """returns the matrix A.

        the matrix A is a stack of 1st coefficients.

        Returns
        -------
        np.ndarray
            the matrix A.
        """
        sorted_coeffs_1st = sorted(self._coeffs_1st.items())
        sorted_values = [k[1] for k in sorted_coeffs_1st]
        matA = np.vstack(sorted_values)
        return matA

    def calc_vecB(self) -> np.ndarray:
        """returns the vector B.

        the vector B is a stack of 0th coefficients.

        Returns
        -------
        np.ndarray
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
    def num_outcomes(self, schedule_index: int) -> int:
        """returns the number of outcomes of probability distribution of a schedule index.

        Parameters
        ----------
        schedule_index: int

        Returns
        -------
        int
            the number of outcomes
        """
        raise NotImplementedError()

    @abstractmethod
    def convert_var_to_qoperation(self, var: np.ndarray) -> QOperation:
        """converts variable to QOperation.

        this function must be implemented in the subclass.

        Parameters
        ----------
        var : np.ndarray
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
        if self._on_para_eq_constraint:
            tmp_prob_dists = self.calc_matA() @ qope.to_var() + self.calc_vecB()
        else:
            tmp_prob_dists = (
                self.calc_matA() @ qope.to_stacked_vector() + self.calc_vecB()
            )
        prob_dists = tmp_prob_dists.reshape((self.num_schedules, -1))

        return prob_dists

    def calc_covariance_mat_single(
        self, qope: QOperation, schedule_index: int, data_num: int
    ) -> np.ndarray:
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
        np.ndarray
            covariance matrix of single probability distribution.
        """
        prob_dist = self.calc_prob_dist(qope, schedule_index)
        val = matrix_util.calc_covariance_mat(prob_dist, data_num)
        return val

    def calc_covariance_mat_total(
        self, qope: QOperation, data_num_list: List[int]
    ) -> np.ndarray:
        """calculates covariance matrix of total probability distributions.

        Parameters
        ----------
        qope : QOperation
            QOperation to calculate covariance matrix of total probability distributions.
        data_num_list : List[int]
            list of number of data.

        Returns
        -------
        np.ndarray
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
    ) -> np.ndarray:
        """calculates covariance matrix of linear estimate of probability distributions.

        Parameters
        ----------
        qope : QOperation
            QOperation to calculate covariance matrix of linear estimate of probability distributions.
        data_num_list : List[int]
            list of number of data.

        Returns
        -------
        np.ndarray
            covariance matrix of linear estimate of probability distributions.
        """
        A_inv = matrix_util.calc_left_inv(self.calc_matA())
        val = matrix_util.calc_conjugate(
            A_inv, self.calc_covariance_mat_total(qope, data_num_list)
        )
        return val

    def _calc_mse_linear_analytical_mode_var(
        self, qope: QOperation, data_num_list: List[int]
    ) -> np.float64:
        val = np.trace(self.calc_covariance_linear_mat_total(qope, data_num_list))
        return val

    def _calc_mse_linear_analytical_mode_qoperation(
        self, qope: QOperation, data_num_list: List[int]
    ) -> np.float64:
        return self._calc_mse_linear_analytical_mode_var(qope, data_num_list)

    def calc_mse_linear_analytical(
        self, qope: QOperation, data_num_list: List[int], mode: str = "qoperation"
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
        if mode == "qoperation":
            val = self._calc_mse_linear_analytical_mode_qoperation(qope, data_num_list)
        elif mode == "var":
            val = self._calc_mse_linear_analytical_mode_var(qope, data_num_list)
        else:
            error_message = "​The argument `mode` must be `qoperation` or `var`"
            raise ValueError(error_message)

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

    def calc_fisher_matrix(
        self, j: int, var: Union[QOperation, np.ndarray]
    ) -> np.ndarray:
        """calculates Fisher matrix of one schedule.

        Parameters
        ----------
        j : int
            schedule_index
        var : Union[QOperation, np.ndarray]
            variables to calculate Fisher matrix of one schedule.

        Returns
        -------
        np.ndarray
            Fisher matrix of one schedule.
        """
        if isinstance(var, QOperation):
            var = var.to_var()

        matA = self.calc_matA()
        vecB = self.calc_vecB()
        size_prob_dist = int(len(matA) / self.num_schedules)
        prob_dist = (
            matA[size_prob_dist * j : size_prob_dist * (j + 1)] @ var
            + vecB[size_prob_dist * j : size_prob_dist * (j + 1)]
        )
        grad_prob_dist = matA[size_prob_dist * j : size_prob_dist * (j + 1)]
        fisher_matrix = matrix_util.calc_fisher_matrix(prob_dist, grad_prob_dist)

        return fisher_matrix

    def calc_fisher_matrix_total(
        self, var: Union[QOperation, np.ndarray], weights: List[float]
    ) -> np.ndarray:
        """calculates Fisher matrix of the total schedule.

        Parameters
        ----------
        var : Union[QOperation, np.ndarray]
            variables to calculate Fisher matrix of one schedule.
        weights : List[float]
            weights to calculate Fisher matrix of one schedule.

        Returns
        -------
        np.ndarray
            Fisher matrix of the total schedule.
        """
        fisher_matrices = []
        for schedule_index in range(self.num_schedules):
            fisher_matrices.append(
                weights[schedule_index] * self.calc_fisher_matrix(schedule_index, var)
            )
        return sum(fisher_matrices)

    def calc_cramer_rao_bound(
        self, var: Union[QOperation, np.ndarray], N: int, list_N: List[int]
    ) -> np.ndarray:
        """calculates Cramer-Rao bound.

        Parameters
        ----------
        var : Union[QOperation, np.ndarray]
            variables to calculate Cramer-Rao bound.
        N : int
            representative value of the number of data.
        list_N : List[int]
            the number of data for each schedule.

        Returns
        -------
        np.ndarray
            Cramer-Rao bound.
        """
        return self._calc_cramer_rao_bound(var, N, list_N)

    def _calc_cramer_rao_bound(
        self, var: Union[QOperation, np.ndarray], N: int, list_N: List[int]
    ) -> np.ndarray:
        weights = [tmp_N / N for tmp_N in list_N]
        fisher = self.calc_fisher_matrix_total(var, weights)
        val = np.trace(np.linalg.inv(fisher)) / N
        return val

    @abstractmethod
    def _get_target_index(self, experiment: Experiment, schedule_index: int) -> int:
        raise NotImplementedError()

    @abstractproperty
    def _estimated_qoperation_type(cls):
        raise NotImplementedError()

    def generate_prob_dists_sequence(
        self, true_object: QOperation
    ) -> List[List[Tuple[int, np.ndarray]]]:
        tmp_experiment = self._experiment.copy()
        attribute_name = (
            self.__class__._estimated_qoperation_type.__name__.lower() + "s"
        )
        for schedule_index in range(len(tmp_experiment.schedules)):
            target_index = self._get_target_index(tmp_experiment, schedule_index)
            getattr(tmp_experiment, attribute_name)[target_index] = true_object

        prob_dists_sequence_tmp = tmp_experiment.calc_prob_dists()

        return prob_dists_sequence_tmp

    def _validate_schedules_str(self, schedules: str) -> None:
        supported_schedule_strs = ["all"]
        if schedules not in supported_schedule_strs:
            message = f"The string specified in schedules must be one of the following, not '{schedules}': {supported_schedule_strs}"
            raise ValueError(message)
