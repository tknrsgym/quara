from abc import abstractmethod
from typing import Callable, List, Tuple

import numpy as np

from quara.loss_function.loss_function import LossFunctionOption
from quara.loss_function.probability_based_loss_function import (
    ProbabilityBasedLossFunction,
    ProbabilityBasedLossFunctionOption,
)
from quara.math.matrix import multiply_veca_vecb, multiply_veca_vecb_matc
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.utils import matrix_util


class WeightedProbabilityBasedSquaredErrorOption(ProbabilityBasedLossFunctionOption):
    def __init__(
        self, mode_weight: str = None, weights: List = None, weight_name: str = None
    ):
        """Constructor

        mode_weight should be the following value:

        - "identity" then uses identity matrices for weights.
        - "custom" then uses user custom matrices for weights.
        - "inverse_sample_covariance" then uses inverse matrices of Sample Covariance Matrices.
        - "inverse_unbiased_covariance" then uses inverse matrices of Unbiased Covariance Matrices.

        Parameters
        ----------
        mode_weight : str, optional
            mode_weight string, by default None
        weights : List, optional
            values of weight, by default None
        weight_name : str, optional
            weight_name string, by default None

        Raises
        ------
        ValueError
            unsupported ``mode_weight``
        """
        if weights is not None:
            mode_weight = "custom"

        if not mode_weight in [
            "identity",
            "custom",
            "inverse_sample_covariance",
            "inverse_unbiased_covariance",
            "unbiased_inverse_covariance",
        ]:
            raise ValueError(f"unsupported mode_weight={mode_weight}")

        super().__init__(
            mode_weight=mode_weight, weights=weights, weight_name=weight_name
        )


class WeightedProbabilityBasedSquaredError(ProbabilityBasedLossFunction):
    def __init__(
        self,
        num_var: int = None,
        func_prob_dists: List = None,
        func_gradient_prob_dists: List = None,
        func_hessian_prob_dists: List = None,
        prob_dists_q: List[np.ndarray] = None,
        weight_matrices: List[np.ndarray] = None,
    ):
        """Constructor

        Parameters
        ----------
        num_var : int, optional
            number of variables, by default None
        func_prob_dists : List[Callable[[np.ndarray], np.ndarray]], optional
            functions map variables to a probability distribution.
        func_gradient_prob_dists : List[Callable[[int, np.ndarray], np.ndarray]], optional
            functions map variables and an index of variables to gradient of probability distributions.
        func_hessian_prob_dists : List[Callable[[int, int, np.ndarray], np.ndarray]], optional
            functions map variables and indices of variables to Hessian of probability distributions.
        prob_dists_q : List[np.ndarray], optional
            vectors of ``q``, by default None.
        weight_matrices : List[np.ndarray], optional
            weight matrices, by default None
        """
        super().__init__(
            num_var,
            func_prob_dists,
            func_gradient_prob_dists,
            func_hessian_prob_dists,
            prob_dists_q,
        )

        # validate
        self._validate_weight_matrices(weight_matrices)
        self._weight_matrices = weight_matrices

        # update on_value, on_gradient and on_hessian
        self._update_on_value_true()
        self._update_on_gradient_true()
        self._update_on_hessian_true()

    def _validate_weight_matrices(self, weight_matrices: List[np.ndarray]) -> None:
        if weight_matrices:
            for index, weight_matrix in enumerate(weight_matrices):
                # weight_matrices are real matrices
                if weight_matrix.dtype != np.float64:
                    raise ValueError(
                        f"entries of weight_matrices must be real numbers. dtype of weight_matrices[{index}] is {weight_matrix.dtype}"
                    )

                # weight_matrices are symmetric matrices
                if not matrix_util.is_hermitian(weight_matrix):
                    raise ValueError(
                        f"weight_matrices must be symmetric. dtype of weight_matrices[{index}] is not symmetric"
                    )

    @property
    def weight_matrices(self) -> List[np.ndarray]:
        """returns weight matrices.

        Returns
        -------
        List[np.ndarray]
            weight matrices.
        """
        return self._weight_matrices

    def set_weight_matrices(self, weight_matrices: List[np.ndarray]) -> None:
        """sets weight matrices.

        Parameters
        ----------
        weight_matrices : List[np.ndarray]
            weight matrices.
        """
        self._validate_weight_matrices(weight_matrices)
        self._weight_matrices = weight_matrices

    def _set_weights_by_mode(
        self, mode_weight: str, data: List[Tuple[int, np.ndarray]]
    ) -> None:
        if mode_weight == "identity":
            pass
        elif mode_weight == "custom":
            self.set_weight_matrices(self.option.weights)
        elif (
            mode_weight == "inverse_sample_covariance"
            or mode_weight == "inverse_unbiased_covariance"
        ):
            weight_matrices = []
            for (num_data, empi_dist_original) in data:
                empi_dist = matrix_util.replace_prob_dist(empi_dist_original)

                # calc covariance matrix
                if mode_weight == "inverse_sample_covariance":
                    covariance_mat = matrix_util.calc_covariance_mat(
                        empi_dist, num_data
                    )
                else:
                    covariance_mat = matrix_util.calc_covariance_mat(
                        empi_dist, num_data - 1
                    )

                # calc inverse of covariance matrix
                weight_matrix = np.zeros(covariance_mat.shape)
                (row, col) = covariance_mat.shape
                extracted_mat = covariance_mat[:-1, :-1] + np.eye(row - 1) / (
                    num_data ** (3 / 2)
                )

                extracted_mat_inv = np.linalg.inv(extracted_mat)
                if row == 2 and col == 2:
                    weight_matrix[0, 0] = extracted_mat_inv[0, 0]
                else:
                    weight_matrix[:row, :col] = extracted_mat_inv
                weight_matrices.append(weight_matrix)

            self.set_weight_matrices(weight_matrices)

    def _update_on_value_true(self) -> bool:
        """validates and updates ``on_value`` to True.

        see :func:`~quara.data_analysis.loss_function.LossFunction._update_on_value_true`
        """
        if self.on_func_prob_dists is True and self.on_prob_dists_q is True:
            self._set_on_value(True)
        return self.on_value

    def _update_on_gradient_true(self) -> bool:
        """validates and updates ``on_gradient`` to True.

        see :func:`~quara.data_analysis.loss_function.LossFunction._update_on_gradient_true`
        """
        if (
            self.on_func_prob_dists is True
            and self.on_func_gradient_prob_dists is True
            and self.on_prob_dists_q is True
        ):
            self._set_on_gradient(True)
        return self.on_gradient

    def _update_on_hessian_true(self) -> bool:
        """validates and updates ``on_hessian`` to True.

        see :func:`~quara.data_analysis.loss_function.LossFunction._update_on_hessian_true`
        """
        if (
            self.on_func_prob_dists is True
            and self.on_func_gradient_prob_dists is True
            and self.on_func_hessian_prob_dists is True
            and self.on_prob_dists_q is True
        ):
            self._set_on_hessian(True)
        return self.on_hessian

    def value(self, var: np.ndarray) -> np.float64:
        """returns the value of Weighted Probability Based Squared Error.

        see :func:`~quara.data_analysis.loss_function.LossFunction.value`
        """
        tmp_values = []
        for index in range(len(self.func_prob_dists)):
            vec = self.func_prob_dists[index](var) - self.prob_dists_q[index]
            if self.weight_matrices:
                tmp_value = multiply_veca_vecb_matc(
                    vec, vec, self.weight_matrices[index]
                )
            else:
                tmp_value = multiply_veca_vecb(vec, vec)
            tmp_values.append(tmp_value)

        val = np.sum(tmp_values)
        return val

    def gradient(self, var: np.ndarray) -> np.ndarray:
        """returns the gradient of Weighted Probability Based Squared Error.

        see :func:`~quara.data_analysis.loss_function.LossFunction.gradient`
        """
        grad = []
        for alpha in range(self.num_var):
            tmp_values = []
            for index in range(len(self.func_prob_dists)):
                vec_a = self.func_gradient_prob_dists[index](alpha, var)
                vec_b = self.func_prob_dists[index](var) - self.prob_dists_q[index]
                if self.weight_matrices:
                    tmp_value = multiply_veca_vecb_matc(
                        vec_a, vec_b, self.weight_matrices[index]
                    )
                    tmp_values.append(tmp_value)
                else:
                    tmp_value = multiply_veca_vecb(vec_a, vec_b)
                    tmp_values.append(tmp_value)

            val = np.sum(tmp_values)
            grad.append(val)
        return 2 * np.array(grad, dtype=np.float64)

    def hessian(self, var: np.ndarray) -> np.ndarray:
        """returns the Hessian of Weighted Probability Based Squared Error.

        see :func:`~quara.data_analysis.loss_function.LossFunction.hessian`
        """
        hess_all = []
        for alpha in range(self.num_var):
            hess_alpha = []
            for beta in range(self.num_var):
                tmp_values = []
                for index in range(len(self.func_prob_dists)):
                    grad_alpha = self.func_gradient_prob_dists[index](alpha, var)
                    grad_beta = self.func_gradient_prob_dists[index](beta, var)
                    hess = self.func_hessian_prob_dists[index](alpha, beta, var)
                    p_q = self.func_prob_dists[index](var) - self.prob_dists_q[index]
                    if self.weight_matrices:
                        tmp_value = multiply_veca_vecb_matc(
                            grad_alpha, grad_beta, self.weight_matrices[index]
                        ) + multiply_veca_vecb_matc(
                            hess, p_q, self.weight_matrices[index]
                        )
                        tmp_values.append(tmp_value)
                    else:
                        tmp_value = multiply_veca_vecb(
                            grad_alpha, grad_beta
                        ) + multiply_veca_vecb(hess, p_q)
                        tmp_values.append(tmp_value)

                val = np.sum(tmp_values)
                hess_alpha.append(val)
            hess_all.append(hess_alpha)
        return 2 * np.array(hess_all, dtype=np.float64)
