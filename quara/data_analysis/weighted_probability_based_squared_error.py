from abc import abstractmethod
from typing import Callable, List

import numpy as np

from quara.data_analysis.probability_based_loss_function import (
    ProbabilityBasedLossFunction,
    ProbabilityBasedLossFunctionOption,
)
from quara.math.matrix import multiply_veca_vecb, multiply_veca_vecb_matc
from quara.utils import matrix_util


class WeightedProbabilityBasedSquaredErrorOption(ProbabilityBasedLossFunctionOption):
    def __init__(self):
        super().__init__()


class WeightedProbabilityBasedSquaredError(ProbabilityBasedLossFunction):
    def __init__(
        self,
        num_var: int,
        func_prob_dists: List = None,
        func_gradient_prob_dists: List = None,
        func_hessian_prob_dists: List = None,
        prob_dists_q: List[np.array] = None,
        weight_matrices: List[np.array] = None,
    ):
        """Constructor

        Parameters
        ----------
        num_var : int
            number of variables.
        func_prob_dists : List[Callable[[np.array], np.array]], optional
            functions map variables to a probability distribution.
        func_gradient_prob_dists : List[Callable[[int, np.array], np.array]], optional
            functions map variables and an index of variables to gradient of probability distributions.
        func_hessian_prob_dists : List[Callable[[int, int, np.array], np.array]], optional
            functions map variables and indices of variables to Hessian of probability distributions.
        prob_dists_q : List[np.array], optional
            vectors of ``q``, by default None.
        weight_matrices : List[np.array], optional
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

    def _validate_weight_matrices(self, weight_matrices: List[np.array]) -> None:
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
    def weight_matrices(self) -> List[np.array]:
        """returns weight matrices.

        Returns
        -------
        List[np.array]
            weight matrices.
        """
        return self._weight_matrices

    def set_weight_matrices(self, weight_matrices: List[np.array]) -> None:
        """sets weight matrices.

        Parameters
        ----------
        weight_matrices : List[np.array]
            weight matrices.
        """
        self._validate_weight_matrices(weight_matrices)
        self._weight_matrices = weight_matrices

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

    def value(self, var: np.array) -> np.float64:
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

    def gradient(self, var: np.array) -> np.array:
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

    def hessian(self, var: np.array) -> np.array:
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