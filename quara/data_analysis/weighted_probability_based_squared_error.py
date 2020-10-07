from abc import abstractmethod
from typing import Callable, List

import numpy as np

from quara.data_analysis.probability_based_loss_function import (
    ProbabilityBasedLossFunction,
    ProbabilityBasedLossFunctionOption,
)
from quara.math.matrix import multiply_veca_vecb_matc


class WeightedProbabilityBasedSquaredErrorOption(ProbabilityBasedLossFunctionOption):
    def __init__(self):
        super().__init__()


class WeightedProbabilityBasedSquaredErrorFunction(ProbabilityBasedLossFunction):
    def __init__(
        self,
        num_var: int,
        func_prob_dists: List = None,
        func_gradient_dists: List = None,
        func_hessian_dists: List = None,
        prob_dists_q: List[np.array] = None,
        weight_matrices: List[np.array] = None,
    ):
        # TODO Wは実対称行列
        super().__init__(
            num_var,
            func_prob_dists,
            func_gradient_dists,
            func_hessian_dists,
            prob_dists_q,
        )
        if weight_matrices == None:
            # weight_matrices = list of identity matrices
            size = prob_dists_q[0].shape[0]
            weight_matrices = [
                np.eye(size, dtype=np.float64) for _ in range(len(prob_dists_q))
            ]
        self._weight_matrices = weight_matrices

        # update on_value, on_gradient and on_hessian
        self._update_on_value_true()
        self._update_on_gradient_true()
        self._update_on_hessian_true()

    @property
    def weight_matrices(self) -> List[np.array]:
        return self._weight_matrices

    def _update_on_value_true(self) -> bool:
        if self.on_func_prob_dists is not None:
            self._set_on_value(True)
        return self.on_value

    def _update_on_gradient_true(self) -> bool:
        if (
            self.on_func_prob_dists is not None
            and self.on_func_gradient_dists is not None
        ):
            self._set_on_gradient(True)
        return self.on_gradient

    def _update_on_hessian_true(self) -> bool:
        if (
            self.on_func_prob_dists is not None
            and self.on_func_gradient_dists is not None
            and self.on_func_hessian_dists is not None
        ):
            self._set_on_hessian(True)
        return self.on_hessian

    def value(self, var: np.array) -> np.float64:
        tmp_values = []
        for index in range(len(self.func_prob_dists)):
            vec = self.func_prob_dists[index](var) - self.prob_dists_q[index]
            tmp_value = multiply_veca_vecb_matc(vec, vec, self.weight_matrices[index])
            tmp_values.append(tmp_value)

        val = np.sum(tmp_values)
        return val

    def gradient(self, var: np.array) -> np.array:
        grad = []
        for alpha in range(self.num_var):
            tmp_values = []
            for index in range(len(self.func_prob_dists)):
                vec_a = self.func_gradient_dists[index](alpha, var)
                vec_b = self.func_prob_dists[index](var) - self.prob_dists_q[index]
                tmp_value = multiply_veca_vecb_matc(
                    vec_a, vec_b, self.weight_matrices[index]
                )
                tmp_values.append(tmp_value)

            val = np.sum(tmp_values)
            grad.append(val)
        return 2 * np.array(grad, dtype=np.float64)

    def hessian(self, var: np.array) -> np.array:
        hess_all = []
        for alpha in range(self.num_var):
            hess_alpha = []
            for beta in range(self.num_var):
                tmp_values = []
                for index in range(len(self.func_prob_dists)):
                    grad_alpha = self.func_gradient_dists[index](alpha, var)
                    grad_beta = self.func_gradient_dists[index](beta, var)
                    hess = self.func_hessian_dists[index](alpha, beta, var)
                    p_q = self.func_prob_dists[index](var) - self.prob_dists_q[index]
                    tmp_value = multiply_veca_vecb_matc(
                        grad_alpha, grad_beta, self.weight_matrices[index]
                    ) + multiply_veca_vecb_matc(hess, p_q, self.weight_matrices[index])
                    tmp_values.append(tmp_value)

                val = np.sum(tmp_values)
                hess_alpha.append(val)
            hess_all.append(hess_alpha)
        return 2 * np.array(hess_all, dtype=np.float64)
