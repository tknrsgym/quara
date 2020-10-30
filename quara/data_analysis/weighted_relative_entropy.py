from abc import abstractmethod
from typing import Callable, List, Union

import numpy as np

from quara.data_analysis.probability_based_loss_function import (
    ProbabilityBasedLossFunction,
    ProbabilityBasedLossFunctionOption,
)
from quara.math.entropy import (
    relative_entropy,
    gradient_relative_entropy_2nd,
    hessian_relative_entropy_2nd,
)
from quara.utils import matrix_util


class WeightedRelativeEntropyOption(ProbabilityBasedLossFunctionOption):
    def __init__(self):
        super().__init__()


class WeightedRelativeEntropy(ProbabilityBasedLossFunction):
    def __init__(
        self,
        num_var: int,
        func_prob_dists: List = None,
        func_gradient_prob_dists: List = None,
        func_hessian_prob_dists: List = None,
        prob_dists_q: List[np.array] = None,
        weights: Union[List[float], List[np.float64]] = None,
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
        weights : Union[List[float], List[np.float64]], optional
            weights, by default None
        """
        super().__init__(
            num_var,
            func_prob_dists,
            func_gradient_prob_dists,
            func_hessian_prob_dists,
            prob_dists_q,
        )

        # validate
        self._validate_weights(weights)
        self._weights = weights

        # update on_value, on_gradient and on_hessian
        self._update_on_value_true()
        self._update_on_gradient_true()
        self._update_on_hessian_true()

    def _validate_weights(self, weights: List[float]) -> None:
        if weights:
            for index, weight in enumerate(weights):
                # weights are real values
                if type(weight) != float and type(weight) != np.float64:
                    raise ValueError(
                        f"values of weights must be real numbers(float or np.float64). dtype of weights[{index}] is {type(weight)}"
                    )

    @property
    def weights(self) -> List[float]:
        """returns weights.

        Returns
        -------
        List[float]
            weights.
        """
        return self._weights

    def set_weights(self, weights: List[float]) -> None:
        """sets weights.

        Parameters
        ----------
        weights : List[float]
            weights.
        """
        self._validate_weights(weights)
        self._weights = weights

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
        """returns the value of Weighted Relative Entropy.

        see :func:`~quara.data_analysis.loss_function.LossFunction.value`
        """
        val = 0.0
        for index in range(len(self.func_prob_dists)):
            q = self.prob_dists_q[index]
            p = self.func_prob_dists[index](var)
            if self.weights:
                val += self.weights[index] * relative_entropy(q, p)
            else:
                val += relative_entropy(q, p)

        return val

    def gradient(self, var: np.array) -> np.array:
        """returns the gradient of Weighted Relative Entropy.

        see :func:`~quara.data_analysis.loss_function.LossFunction.gradient`
        """
        grad = np.zeros(self.num_var, dtype=np.float64)
        for index in range(len(self.func_prob_dists)):
            # calc list of gradient p
            tmp_grad_ps = []
            for alpha in range(self.num_var):
                tmp_grad_ps.append(self.func_gradient_prob_dists[index](alpha, var))
            grad_ps = np.stack(tmp_grad_ps, 1)

            q = self.prob_dists_q[index]
            p = self.func_prob_dists[index](var)
            if self.weights:
                grad += self.weights[index] * gradient_relative_entropy_2nd(
                    q, p, grad_ps
                )
            else:
                grad += gradient_relative_entropy_2nd(q, p, grad_ps)

        return grad

    def hessian(self, var: np.array) -> np.array:
        """returns the Hessian of Weighted Relative Entropy.

        see :func:`~quara.data_analysis.loss_function.LossFunction.hessian`
        """
        hess = np.zeros((self.num_var, self.num_var), dtype=np.float64)
        for index in range(len(self.func_prob_dists)):
            # calc list of gradient p
            tmp_grad_ps = []
            for alpha in range(self.num_var):
                tmp_grad_ps.append(self.func_gradient_prob_dists[index](alpha, var))
            grad_ps = np.stack(tmp_grad_ps, 1)

            # calc list of Hessian p
            tmp_hess = []
            for alpha in range(self.num_var):
                tmp_hess_row = []
                for beta in range(self.num_var):
                    tmp_hess_row.append(
                        self.func_hessian_prob_dists[index](alpha, beta, var)
                    )
                tmp_hess.append(tmp_hess_row)
            hess_ps = np.array(tmp_hess).transpose(2, 0, 1)

            q = self.prob_dists_q[index]
            p = self.func_prob_dists[index](var)
            if self.weights:
                hess += self.weights[index] * hessian_relative_entropy_2nd(
                    q, p, grad_ps, hess_ps
                )
            else:
                hess += hessian_relative_entropy_2nd(q, p, grad_ps, hess_ps)

        return hess