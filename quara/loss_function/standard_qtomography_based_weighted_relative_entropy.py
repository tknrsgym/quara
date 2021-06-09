from abc import abstractmethod
from typing import Callable, List, Tuple, Union

import numpy as np

from quara.loss_function.weighted_relative_entropy import (
    WeightedRelativeEntropy,
    WeightedRelativeEntropyOption,
)
from quara.math.entropy import (
    relative_entropy_vector,
    gradient_relative_entropy_2nd_vector,
    hessian_relative_entropy_2nd,
)
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography


class StandardQTomographyBasedWeightedRelativeEntropyOption(
    WeightedRelativeEntropyOption
):
    def __init__(
        self, mode_weight: str = None, weights: List = None, weight_name: str = None
    ):
        """Constructor

        mode_weight should be the following value:

        - "identity" then uses identity matrices for weights.
        - "custom" then uses user custom matrices for weights.

        Parameters
        ----------
        mode_weight : str, optional
            mode weight, by default None
        weights : List, optional
            list of weight, by default None
        weight_name : str, optional
            weight name for reporting, by default None
        """
        super().__init__(
            mode_weight=mode_weight, weights=weights, weight_name=weight_name
        )


class StandardQTomographyBasedWeightedRelativeEntropy(WeightedRelativeEntropy):
    def __init__(
        self,
        num_var: int = None,
        func_prob_dists: List = None,
        func_gradient_prob_dists: List = None,
        func_hessian_prob_dists: List = None,
        prob_dists_q: List[np.ndarray] = None,
        weights: Union[List[float], List[np.float64]] = None,
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

    def set_prob_dists_q(self, prob_dists_q: List[np.ndarray]) -> None:
        """sets vectors of ``q``, by default None.

        Parameters
        ----------
        prob_dists_q : List[np.ndarray]
            vectors of ``q``, by default None.
        """
        self._prob_dists_q_flat = np.array(prob_dists_q, dtype=np.float64).flatten()
        super().set_prob_dists_q(prob_dists_q)

    def set_func_prob_dists_from_standard_qt(self, qt: StandardQTomography) -> None:
        """sets the function of probability distributions from StandardQTomography.

        Parameters
        ----------
        qt : StandardQTomography
            StandardQTomography to set the function of probability distributions.
        """
        self._matA = np.copy(qt.calc_matA())
        self._vecB = np.copy(qt.calc_vecB())

        self._on_func_prob_dists = True
        self._update_on_value_true()
        self._update_on_gradient_true()
        self._update_on_hessian_true()

    def set_func_gradient_prob_dists_from_standard_qt(
        self, qt: StandardQTomography
    ) -> None:
        """sets the gradient of probability distributions from StandardQTomography.

        Parameters
        ----------
        qt : StandardQTomography
            StandardQTomography to set the gradient of probability distributions.
        """
        self._matA = np.copy(qt.calc_matA())
        self._num_var = qt.num_variables
        # TODO
        self._matA_deformed = self._matA.reshape((self.num_var, -1))

        self._on_func_gradient_prob_dists = True
        self._update_on_gradient_true()
        self._update_on_hessian_true()

    def value(self, var: np.ndarray) -> np.float64:
        """returns the value of Weighted Relative Entropy.

        see :func:`~quara.data_analysis.loss_function.LossFunction.value`
        """
        q = self.prob_dists_q
        p = (self._matA @ var + self._vecB).reshape((self.num_var, -1))
        if self.weights:
            # TODO
            vector = self.weights * relative_entropy_vector(
                q, p, is_valid_required=False
            )
        else:
            vectors = relative_entropy_vector(q, p, is_valid_required=False)
            val = np.sum(vectors)

        return val

    def gradient(self, var: np.ndarray) -> np.ndarray:
        """returns the gradient of Weighted Relative Entropy.

        see :func:`~quara.data_analysis.loss_function.LossFunction.gradient`
        """
        grad_ps = self._matA
        # TODO
        q = self._prob_dists_q_flat
        p = self._matA @ var + self._vecB

        if self.weights:
            weight_vector = np.array([self.weights, self.weights]).flatten("F")
            grad = weight_vector * gradient_relative_entropy_2nd_vector(
                q, p, grad_ps, is_valid_required=False
            )
        else:
            vectors = gradient_relative_entropy_2nd_vector(
                q, p, grad_ps, is_valid_required=False
            )
            grad = np.sum(vectors, axis=0)

        return grad

    def hessian(self, var: np.ndarray) -> np.ndarray:
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
            p = self._matA @ var + self._vecB
            if self.weights:
                hess += self.weights[index] * hessian_relative_entropy_2nd(
                    q, p, grad_ps, hess_ps, is_valid_required=False
                )
            else:
                hess += hessian_relative_entropy_2nd(
                    q, p, grad_ps, hess_ps, is_valid_required=False
                )

        return hess
