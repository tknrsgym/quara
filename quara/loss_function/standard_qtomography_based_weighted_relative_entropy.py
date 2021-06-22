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
        prob_dists_q: List[np.ndarray] = None,
        weights: Union[List[float], List[np.float64]] = None,
    ):
        """Constructor of StandardQTomography based WeightedRelativeEntropy.

        Parameters
        ----------
        num_var : int, optional
            number of variables, by default None
        prob_dists_q : List[np.ndarray], optional
            vectors of ``q``, by default None.
        weights : Union[List[float], List[np.float64]], optional
            weights, by default None
        """
        if prob_dists_q:
            self._prob_dists_q_flat = np.array(prob_dists_q, dtype=np.float64).flatten()

        super().__init__(
            num_var=num_var,
            func_prob_dists=None,
            func_gradient_prob_dists=None,
            func_hessian_prob_dists=None,
            prob_dists_q=prob_dists_q,
            weights=weights,
        )

    def _calc_extend_weights(self) -> None:
        # calc the extend weights.
        # "extend weights" is a vector that expands the weight vector to fit the size of the probability distributions.
        # this is used in the "value" function and "gradient" function for fast computation.
        if self.weights is not None:
            extend_weights = []
            for weight, prob_dist in zip(self.weights, self.prob_dists_q):
                extend_weights += [weight] * len(prob_dist)
            self._extend_weights = np.array(extend_weights, dtype=np.float64)

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
        self._calc_extend_weights()

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
        self._calc_extend_weights()

        self._on_func_gradient_prob_dists = True
        self._update_on_gradient_true()
        self._update_on_hessian_true()

    def value(self, var: np.ndarray) -> np.float64:
        """returns the value of Weighted Relative Entropy.

        see :func:`~quara.data_analysis.loss_function.LossFunction.value`
        """
        q = self._prob_dists_q_flat
        p = self._matA @ var + self._vecB

        if self.weights is not None:
            vector = self._extend_weights * relative_entropy_vector(
                q, p, is_valid_required=False
            )
        else:
            vector = relative_entropy_vector(q, p, is_valid_required=False)
        val = np.sum(vector)

        return val

    def gradient(self, var: np.ndarray) -> np.ndarray:
        """returns the gradient of Weighted Relative Entropy.

        see :func:`~quara.data_analysis.loss_function.LossFunction.gradient`
        """
        q = self._prob_dists_q_flat
        p = self._matA @ var + self._vecB
        grad_ps = self._matA

        if self.weights is not None:
            grad = np.dot(
                self._extend_weights,
                gradient_relative_entropy_2nd_vector(
                    q, p, grad_ps, is_valid_required=False
                ),
            )
        else:
            vectors = gradient_relative_entropy_2nd_vector(
                q, p, grad_ps, is_valid_required=False
            )
            grad = np.sum(vectors, axis=0)

        return grad

    def hessian(self, var: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
