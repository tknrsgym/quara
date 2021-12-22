from typing import List

import numpy as np

from quara.loss_function.weighted_probability_based_squared_error import (
    WeightedProbabilityBasedSquaredError,
    WeightedProbabilityBasedSquaredErrorOption,
)
from quara.math.matrix import multiply_veca_vecb, multiply_veca_vecb_matc
from quara.math.probability import validate_prob_dist
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography


class StandardQTomographyBasedWeightedProbabilityBasedSquaredErrorOption(
    WeightedProbabilityBasedSquaredErrorOption
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


class StandardQTomographyBasedWeightedProbabilityBasedSquaredError(
    WeightedProbabilityBasedSquaredError
):
    def __init__(
        self,
        num_var: int = None,
        prob_dists_q: List[np.ndarray] = None,
        weight_matrices: List[np.ndarray] = None,
    ):
        """Constructor of StandardQTomography based WeightedProbabilityBasedSquaredError.

        Parameters
        ----------
        num_var : int, optional
            number of variables, by default None
        prob_dists_q : List[np.ndarray], optional
            vectors of ``q``, by default None.
        weight_matrices : List[np.ndarray], optional
            weight matrices, by default None
        """
        if prob_dists_q:
            self._prob_dists_q_flat = np.array(prob_dists_q, dtype=np.float64).flatten()
        self._extend_weight_matrix = None

        super().__init__(
            num_var=num_var,
            func_prob_dists=None,
            func_gradient_prob_dists=None,
            func_hessian_prob_dists=None,
            prob_dists_q=prob_dists_q,
            weight_matrices=weight_matrices,
        )

    def _calc_extend_weight_matrix(self) -> None:
        # if weight_matrices is None, not calculate.
        if self.weight_matrices is None:
            return

        # calc the extend weight matrix.
        # if weight_matrices=[W0, W1, W2], then "extend weight matrix"=[[W0, 0, 0], [0, W1, 0], [0, 0, W2]].
        # this is used in the "value" function and "gradient" function for fast computation.
        zero = np.zeros((self.weight_matrices[0].shape))
        size = len(self.weight_matrices)
        block_matrix = []
        for index, weight_matrix in enumerate(self.weight_matrices):
            row = [zero] * size
            row[index] = weight_matrix
            block_matrix.append(row)

        self._extend_weight_matrix = np.block(block_matrix)

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
        self._calc_extend_weight_matrix()

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
        self._calc_extend_weight_matrix()

        self._on_func_gradient_prob_dists = True
        self._update_on_gradient_true()
        self._update_on_hessian_true()

    def value(self, var: np.ndarray, validate: bool = False) -> np.float64:
        """returns the value of Weighted Relative Entropy.

        see :func:`~quara.data_analysis.loss_function.LossFunction.value`
        """
        q = self._prob_dists_q_flat
        p = self._matA @ var + self._vecB
        if validate:
            num_prob_dists = len(self.prob_dists_q)
            ps = p.reshape((num_prob_dists, -1))
            for index, prob in enumerate(ps):
                validate_prob_dist(
                    prob,
                    raise_error=False,
                    message=f"StandardQTomographyBasedWeightedProbabilityBasedSquaredError.value({index})",
                )
        vec = p - q

        if self._extend_weight_matrix is not None:
            val = multiply_veca_vecb_matc(vec, vec, self._extend_weight_matrix)
        else:
            val = multiply_veca_vecb(vec, vec)

        return val

    def gradient(self, var: np.ndarray, validate: bool = False) -> np.ndarray:
        """returns the gradient of Weighted Relative Entropy.

        see :func:`~quara.data_analysis.loss_function.LossFunction.gradient`
        """
        q = self._prob_dists_q_flat
        p = self._matA @ var + self._vecB
        if validate:
            num_prob_dists = len(self.prob_dists_q)
            ps = p.reshape((num_prob_dists, -1))
            for index, prob in enumerate(ps):
                validate_prob_dist(
                    prob,
                    raise_error=False,
                    message=f"StandardQTomographyBasedWeightedProbabilityBasedSquaredError.gradient({index})",
                )
        vec = p - q
        grad_ps = self._matA

        if self._extend_weight_matrix is not None:
            grad = 2 * grad_ps.T @ self._extend_weight_matrix @ vec
        else:
            grad = 2 * grad_ps.T @ vec

        return grad

    def hessian(self, var: np.ndarray, validate: bool = False) -> np.ndarray:
        raise NotImplementedError()
