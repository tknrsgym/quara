from abc import abstractmethod
from typing import List

import numpy as np

from quara.data_analysis.loss_function import LossFunction, LossFunctionOption


class ProbabilityBasedLossFunctionOption(LossFunctionOption):
    def __init__(self):
        super().__init__()


class ProbabilityBasedLossFunction(LossFunction):
    def __init__(
        self,
        num_var: int,
        func_prob_dists: List,
        func_gradient_dists: List,
        func_hessian_dists: List,
        prob_dists_q: List[np.array],
    ):
        super().__init__(num_var, True, True)
        self._func_prob_dists = func_prob_dists
        # TODO 以下はオプション
        self._func_gradient_dists = func_gradient_dists
        self._func_hessian_dists = func_hessian_dists
        self._prob_dists_q = prob_dists_q

    @property
    def func_prob_dists(self):
        return self._func_prob_dists

    @property
    def func_gradient_dists(self):
        return self._func_gradient_dists

    @property
    def func_hessian_dists(self):
        return self._func_hessian_dists

    @property
    def prob_dists_q(self):
        return self._prob_dists_q

    def size_prob_dists(self) -> int:
        return len(self._func_prob_dists)

    def size_prob_dist(self, i: int) -> int:
        return len(self._func_prob_dists[0])

    def set_prob_dists_q(self, prob_dists_q: List[np.array]) -> None:
        self._prob_dists_q = prob_dists_q
