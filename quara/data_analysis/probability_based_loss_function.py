from abc import abstractmethod
from typing import Callable, List

import numpy as np

from quara.data_analysis.loss_function import LossFunction, LossFunctionOption


class ProbabilityBasedLossFunctionOption(LossFunctionOption):
    def __init__(self):
        super().__init__()


class ProbabilityBasedLossFunction(LossFunction):
    def __init__(
        self,
        num_var: int,
        func_prob_dists: List[Callable[[np.array], np.array]] = None,
        func_gradient_dists: List[Callable[[int, np.array], np.array]] = None,
        func_hessian_dists: List[Callable[[int, int, np.array], np.array]] = None,
        prob_dists_q: List[np.array] = None,
    ):
        # TODO on_value, on_gradientとon_hessianは子クラスでupdateすることをdocに記述すること
        super().__init__(num_var)
        self._func_prob_dists: List[Callable[[np.array], np.array]] = func_prob_dists
        self._func_gradient_dists: List[
            Callable[[int, np.array], np.array]
        ] = func_gradient_dists
        self._func_hessian_dists: List[
            Callable[[int, int, np.array], np.array]
        ] = func_hessian_dists
        self._prob_dists_q: List[np.array] = prob_dists_q

        self._on_func_prob_dists: bool = True if self._func_prob_dists is not None else False
        self._on_func_gradient_dists: bool = True if self._func_gradient_dists is not None else False
        self._on_func_hessian_dists: bool = True if self._func_hessian_dists is not None else False
        self._on_prob_dists_q: bool = True if self._prob_dists_q is not None else False

    @property
    def func_prob_dists(self) -> List[Callable[[np.array], np.array]]:
        return self._func_prob_dists

    def set_func_prob_dists(
        self, func_prob_dists: List[Callable[[np.array], np.array]]
    ) -> None:
        self._func_prob_dists = func_prob_dists
        self._on_func_prob_dists = True
        self._update_on_value_true()
        self._update_on_gradient_true()
        self._update_on_hessian_true()

    def _set_on_func_prob_dists(self, on_func_prob_dists: bool) -> None:
        self._on_func_prob_dists = on_func_prob_dists

    @property
    def func_gradient_dists(self) -> List[Callable[[int, np.array], np.array]]:
        return self._func_gradient_dists

    def set_func_gradient_dists(
        self, func_gradient_dists: List[Callable[[int, np.array], np.array]]
    ) -> None:
        self._func_gradient_dists = func_gradient_dists
        self._on_func_gradient_dists = True
        self._update_on_gradient_true()
        self._update_on_hessian_true()

    @property
    def func_hessian_dists(self) -> List[Callable[[int, int, np.array], np.array]]:
        return self._func_hessian_dists

    def set_func_hessian_dists(
        self, func_hessian_dists: List[Callable[[int, int, np.array], np.array]]
    ) -> None:
        self._func_hessian_dists = func_hessian_dists
        self._on_func_hessian_dists = True
        self._update_on_hessian_true()

    @property
    def prob_dists_q(self) -> List[np.array]:
        return self._prob_dists_q

    @property
    def on_func_prob_dists(self) -> bool:
        return self._on_func_prob_dists

    @property
    def on_func_gradient_dists(self) -> bool:
        return self._on_func_gradient_dists

    @property
    def on_func_hessian_dists(self) -> bool:
        return self._on_func_hessian_dists

    @property
    def on_prob_dists_q(self) -> bool:
        return self._on_prob_dists_q

    def size_prob_dists(self) -> int:
        if self._func_prob_dists is None:
            return 0
        else:
            return len(self._func_prob_dists)

    def set_prob_dists_q(self, prob_dists_q: List[np.array]) -> None:
        self._prob_dists_q = prob_dists_q
