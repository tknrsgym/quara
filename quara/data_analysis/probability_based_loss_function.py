from abc import abstractmethod
from typing import Callable, List

import numpy as np

from quara.data_analysis.loss_function import LossFunction, LossFunctionOption
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography


class ProbabilityBasedLossFunctionOption(LossFunctionOption):
    def __init__(self):
        super().__init__()


class ProbabilityBasedLossFunction(LossFunction):
    def __init__(
        self,
        num_var: int,
        func_prob_dists: List[Callable[[np.array], np.array]] = None,
        func_gradient_prob_dists: List[Callable[[int, np.array], np.array]] = None,
        func_hessian_prob_dists: List[Callable[[int, int, np.array], np.array]] = None,
        prob_dists_q: List[np.array] = None,
    ):
        """Constructor

        Subclasses have a responsibility to set ``on_value``, ``on_gradient``, ``on_hessian``.

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
        """
        super().__init__(num_var)
        self._func_prob_dists: List[Callable[[np.array], np.array]] = func_prob_dists
        self._func_gradient_prob_dists: List[
            Callable[[int, np.array], np.array]
        ] = func_gradient_prob_dists
        self._func_hessian_prob_dists: List[
            Callable[[int, int, np.array], np.array]
        ] = func_hessian_prob_dists
        self._prob_dists_q: List[np.array] = prob_dists_q

        self._on_func_prob_dists: bool = True if self._func_prob_dists is not None else False
        self._on_func_gradient_prob_dists: bool = True if self._func_gradient_prob_dists is not None else False
        self._on_func_hessian_prob_dists: bool = True if self._func_hessian_prob_dists is not None else False
        self._on_prob_dists_q: bool = True if self._prob_dists_q is not None else False

    @property
    def func_prob_dists(self) -> List[Callable[[np.array], np.array]]:
        """returns functions map variables to a probability distribution.

        Returns
        -------
        List[Callable[[np.array], np.array]]
            functions map variables to a probability distribution.
        """
        return self._func_prob_dists

    def set_func_prob_dists(
        self, func_prob_dists: List[Callable[[np.array], np.array]]
    ) -> None:
        """sets functions map variables to a probability distribution.

        Parameters
        ----------
        func_prob_dists : List[Callable[[np.array], np.array]]
            functions map variables to a probability distribution.
        """
        self._func_prob_dists = func_prob_dists
        self._on_func_prob_dists = True
        self._update_on_value_true()
        self._update_on_gradient_true()
        self._update_on_hessian_true()

    def _set_on_func_prob_dists(self, on_func_prob_dists: bool) -> None:
        self._on_func_prob_dists = on_func_prob_dists

    @property
    def func_gradient_prob_dists(self) -> List[Callable[[int, np.array], np.array]]:
        """returns functions map variables and an index of variables to gradient of probability distributions.

        Returns
        -------
        List[Callable[[int, np.array], np.array]]
            functions map variables and an index of variables to gradient of probability distributions.
        """
        return self._func_gradient_prob_dists

    def set_func_gradient_prob_dists(
        self, func_gradient_prob_dists: List[Callable[[int, np.array], np.array]]
    ) -> None:
        """sets functions map variables and an index of variables to gradient of probability distributions.

        Parameters
        ----------
        func_gradient_prob_dists : List[Callable[[int, np.array], np.array]]
            functions map variables and an index of variables to gradient of probability distributions.
        """
        self._func_gradient_prob_dists = func_gradient_prob_dists
        self._on_func_gradient_prob_dists = True
        self._update_on_gradient_true()
        self._update_on_hessian_true()

    @property
    def func_hessian_prob_dists(self) -> List[Callable[[int, int, np.array], np.array]]:
        """returns functions map variables and indices of variables to Hessian of probability distributions.

        Returns
        -------
        List[Callable[[int, int, np.array], np.array]]
            functions map variables and indices of variables to Hessian of probability distributions.
        """
        return self._func_hessian_prob_dists

    def set_func_hessian_prob_dists(
        self, func_hessian_prob_dists: List[Callable[[int, int, np.array], np.array]]
    ) -> None:
        """sets functions map variables and indices of variables to Hessian of probability distributions.

        Parameters
        ----------
        func_hessian_prob_dists : List[Callable[[int, int, np.array], np.array]]
            functions map variables and indices of variables to Hessian of probability distributions.
        """
        self._func_hessian_prob_dists = func_hessian_prob_dists
        self._on_func_hessian_prob_dists = True
        self._update_on_hessian_true()

    @property
    def prob_dists_q(self) -> List[np.array]:
        """returns vectors of ``q``, by default None.

        Returns
        -------
        List[np.array]
            vectors of ``q``, by default None.
        """
        return self._prob_dists_q

    @property
    def on_func_prob_dists(self) -> bool:
        """returns whether or not to support ``func_prob_dists``.

        Returns
        -------
        bool
            whether or not to support ``func_prob_dists``.
        """
        return self._on_func_prob_dists

    @property
    def on_func_gradient_prob_dists(self) -> bool:
        """returns whether or not to support ``func_gradient_dists``.

        Returns
        -------
        bool
            whether or not to support ``func_gradient_dists``.
        """
        return self._on_func_gradient_prob_dists

    @property
    def on_func_hessian_prob_dists(self) -> bool:
        """returns whether or not to support ``func_hessian_dists``.

        Returns
        -------
        bool
            whether or not to support ``func_hessian_dists``.
        """
        return self._on_func_hessian_prob_dists

    @property
    def on_prob_dists_q(self) -> bool:
        """returns whether or not to support ``prob_dists_q``.

        Returns
        -------
        bool
            whether or not to support ``prob_dists_q``.
        """
        return self._on_prob_dists_q

    def size_prob_dists(self) -> int:
        """returns size of ``func_prob_dists``.

        Returns
        -------
        int
            size of ``func_prob_dists``.
        """
        if self._func_prob_dists is None:
            return 0
        else:
            return len(self._func_prob_dists)

    def set_prob_dists_q(self, prob_dists_q: List[np.array]) -> None:
        """sets vectors of ``q``, by default None.

        Parameters
        ----------
        prob_dists_q : List[np.array]
            vectors of ``q``, by default None.
        """
        self._prob_dists_q = prob_dists_q
        if self.prob_dists_q:
            self._on_prob_dists_q = True
        else:
            self._on_prob_dists_q = False

        self._update_on_value_true()
        self._update_on_gradient_true()
        self._update_on_hessian_true()

    def _generate_func_prob_dist(
        self, matA: np.array, vecB: np.array, size_prob_dist: int, index: int
    ):
        def _process(var: np.array) -> np.array:
            return (
                matA[size_prob_dist * index : size_prob_dist * (index + 1)] @ var
                + vecB[size_prob_dist * index : size_prob_dist * (index + 1)]
            )

        return _process

    def set_func_prob_dists_from_standard_qt(self, qt: StandardQTomography) -> None:
        matA = np.copy(qt.calc_matA())
        vecB = np.copy(qt.calc_vecB())
        num_func = qt.num_schedules
        size_prob_dist = int(matA.shape[0] / num_func)

        func_prob_dists = []
        for index in range(num_func):
            func = self._generate_func_prob_dist(matA, vecB, size_prob_dist, index)
            func_prob_dists.append(func)
        self.set_func_prob_dists(func_prob_dists)

    def _generate_func_gradient_prob_dist(
        self, matA: np.array, size_prob_dist: int, index: int
    ):
        def _process(alpha: int, var: np.array) -> np.array:
            prob_dist = [
                matA[size_prob_dist * index + prob_dist_index, alpha]
                for prob_dist_index in range(size_prob_dist)
            ]
            return np.array(prob_dist, dtype=np.float64)

        return _process

    def set_func_gradient_prob_dists_from_standard_qt(
        self, qt: StandardQTomography
    ) -> None:
        matA = np.copy(qt.calc_matA())
        num_func = qt.num_schedules
        size_prob_dist = int(matA.shape[0] / num_func)

        func_gradient_prob_dists = []
        for index in range(num_func):
            func = self._generate_func_gradient_prob_dist(matA, size_prob_dist, index)
            func_gradient_prob_dists.append(func)
        self.set_func_gradient_prob_dists(func_gradient_prob_dists)

    def _generate_func_hessian_prob_dist(self, size_prob_dist: int, index: int):
        def _process(alpha: int, beta: int, var: np.array):
            return np.array([0.0] * size_prob_dist, dtype=np.float64)

        return _process

    def set_func_hessian_prob_dists_from_standard_qt(
        self, qt: StandardQTomography
    ) -> None:
        matA = np.copy(qt.calc_matA())
        num_func = qt.num_schedules
        size_prob_dist = int(matA.shape[0] / num_func)

        func_hessian_prob_dists = []
        for index in range(num_func):
            func = self._generate_func_hessian_prob_dist(size_prob_dist, index)
            func_hessian_prob_dists.append(func)
        self.set_func_hessian_prob_dists(func_hessian_prob_dists)
