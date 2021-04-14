from abc import abstractmethod
from typing import Callable, List, Tuple

import numpy as np

from quara.loss_function.loss_function import LossFunction, LossFunctionOption
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography


class ProbabilityBasedLossFunctionOption(LossFunctionOption):
    def __init__(self, mode_weight: str, weights: List = None, weight_name: str = None):
        super().__init__(
            mode_weight=mode_weight, weights=weights, weight_name=weight_name
        )


class ProbabilityBasedLossFunction(LossFunction):
    def __init__(
        self,
        num_var: int = None,
        func_prob_dists: List[Callable[[np.ndarray], np.ndarray]] = None,
        func_gradient_prob_dists: List[Callable[[int, np.ndarray], np.ndarray]] = None,
        func_hessian_prob_dists: List[
            Callable[[int, int, np.ndarray], np.ndarray]
        ] = None,
        prob_dists_q: List[np.ndarray] = None,
    ):
        """Constructor

        Subclasses have a responsibility to set ``on_value``, ``on_gradient``, ``on_hessian``.

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
        """
        super().__init__(num_var)
        self._func_prob_dists: List[
            Callable[[np.ndarray], np.ndarray]
        ] = func_prob_dists
        self._func_gradient_prob_dists: List[
            Callable[[int, np.ndarray], np.ndarray]
        ] = func_gradient_prob_dists
        self._func_hessian_prob_dists: List[
            Callable[[int, int, np.ndarray], np.ndarray]
        ] = func_hessian_prob_dists
        self._prob_dists_q: List[np.ndarray] = prob_dists_q

        self._on_func_prob_dists: bool = (
            True if self._func_prob_dists is not None else False
        )
        self._on_func_gradient_prob_dists: bool = (
            True if self._func_gradient_prob_dists is not None else False
        )
        self._on_func_hessian_prob_dists: bool = (
            True if self._func_hessian_prob_dists is not None else False
        )
        self._on_prob_dists_q: bool = True if self._prob_dists_q is not None else False

    @property
    def func_prob_dists(self) -> List[Callable[[np.ndarray], np.ndarray]]:
        """returns functions map variables to a probability distribution.

        Returns
        -------
        List[Callable[[np.ndarray], np.ndarray]]
            functions map variables to a probability distribution.
        """
        return self._func_prob_dists

    def set_func_prob_dists(
        self, func_prob_dists: List[Callable[[np.ndarray], np.ndarray]]
    ) -> None:
        """sets functions map variables to a probability distribution.

        Parameters
        ----------
        func_prob_dists : List[Callable[[np.ndarray], np.ndarray]]
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
    def func_gradient_prob_dists(self) -> List[Callable[[int, np.ndarray], np.ndarray]]:
        """returns functions map variables and an index of variables to gradient of probability distributions.

        Returns
        -------
        List[Callable[[int, np.ndarray], np.ndarray]]
            functions map variables and an index of variables to gradient of probability distributions.
        """
        return self._func_gradient_prob_dists

    def set_func_gradient_prob_dists(
        self, func_gradient_prob_dists: List[Callable[[int, np.ndarray], np.ndarray]]
    ) -> None:
        """sets functions map variables and an index of variables to gradient of probability distributions.

        Parameters
        ----------
        func_gradient_prob_dists : List[Callable[[int, np.ndarray], np.ndarray]]
            functions map variables and an index of variables to gradient of probability distributions.
        """
        self._func_gradient_prob_dists = func_gradient_prob_dists
        self._on_func_gradient_prob_dists = True
        self._update_on_gradient_true()
        self._update_on_hessian_true()

    @property
    def func_hessian_prob_dists(
        self,
    ) -> List[Callable[[int, int, np.ndarray], np.ndarray]]:
        """returns functions map variables and indices of variables to Hessian of probability distributions.

        Returns
        -------
        List[Callable[[int, int, np.ndarray], np.ndarray]]
            functions map variables and indices of variables to Hessian of probability distributions.
        """
        return self._func_hessian_prob_dists

    def set_func_hessian_prob_dists(
        self,
        func_hessian_prob_dists: List[Callable[[int, int, np.ndarray], np.ndarray]],
    ) -> None:
        """sets functions map variables and indices of variables to Hessian of probability distributions.

        Parameters
        ----------
        func_hessian_prob_dists : List[Callable[[int, int, np.ndarray], np.ndarray]]
            functions map variables and indices of variables to Hessian of probability distributions.
        """
        self._func_hessian_prob_dists = func_hessian_prob_dists
        self._on_func_hessian_prob_dists = True
        self._update_on_hessian_true()

    @property
    def prob_dists_q(self) -> List[np.ndarray]:
        """returns vectors of ``q``, by default None.

        Returns
        -------
        List[np.ndarray]
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

    def set_prob_dists_q(self, prob_dists_q: List[np.ndarray]) -> None:
        """sets vectors of ``q``, by default None.

        Parameters
        ----------
        prob_dists_q : List[np.ndarray]
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
        self, matA: np.ndarray, vecB: np.ndarray, size_prob_dist: int, index: int
    ):
        def _process(var: np.ndarray) -> np.ndarray:
            return (
                matA[size_prob_dist * index : size_prob_dist * (index + 1)] @ var
                + vecB[size_prob_dist * index : size_prob_dist * (index + 1)]
            )

        return _process

    def set_func_prob_dists_from_standard_qt(self, qt: StandardQTomography) -> None:
        """sets the function of probability distributions from StandardQTomography.

        Parameters
        ----------
        qt : StandardQTomography
            StandardQTomography to set the function of probability distributions.
        """
        matA = np.copy(qt.calc_matA())
        vecB = np.copy(qt.calc_vecB())
        self._num_var = qt.num_variables
        num_func = qt.num_schedules
        size_prob_dist = int(matA.shape[0] / num_func)

        func_prob_dists = []
        for index in range(num_func):
            func = self._generate_func_prob_dist(matA, vecB, size_prob_dist, index)
            func_prob_dists.append(func)
        self.set_func_prob_dists(func_prob_dists)

    def _generate_func_gradient_prob_dist(
        self, matA: np.ndarray, size_prob_dist: int, index: int
    ):
        def _process(alpha: int, var: np.ndarray) -> np.ndarray:
            prob_dist = [
                matA[size_prob_dist * index + prob_dist_index, alpha]
                for prob_dist_index in range(size_prob_dist)
            ]
            return np.array(prob_dist, dtype=np.float64)

        return _process

    def set_func_gradient_prob_dists_from_standard_qt(
        self, qt: StandardQTomography
    ) -> None:
        """sets the gradient of probability distributions from StandardQTomography.

        Parameters
        ----------
        qt : StandardQTomography
            StandardQTomography to set the gradient of probability distributions.
        """
        matA = np.copy(qt.calc_matA())
        num_func = qt.num_schedules
        size_prob_dist = int(matA.shape[0] / num_func)

        func_gradient_prob_dists = []
        for index in range(num_func):
            func = self._generate_func_gradient_prob_dist(matA, size_prob_dist, index)
            func_gradient_prob_dists.append(func)
        self.set_func_gradient_prob_dists(func_gradient_prob_dists)

    def _generate_func_hessian_prob_dist(self, size_prob_dist: int, index: int):
        def _process(alpha: int, beta: int, var: np.ndarray):
            return np.array([0.0] * size_prob_dist, dtype=np.float64)

        return _process

    def set_func_hessian_prob_dists_from_standard_qt(
        self, qt: StandardQTomography
    ) -> None:
        """sets the Hessian of probability distributions from StandardQTomography.

        Parameters
        ----------
        qt : StandardQTomography
            StandardQTomography to set the Hessian of probability distributions.
        """
        matA = np.copy(qt.calc_matA())
        num_func = qt.num_schedules
        size_prob_dist = int(matA.shape[0] / num_func)

        func_hessian_prob_dists = []
        for index in range(num_func):
            func = self._generate_func_hessian_prob_dist(size_prob_dist, index)
            func_hessian_prob_dists.append(func)
        self.set_func_hessian_prob_dists(func_hessian_prob_dists)

    def _set_weights_by_mode(
        self, mode_weight: str, data: List[Tuple[int, np.ndarray]]
    ) -> None:
        """sets weights of loss function.

        This function does not do anything by default.
        If necessary, implement this function in a subclass.

        Parameters
        ----------
        mode_weight : str
            mode for weights.
        data : List[Tuple[int, np.ndarray]]
            empirical distributions.
        """
        pass

    def set_from_standard_qtomography_option_data(
        self,
        qtomography: StandardQTomography,
        option: LossFunctionOption,
        data: List[Tuple[int, np.ndarray]],
        is_gradient_required: bool,
        is_hessian_required: bool,
    ) -> None:
        """sets settings of loss function.

        Parameters
        ----------
        qtomography : StandardQTomography
            StandardQTomography for settings of loss function.
        option : LossFunctionOption
            ProbabilityBasedLossFunctionOption for settings of loss function.
        data : List[Tuple[int, np.ndarray]]
            empirical distributions for settings of loss function.
        is_gradient_required : bool
            whether or not to require gradient.
        is_hessian_required : bool
            whether or not to require Hessian.
        """
        self.set_from_option(option)
        empi_dists = [empi_dist_tmp[1] for empi_dist_tmp in data]
        self.set_prob_dists_q(empi_dists)
        self.set_func_prob_dists_from_standard_qt(qtomography)
        if is_gradient_required:
            self.set_func_gradient_prob_dists_from_standard_qt(qtomography)
        if is_hessian_required:
            self.set_func_hessian_prob_dists_from_standard_qt(qtomography)
        self._set_weights_by_mode(option.mode_weight, data)
