import time
from typing import Callable, List

import numpy as np


from quara.data_analysis.loss_function import LossFunction, LossFunctionOption
from quara.data_analysis.minimization_algorithm import (
    MinimizationAlgorithm,
    MinimizationAlgorithmOption,
    MinimizationResult,
)


class ProjectedGradientDescentBaseResult(MinimizationResult):
    def __init__(
        self,
        value: np.array,
        computation_time: float = None,
        k: int = None,
        fx: List[np.array] = None,
        x: List[np.array] = None,
        y: List[np.array] = None,
        alpha: List[float] = None,
    ):
        super().__init__(value, computation_time)
        self._k: int = k
        self._fx: List[np.array] = fx
        self._x: List[np.array] = x
        self._y: List[np.array] = y
        self._alpha: List[float] = alpha

    @property
    def k(self) -> int:
        return self._k

    @property
    def fx(self) -> List[np.array]:
        return self._fx

    @property
    def x(self) -> List[np.array]:
        return self._x

    @property
    def y(self) -> List[np.array]:
        return self._y

    @property
    def alpha(self) -> List[np.array]:
        return self._alpha


class ProjectedGradientDescentBaseOption(MinimizationAlgorithmOption):
    def __init__(
        self,
        func_proj: Callable[[np.array], np.array],
        var_start: np.array,
        mu: float = None,
        gamma: float = 0.3,
        eps: float = 1.0e-10,
    ):
        super().__init__(var_start, True, False)

        self._func_proj: Callable[[np.array], np.array] = func_proj

        if mu is None:
            mu = 3 / (2 * np.sqrt(var_start.shape[0]))
        self._mu: float = mu
        self._gamma: float = gamma
        self._eps: float = eps

    @property
    def func_proj(self) -> Callable[[np.array], np.array]:
        return self._func_proj

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def eps(self) -> float:
        return self._eps


class ProjectedGradientDescentBase(MinimizationAlgorithm):
    def __init__(self):
        super().__init__()

    def optimize(
        self,
        loss_function: LossFunction,
        loss_function_option: LossFunctionOption,
        algorithm_option: ProjectedGradientDescentBaseOption,
        on_iteration_history: bool = False,
    ) -> ProjectedGradientDescentBaseResult:
        if loss_function.on_gradient == False:
            raise ValueError(
                "to execute ProjectedGradientDescentBase, 'on_gradient' of loss_function must be True."
            )
        if algorithm_option.is_gradient_required == False:
            raise ValueError(
                "to execute ProjectedGradientDescentBase, 'is_gradient_required' of algorithm option must be True."
            )

        x_prev = algorithm_option.var_start
        x_next = None
        mu = algorithm_option.mu
        gamma = algorithm_option.gamma
        eps = algorithm_option.eps

        # variables for debug
        if on_iteration_history:
            start_time = time.time()
            fxs = [loss_function.value(x_prev)]
            xs = [x_prev]
            ys = []
            alphas = []

        k = 0
        is_doing = True
        while is_doing:
            # shift variables
            if x_next is not None:
                x_prev = x_next

            y_prev = (
                algorithm_option.func_proj(x_prev - loss_function.gradient(x_prev) / mu)
                - x_prev
            )

            alpha = 1.0
            while self._is_doing_for_alpha(x_prev, y_prev, alpha, gamma, loss_function):
                alpha = 0.5 * alpha

            x_next = x_prev + alpha * y_prev
            k += 1

            val = loss_function.value(x_prev) - loss_function.value(x_next)
            is_doing = True if val > eps else False

            # variables for iteration history
            if on_iteration_history:
                fxs.append(loss_function.value(x_next))
                xs.append(x_next)
                ys.append(y_prev)
                alphas.append(alpha)

        if on_iteration_history:
            computation_time = time.time() - start_time
            result = ProjectedGradientDescentBaseResult(
                x_next,
                computation_time=computation_time,
                k=k,
                fx=fxs,
                x=xs,
                y=ys,
                alpha=alphas,
            )
            return result
        else:
            result = ProjectedGradientDescentBaseResult(x_next)
            return result

    def _is_doing_for_alpha(
        self,
        x_prev: np.array,
        y_prev: np.array,
        alpha: float,
        gamma: float,
        loss_function: LossFunction,
    ) -> bool:
        left_side = loss_function.value(x_prev + alpha * y_prev)
        right_side = loss_function.value(x_prev) + gamma * alpha * (
            np.dot(y_prev, loss_function.gradient(x_prev))
        )
        return left_side > right_side
