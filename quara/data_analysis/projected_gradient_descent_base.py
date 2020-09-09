import numpy as np


from quara.data_analysis.loss_function import LossFunction, LossFunctionOption
from quara.data_analysis.minimization_algorithm import (
    MinimizationAlgorithm,
    MinimizationAlgorithmOption,
)


class ProjectedGradientDescentBaseOption(MinimizationAlgorithmOption):
    def __init__(
        self,
        var_start: np.array,
        mu: float = None,
        gamma: float = 0.3,
        eps: float = 10 ^ (-10),
        on_iteration_history: bool = False,
    ):
        super().__init__(
            var_start, True, False, on_iteration_history=on_iteration_history
        )

        if mu is None:
            mu = 3 / (2 * np.sqrt(var_start.shape[0]))
        self._mu: float = mu
        self._gamma: float = gamma
        self._eps: float = eps

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
    ):
        # TODO loss_function, algorithm_option is_gradient_requiredとかのチェック
        # TODO history対応

        x_prev = algorithm_option.var_start
        mu = algorithm_option.mu
        gamma = algorithm_option.gamma

        k = 0
        is_doing = False
        while is_doing:
            y_prev = (
                self._func_proj(
                    x_prev - loss_function.gradient(x_prev) / mu, loss_function_option
                )
                - x_prev
            )

            alpha = 1.0
            while self._is_doing_for_alpha(x_prev, y_prev, alpha, gamma, loss_function):
                alpha = 0.5 * alpha

            x_next = x_prev + alpha * y_prev
            k += 1

            is_doing = (
                loss_function.value(x_prev) - loss_function.value(x_next)
                > algorithm_option.eps
            )

    def _func_proj(
        self, var: np.array, loss_function_option: LossFunctionOption
    ) -> np.array:
        # TODO var_aを使うと、QuadraticLossFunctionOptionに依存しているがよいのか？
        var_a = loss_function_option.var_a
        if var_a is None:
            proj_value = var
        else:
            proj_value = var - np.dot(var_a, var) / np.dot(var_a, var_a) * var_a + var_a

        return proj_value

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
