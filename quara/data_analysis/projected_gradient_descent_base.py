import numpy as np


from quara.data_analysis.loss_function import LossFunction, LossFunctionOption
from quara.data_analysis.minimization_algorithm import (
    MinimizationAlgorithm,
    MinimizationAlgorithmOption,
)


class ProjectedGradientDescentBaseOption(MinimizationAlgorithmOption):
    def __init__(
        self,
        func_proj,
        var_start: np.array,
        mu: float = None,
        gamma: float = 0.3,
        eps: float = 1.0e-10,
        on_iteration_history: bool = False,
    ):
        super().__init__(
            var_start, True, False, on_iteration_history=on_iteration_history
        )

        self._func_proj = func_proj

        if mu is None:
            mu = 3 / (2 * np.sqrt(var_start.shape[0]))
        self._mu: float = mu
        self._gamma: float = gamma
        self._eps: float = eps

    @property
    def func_proj(self):
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
    ) -> np.array:
        # TODO loss_function, algorithm_option is_gradient_requiredとかのチェック
        # TODO history対応

        x_prev = algorithm_option.var_start
        x_next = None
        mu = algorithm_option.mu
        gamma = algorithm_option.gamma
        eps = algorithm_option.eps
        # print(
        #    f"x_start={x_prev}, var_ref={loss_function._var_ref}, var_a={loss_function_option.var_a}, mu={mu}, gamma={gamma}, eps={eps}"
        # )
        # print(f"grad(x_start)={loss_function.gradient(x_prev)}")

        k = 0
        is_doing = True
        while is_doing:
            # shift variables
            if x_next is not None:
                x_prev = x_next

            y_prev = (
                algorithm_option.func_proj(
                    x_prev - loss_function.gradient(x_prev) / mu, loss_function_option
                )
                - x_prev
            )
            # print(f"k={k}, y_prev={y_prev}")

            alpha = 1.0
            while self._is_doing_for_alpha(x_prev, y_prev, alpha, gamma, loss_function):
                alpha = 0.5 * alpha
                # print(f"k={k}, alpha={alpha}")

            x_next = x_prev + alpha * y_prev
            # print(f"k={k}, x_prev={x_prev}, x_next={x_next}")
            k += 1

            val = loss_function.value(x_prev) - loss_function.value(x_next)
            # print(
            #    f"k={k}, val={val}, grad(x_prev)={loss_function.gradient(x_prev)}, grad(x_next)={loss_function.gradient(x_next)}"
            # )
            is_doing = True if val > eps else False
            if k == 10:
                break

        print(
            f"var_ref={loss_function._var_ref}, var_a={loss_function_option.var_a}: {algorithm_option.var_start} -> {x_next}"
        )
        return x_next

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
