import time
from typing import Callable, List

import numpy as np


from quara.data_analysis.loss_function import LossFunction, LossFunctionOption
from quara.data_analysis.minimization_algorithm import (
    MinimizationAlgorithm,
    MinimizationAlgorithmOption,
    MinimizationResult,
)
from quara.math import func_proj
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.settings import Settings


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
        var_start: np.array = None,
        mu: float = None,
        gamma: float = 0.3,
        eps: float = None,
    ):
        super().__init__(var_start)

        if mu is None and var_start is not None:
            mu = 3 / (2 * np.sqrt(var_start.shape[0]))
        self._mu: float = mu
        self._gamma: float = gamma
        if eps is None:
            eps = Settings.get_atol() / 10.0
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
    def __init__(self, func_proj: Callable[[np.array], np.array]):
        super().__init__()
        self._func_proj: Callable[[np.array], np.array] = func_proj
        self._is_gradient_required: bool = True
        self._is_hessian_required: bool = False
        self._qt: StandardQTomography = None

    @property
    def func_proj(self) -> Callable[[np.array], np.array]:
        return self._func_proj

    def set_constraint_from_standard_qt(self, qt: StandardQTomography) -> None:
        # TOOD this implentation is wrong. qt does not have on_algo_eq_constraint and on_algo_ineq_constraint.
        self._qt = qt
        setting_info = self._qt.generate_empty_estimation_obj_with_setting_info()
        if (
            setting_info.on_algo_eq_constraint == True
            and setting_info.on_algo_ineq_constraint == True
        ):
            # TODO use QOperation.calc_proj_physical()
            pass
        elif (
            setting_info.on_algo_eq_constraint == True
            and setting_info.on_algo_ineq_constraint == False
        ):
            # TODO use QOperation.calc_eq_constraint()
            pass
        elif (
            setting_info.on_algo_eq_constraint == False
            and setting_info.on_algo_ineq_constraint == True
        ):
            # TODO use QOperation.calc_ineq_constraint()
            pass
        else:
            self._func_proj = func_proj.proj_to_self()

    def is_loss_sufficient(self) -> bool:
        """returns whether the loss is sufficient.

        Returns
        -------
        bool
            whether the loss is sufficient.
        """
        # TODO validate
        if self.loss is None:
            return False
        elif self.loss.on_value is False:
            return False
        elif self.loss.on_gradient is False:
            return False
        else:
            return True

    def is_option_sufficient(self) -> bool:
        """returns whether the option is sufficient.

        Returns
        -------
        bool
            whether the option is sufficient.
        """
        # TODO validate
        if self.option is None:
            return False
        elif self.option.mu <= 0:
            return False
        elif self.option.gamma <= 0:
            return False
        elif self.option.eps <= 0:
            return False
        else:
            return True

    def is_loss_and_option_sufficient(self) -> bool:
        """returns whether the loss and the option are sufficient.

        Returns
        -------
        bool
            whether the loss and the option are sufficient.
        """
        # TODO validate when option.var_start exists
        num_var_option = self.option.var_start.shape[0]
        num_var_loss = self.loss.num_var
        if num_var_option != num_var_loss:
            return False
        else:
            return True

    def optimize(
        self,
        loss_function: LossFunction,
        loss_function_option: LossFunctionOption,
        algorithm_option: ProjectedGradientDescentBaseOption,
        on_iteration_history: bool = False,
    ) -> ProjectedGradientDescentBaseResult:
        """optimizes using specified parameters.

        Parameters
        ----------
        loss_function : LossFunction
            Loss Function
        loss_function_option : LossFunctionOption
            Loss Function Option
        algorithm_option : ProjectedGradientDescentBaseOption
            Projected Gradient Descent Base Algorithm Option
        on_iteration_history : bool, optional
            whether to return iteration history, by default False

        Returns
        -------
        ProjectedGradientDescentBaseResult
            the result of the optimization.

        Raises
        ------
        ValueError
            when ``on_gradient`` of ``loss_function`` is False. 
        ValueError
            when ``is_gradient_required`` of this algorithm is False.
        """
        # TODO delete these checks
        if loss_function.on_gradient == False:
            raise ValueError(
                "to execute ProjectedGradientDescentBase, 'on_gradient' of loss_function must be True."
            )
        if self.is_gradient_required == False:
            raise ValueError(
                "to execute ProjectedGradientDescentBase, 'is_gradient_required' of this algorithm must be True."
            )

        if algorithm_option.var_start is None:
            x_prev = (
                self._qt.generate_empty_estimation_obj_with_setting_info().generate_origin_obj()
            )
        else:
            x_prev = algorithm_option.var_start
        x_next = None
        if algorithm_option.var_start is None:
            mu = 3 / (2 * np.sqrt(self._qt.num_variables))
        else:
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
                self.func_proj(x_prev - loss_function.gradient(x_prev) / mu) - x_prev
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
