import time
from typing import Callable, List

import numpy as np


from quara.loss_function.loss_function import LossFunction, LossFunctionOption
from quara.minimization_algorithm.projected_gradient_descent import (
    ProjectedGradientDescent,
    ProjectedGradientDescentOption,
    ProjectedGradientDescentResult,
)


class ProjectedGradientDescentWithMomentumResult(ProjectedGradientDescentResult):
    def __init__(
        self,
        value: np.ndarray,
        computation_time: float = None,
        k: int = None,
        fx: List[np.ndarray] = None,
        x: List[np.ndarray] = None,
        moment: List[np.ndarray] = None,
        error_values: List[float] = None,
    ):
        super().__init__(value, computation_time, k, fx, x, error_values)
        self._moment: List[np.ndarray] = moment

    @property
    def moment(self) -> List[np.ndarray]:
        """return the moment per iteration.

        Returns
        -------
        List[np.ndarray]
            the moment per iteration.
        """
        return self._moment


class ProjectedGradientDescentWithMomentumOption(ProjectedGradientDescentOption):
    def __init__(
        self,
        on_algo_eq_constraint: bool = True,
        on_algo_ineq_constraint: bool = True,
        var_start: np.ndarray = None,
        r: float = 1.0,
        moment_0: List[np.float] = None,
        mode_stopping_criterion_gradient_descent: str = "single_difference_loss",
        num_history_stopping_criterion_gradient_descent: int = 1,
        mode_proj_order: str = "eq_ineq",
        eps: float = None,
    ):
        """Constructor

        Parameters
        ----------
        on_algo_eq_constraint : bool, optional
            whether this algorithm needs on algorithm equality constraint, by default True
        on_algo_ineq_constraint : bool, optional
            whether this algorithm needs on algorithm inequality constraint, by default True
        var_start : np.ndarray, optional
            initial variable for the algorithm, by default None
        r : float, optional
            algorithm option ``r``, by default 1.0
        moment_0 : List[np.float], optional
            algorithm option ``moment_0``, by default None
        mode_stopping_criterion_gradient_descent : str, optional
            mode of stopping criterion for gradient descent, by default "single_difference_loss"
        num_history_stopping_criterion_gradient_descent : int, optional
            number of history to be used stopping criterion for gradient descent, by default 1
            this must be a integer and greater than or equal to 1.
        mode_proj_order : str, optional
            the order in which the projections are performed, by default "eq_ineq".
        eps : float, optional
            algorithm option ``epsilon``, by default None
        """
        super().__init__(
            on_algo_eq_constraint=on_algo_eq_constraint,
            on_algo_ineq_constraint=on_algo_ineq_constraint,
            var_start=var_start,
            mode_stopping_criterion_gradient_descent=mode_stopping_criterion_gradient_descent,
            num_history_stopping_criterion_gradient_descent=num_history_stopping_criterion_gradient_descent,
            mode_proj_order=mode_proj_order,
            eps=eps,
        )

        self._r: float = r
        self._moment_0: List[np.float] = moment_0

    @property
    def r(self) -> float:
        """returns algorithm option ``r``.

        :math:`\\gamma = \\frac{1}{2r\\sqrt{n}}`

        Returns
        -------
        float
            algorithm option ``r``.
        """
        return self._r

    @property
    def moment_0(self) -> List[np.float]:
        """returns algorithm option ``moment_0``.

        Returns
        -------
        List[np.float]
            algorithm option ``moment_0``.
        """
        return self._moment_0


class ProjectedGradientDescentWithMomentum(ProjectedGradientDescent):
    def __init__(self, func_proj: Callable[[np.ndarray], np.ndarray] = None):
        """Constructor

        Parameters
        ----------
        func_proj : Callable[[np.ndarray], np.ndarray], optional
            function of projection, by default None
        """
        super().__init__(func_proj)
        self._is_gradient_required: bool = True
        self._is_hessian_required: bool = False

    def is_loss_sufficient(self) -> bool:
        """returns whether the loss is sufficient.

        Returns
        -------
        bool
            whether the loss is sufficient.
        """
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
        if self.option is None:
            return False
        elif self.option.r is not None and self.option.r <= 0:
            return False
        elif self.option.eps is None or self.option.eps <= 0:
            return False
        else:
            return True

    def optimize(
        self,
        loss_function: LossFunction,
        loss_function_option: LossFunctionOption,
        algorithm_option: ProjectedGradientDescentWithMomentumOption,
        max_iteration: int = 1000,
        on_iteration_history: bool = False,
    ) -> ProjectedGradientDescentWithMomentumResult:
        """optimizes using specified parameters.

        Parameters
        ----------
        loss_function : LossFunction
            Loss Function
        loss_function_option : LossFunctionOption
            Loss Function Option
        algorithm_option : ProjectedGradientDescentWithMomentumOption
            Projected Gradient Descent with Momentum Option
        max_iteration: int, optional
            maximun number of iterations, by default 1000.
        on_iteration_history : bool, optional
            whether to return iteration history, by default False

        Returns
        -------
        ProjectedGradientDescentWithMomentumResult
            the result of the optimization.

        Raises
        ------
        ValueError
            when ``on_value`` of ``loss_function`` is False.
        ValueError
            when ``on_gradient`` of ``loss_function`` is False.
        """
        if loss_function.on_value == False:
            raise ValueError(
                "to execute ProjectedGradientDescentBase, 'on_value' of loss_function must be True."
            )
        if loss_function.on_gradient == False:
            raise ValueError(
                "to execute ProjectedGradientDescentBase, 'on_gradient' of loss_function must be True."
            )

        if algorithm_option.var_start is None:
            x_prev = (
                self._qt.generate_empty_estimation_obj_with_setting_info()
                .generate_origin_obj()
                .to_var()
            )
        else:
            x_prev = algorithm_option.var_start

        if algorithm_option.r and self._qt:
            gamma = 1 / (2 * algorithm_option.r * np.sqrt(self._qt.num_variables))
        elif algorithm_option.r and algorithm_option.var_start is not None:
            gamma = 1 / (
                2 * algorithm_option.r * np.sqrt(len(algorithm_option.var_start))
            )
        else:
            raise ValueError("unable to set the algorithm option mu.")

        if algorithm_option.moment_0:
            moment_prev = algorithm_option.moment_0
        elif algorithm_option.moment_0 is None and self._qt:
            moment_prev = np.zeros(self._qt.num_variables)
        elif (
            algorithm_option.moment_0 is None and algorithm_option.var_start is not None
        ):
            moment_prev = np.zeros(len(algorithm_option.var_start))
        else:
            raise ValueError("unable to set the algorithm option mu.")

        x_next = None
        moment_next = None

        zeta = 0.95
        eps = algorithm_option.eps

        # variables for debug
        if on_iteration_history:
            start_time = time.time()
            fxs = [loss_function.value(x_prev)]
            xs = [x_prev]
            moments = [moment_prev]
        error_values = []

        is_doing = True
        for k in range(1, max_iteration + 1):
            # shift variables
            if x_next is not None:
                x_prev = x_next
                moment_prev = moment_next

            moment_next = zeta * moment_prev - gamma * loss_function.gradient(x_prev)
            x_next = self.func_proj(x_prev + moment_next)

            # calc error value depend on "mode_stopping_criterion_gradient_descent"
            if (
                algorithm_option.mode_stopping_criterion_gradient_descent
                == "single_difference_loss"
            ):
                error_value = loss_function.value(x_prev) - loss_function.value(x_next)
            elif (
                algorithm_option.mode_stopping_criterion_gradient_descent
                == "sum_absolute_difference_loss"
            ):
                error_value = np.abs(
                    loss_function.value(x_prev) - loss_function.value(x_next)
                )
            elif (
                algorithm_option.mode_stopping_criterion_gradient_descent
                == "sum_absolute_difference_variable"
            ):
                error_value = np.sqrt(np.sum((x_prev - x_next) ** 2))
            elif (
                algorithm_option.mode_stopping_criterion_gradient_descent
                == "sum_absolute_difference_projected_gradient"
            ):
                error_value = np.sqrt(np.sum(x_next ** 2))
            error_values.append(error_value)

            # calc sum of error values
            # if num_history_stopping_criterion_gradient_descent = 1, then this is single sum
            sum_range = min(
                len(error_values),
                algorithm_option.num_history_stopping_criterion_gradient_descent,
            )
            value = np.sum(error_values[-sum_range:])

            # variables for iteration history
            if on_iteration_history:
                fxs.append(loss_function.value(x_next))
                xs.append(x_next)
                moments.append(moment_next)

            is_doing = True if value > eps else False
            if not is_doing:
                break

        if on_iteration_history:
            computation_time = time.time() - start_time
            result = ProjectedGradientDescentWithMomentumResult(
                x_next,
                computation_time=computation_time,
                k=k,
                fx=fxs,
                x=xs,
                moment=moments,
                error_values=error_values,
            )
            return result
        else:
            result = ProjectedGradientDescentWithMomentumResult(x_next)
            return result
