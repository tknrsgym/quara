import time
from typing import Callable, List

import numpy as np


from quara.loss_function.loss_function import LossFunction, LossFunctionOption
from quara.minimization_algorithm.projected_gradient_descent import (
    ProjectedGradientDescent,
    ProjectedGradientDescentOption,
    ProjectedGradientDescentResult,
)


class ProjectedFastIterativeShrinkageThresholdingAlgorithmResult(
    ProjectedGradientDescentResult
):
    def __init__(
        self,
        value: np.ndarray,
        computation_time: float = None,
        k: int = None,
        fx: List[np.ndarray] = None,
        x: List[np.ndarray] = None,
        error_values: List[float] = None,
    ):
        super().__init__(value, computation_time, k, fx, x, error_values)


class ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
    ProjectedGradientDescentOption
):
    def __init__(
        self,
        on_algo_eq_constraint: bool = True,
        on_algo_ineq_constraint: bool = True,
        var_start: np.ndarray = None,
        max_iteration_optimization: int = 1000,
        max_iteration_proj_physical: int = 100000,
        delta: float = None,
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
        max_iteration_optimization: int, optional
            maximun number of iterations of optimization, by default 1000.
        max_iteration_proj_physical: int, optional
            maximun number of iterations of projection to physical, by default 100000.
        delta : float, optional
            algorithm option ``r``, by default None
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
            max_iteration_optimization=max_iteration_optimization,
            max_iteration_proj_physical=max_iteration_proj_physical,
            mode_stopping_criterion_gradient_descent=mode_stopping_criterion_gradient_descent,
            num_history_stopping_criterion_gradient_descent=num_history_stopping_criterion_gradient_descent,
            mode_proj_order=mode_proj_order,
            eps=eps,
        )

        self._delta: float = delta

    @property
    def delta(self) -> float:
        """returns algorithm option ``delta``.

        Returns
        -------
        float
            algorithm option ``delta``.
        """
        return self._delta


class ProjectedFastIterativeShrinkageThresholdingAlgorithm(ProjectedGradientDescent):
    """This algorithm is based on the following:

    Eliot Bolduc, George C. Knee, Erik M. Gauger & Jonathan Leach, "Projected gradient descent algorithms for quantum state tomography",

    - npj Quantum Information volume 3, Article number: 44 (2017) https://www.nature.com/articles/s41534-017-0043-1
    - arXiv:1612.09531 https://arxiv.org/abs/1612.09531
    """

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
        elif self.option.delta is not None and self.option.delta <= 0:
            return False
        elif self.option.eps is None or self.option.eps <= 0:
            return False
        else:
            return True

    def optimize(
        self,
        loss_function: LossFunction,
        loss_function_option: LossFunctionOption,
        algorithm_option: ProjectedFastIterativeShrinkageThresholdingAlgorithmOption,
        on_iteration_history: bool = False,
    ) -> ProjectedFastIterativeShrinkageThresholdingAlgorithmResult:
        """optimizes using specified parameters.

        Parameters
        ----------
        loss_function : LossFunction
            Loss Function
        loss_function_option : LossFunctionOption
            Loss Function Option
        algorithm_option : ProjectedFastIterativeShrinkageThresholdingAlgorithmOption
            Projected Fast Iterative Shrinkage-Thresholding Algorithm Option
        on_iteration_history : bool, optional
            whether to return iteration history, by default False

        Returns
        -------
        ProjectedFastIterativeShrinkageThresholdingAlgorithmResult
            the result of the optimization.

        Raises
        ------
        ValueError
            when ``on_value`` of ``loss_function`` is False.
        ValueError
            when ``on_gradient`` of ``loss_function`` is False.
        """
        max_iteration = algorithm_option.max_iteration_optimization

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

        if algorithm_option.delta:
            delta = algorithm_option.delta
        elif algorithm_option.delta is None and self._qt:
            delta = 1 / (10 * np.sqrt(self._qt.num_variables))
        elif algorithm_option.delta is None and algorithm_option.var_start is not None:
            delta = 1 / (10 * np.sqrt(len(algorithm_option.var_start)))
        else:
            raise ValueError("unable to set the algorithm option delta.")

        eps = algorithm_option.eps
        x_prev_prev = x_prev
        x_next = None

        # variables for debug
        if on_iteration_history:
            start_time = time.time()
            fxs = [loss_function.value(x_prev)]
            xs = [x_prev]
        error_values = []

        is_doing = True
        for k in range(1, max_iteration + 1):
            # shift variables
            if x_next is not None:
                x_prev_prev = x_prev
                x_prev = x_next

            tmp = (
                x_prev
                + (k - 2) / (k + 1) * (x_prev - x_prev_prev)
                - delta * loss_function.gradient(x_prev)
            )
            x_next = self.func_proj(tmp)

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
                fxs.append(loss_function.value(x_next, validate=True))
                xs.append(x_next)

            is_doing = True if value > eps else False
            if not is_doing:
                break

        if k == max_iteration:
            start_red = "\033[31m"
            end_color = "\033[0m"
            print(
                f"{start_red}Warning!{end_color} pfista iterations exceeds the limit {max_iteration}."
            )

        if on_iteration_history:
            computation_time = time.time() - start_time
            result = ProjectedFastIterativeShrinkageThresholdingAlgorithmResult(
                x_next,
                computation_time=computation_time,
                k=k,
                fx=fxs,
                x=xs,
                error_values=error_values,
            )
            return result
        else:
            result = ProjectedFastIterativeShrinkageThresholdingAlgorithmResult(x_next)
            return result
