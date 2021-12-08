from abc import abstractmethod
from typing import Callable, List

import numpy as np


from quara.loss_function.loss_function import LossFunction, LossFunctionOption
from quara.minimization_algorithm.minimization_algorithm import (
    MinimizationAlgorithm,
    MinimizationAlgorithmOption,
    MinimizationResult,
)
from quara.math import func_proj
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.settings import Settings


class ProjectedGradientDescentResult(MinimizationResult):
    def __init__(
        self,
        value: np.ndarray,
        computation_time: float = None,
        k: int = None,
        fx: List[np.ndarray] = None,
        x: List[np.ndarray] = None,
        error_values: List[float] = None,
    ):
        super().__init__(value, computation_time)
        self._k: int = k
        self._fx: List[np.ndarray] = fx
        self._x: List[np.ndarray] = x
        self._error_values: List[float] = error_values

    @property
    def k(self) -> int:
        """returns the number of iterations.

        Returns
        -------
        int
            the number of iterations.
        """
        return self._k

    @property
    def fx(self) -> List[np.ndarray]:
        """return the value of f(x) per iteration.

        Returns
        -------
        List[np.ndarray]
            the value of f(x) per iteration.
        """
        return self._fx

    @property
    def x(self) -> List[np.ndarray]:
        """return the x per iteration.

        Returns
        -------
        List[np.ndarray]
            the x per iteration.
        """
        return self._x

    @property
    def error_values(self) -> List[np.ndarray]:
        """return the error_values per iteration.

        Returns
        -------
        List[np.ndarray]
            the error_values per iteration.
        """
        return self._error_values


class ProjectedGradientDescentOption(MinimizationAlgorithmOption):
    def __init__(
        self,
        on_algo_eq_constraint: bool = True,
        on_algo_ineq_constraint: bool = True,
        var_start: np.ndarray = None,
        max_iteration_optimization: int = None,
        max_iteration_proj_physical: int = None,
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
            maximun number of iterations of optimization, by default None.
        max_iteration_proj_physical: int, optional
            maximun number of iterations of projection to physical, by default None.
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
        )

        self._max_iteration_proj_physical: int = max_iteration_proj_physical

        if not mode_stopping_criterion_gradient_descent in [
            "single_difference_loss",
            "sum_absolute_difference_loss",
            "sum_absolute_difference_variable",
            "sum_absolute_difference_projected_gradient",
        ]:
            raise ValueError(
                f"unsupported 'mode_stopping_criterion_gradient_descent'={mode_stopping_criterion_gradient_descent}"
            )
        self._mode_stopping_criterion_gradient_descent = (
            mode_stopping_criterion_gradient_descent
        )

        if type(num_history_stopping_criterion_gradient_descent) != int:
            raise ValueError(
                f"type(num_history_stopping_criterion_gradient_descent) is not int. type={type(num_history_stopping_criterion_gradient_descent)}"
            )
        if num_history_stopping_criterion_gradient_descent < 1:
            raise ValueError(
                f"num_history_stopping_criterion_gradient_descent must be greater than or equal to 1. num_history_stopping_criterion_gradient_descent={num_history_stopping_criterion_gradient_descent}"
            )
        self._num_history_stopping_criterion_gradient_descent = (
            num_history_stopping_criterion_gradient_descent
        )

        if not mode_proj_order in ["eq_ineq", "ineq_eq"]:
            raise ValueError(f"unsupported mode_proj_order={mode_proj_order}")
        self._mode_proj_order: str = mode_proj_order

        if eps is None:
            eps = Settings.get_atol() / 10.0
        self._eps: float = eps

    @property
    def max_iteration_proj_physical(self) -> int:
        """returns maximun number of iterations of optimization.

        Returns
        -------
        int
            maximun number of iterations of optimization.
        """
        return self._max_iteration_proj_physical

    @property
    def mode_stopping_criterion_gradient_descent(self) -> str:
        """returns mode of stopping criterion for gradient descent.

        Returns
        -------
        str
            mode of stopping criterion for gradient descent.
        """
        return self._mode_stopping_criterion_gradient_descent

    @property
    def num_history_stopping_criterion_gradient_descent(self) -> int:
        """returns number of history to be used stopping criterion for gradient descent.

        Returns
        -------
        int
            number of history to be used stopping criterion for gradient descent.
        """
        return self._num_history_stopping_criterion_gradient_descent

    @property
    def mode_proj_order(self) -> str:
        """returns the order in which the projections are performed.

        Returns
        -------
        str
            the order in which the projections are performed.
        """
        return self._mode_proj_order

    @property
    def eps(self) -> float:
        """returns algorithm option ``eps``.

        Returns
        -------
        float
            algorithm option ``eps``.
        """
        return self._eps


class ProjectedGradientDescent(MinimizationAlgorithm):
    def __init__(self, func_proj: Callable[[np.ndarray], np.ndarray] = None):
        """Constructor

        Parameters
        ----------
        func_proj : Callable[[np.ndarray], np.ndarray], optional
            function of projection, by default None
        """
        super().__init__()
        self._func_proj: Callable[[np.ndarray], np.ndarray] = func_proj
        self._qt: StandardQTomography = None

    @property
    def func_proj(self) -> Callable[[np.ndarray], np.ndarray]:
        """returns function of projection.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            function of projection.
        """
        return self._func_proj

    def set_constraint_from_standard_qt_and_option(
        self,
        qt: StandardQTomography,
        option: ProjectedGradientDescentOption,
    ) -> None:
        """sets constraint from StandardQTomography and Algorithm Option.

        Parameters
        ----------
        qt : StandardQTomography
            StandardQTomography to set constraint.
        option : ProjectedGradientDescentOption
            Algorithm Option.
        """
        self._qt = qt

        if self._func_proj is not None:
            return

        setting_info = self._qt.generate_empty_estimation_obj_with_setting_info()
        if (
            option.on_algo_eq_constraint == True
            and option.on_algo_ineq_constraint == True
        ):
            self._func_proj = setting_info.func_calc_proj_physical_with_var(
                on_para_eq_constraint=setting_info.on_para_eq_constraint,
                mode_proj_order=option.mode_proj_order,
                max_iteration=option.max_iteration_proj_physical,
            )
        elif (
            option.on_algo_eq_constraint == True
            and option.on_algo_ineq_constraint == False
        ):
            self._func_proj = setting_info.func_calc_proj_eq_constraint_with_var(
                setting_info.on_para_eq_constraint
            )
        elif (
            option.on_algo_eq_constraint == False
            and option.on_algo_ineq_constraint == True
        ):
            self._func_proj = setting_info.func_calc_proj_ineq_constraint_with_var(
                setting_info.on_para_eq_constraint
            )
        else:
            self._func_proj = func_proj.proj_to_self()

    @abstractmethod
    def is_loss_sufficient(self) -> bool:
        """returns whether the loss is sufficient.

        Returns
        -------
        bool
            whether the loss is sufficient.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def is_option_sufficient(self) -> bool:
        """returns whether the option is sufficient.

        Returns
        -------
        bool
            whether the option is sufficient.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    def is_loss_and_option_sufficient(self) -> bool:
        """returns whether the loss and the option are sufficient.

        Returns
        -------
        bool
            whether the loss and the option are sufficient.
        """
        # validate when option.var_start exists
        if (
            self.option is not None
            and self.option.var_start is not None
            and self.loss is not None
        ):
            num_var_option = self.option.var_start.shape[0]
            num_var_loss = self.loss.num_var
            if num_var_option != num_var_loss:
                return False

        return True

    @abstractmethod
    def optimize(
        self,
        loss_function: LossFunction,
        loss_function_option: LossFunctionOption,
        algorithm_option: ProjectedGradientDescentOption,
        on_iteration_history: bool = False,
    ) -> ProjectedGradientDescentResult:
        """optimizes using specified parameters.

        Parameters
        ----------
        loss_function : LossFunction
            Loss Function
        loss_function_option : LossFunctionOption
            Loss Function Option
        algorithm_option : ProjectedGradientDescentOption
            Projected Gradient Descent Base Algorithm Option
        on_iteration_history : bool, optional
            whether to return iteration history, by default False

        Returns
        -------
        ProjectedGradientDescentResult
            the result of the optimization.

        Raises
        ------
        ValueError
            when ``on_value`` of ``loss_function`` is False.
        ValueError
            when ``on_gradient`` of ``loss_function`` is False.
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()
