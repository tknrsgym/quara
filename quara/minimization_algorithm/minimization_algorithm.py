from abc import abstractmethod

import numpy as np

from quara.loss_function.loss_function import LossFunction, LossFunctionOption
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography


class MinimizationResult:
    def __init__(self, value: np.ndarray, computation_time: float = None):
        """Constructor

        Parameters
        ----------
        value : np.ndarray
            the result of the minimization.
        computation_time : float, optional
            computation time for the minimization, by default None
        """
        self._value: np.ndarray = value
        self._computation_time: float = computation_time

    @property
    def value(self) -> np.ndarray:
        """returns the result of the minimization.

        Returns
        -------
        np.ndarray
            the result of the minimization.
        """
        return self._value

    @property
    def computation_time(self) -> float:
        """returns computation time for the estimate.

        Returns
        -------
        float
            computation time for the estimate.
        """
        return self._computation_time


class MinimizationAlgorithmOption:
    def __init__(
        self,
        on_algo_eq_constraint: bool = True,
        on_algo_ineq_constraint: bool = True,
        var_start: np.ndarray = None,
    ):
        """Constructor

        Parameters
        ----------
        on_algo_eq_constraint : bool, optional
            whether this algorithm needs on algorithm equality constraint, by default True
        on_algo_ineq_constraint : bool, optional
            whether this algorithm needs on algorithm inequality constraint, by default True
        var_start : np.ndarray, optional
            initial variable for the algorithm, by default None.
        """
        self._on_algo_eq_constraint = on_algo_eq_constraint
        self._on_algo_ineq_constraint = on_algo_ineq_constraint
        self._var_start: np.ndarray = var_start

    @property
    def on_algo_eq_constraint(self) -> bool:  # read only
        """returns whether this algorithm needs on algorithm equality constraint.

        Returns
        -------
        bool
            whether this algorithm needs on algorithm equality constraint.
        """
        return self._on_algo_eq_constraint

    @property
    def on_algo_ineq_constraint(self) -> bool:  # read only
        """returns whether this QOperation is on algorithm inequality constraint.

        Returns
        -------
        bool
            whether this QOperation is on algorithm inequality constraint.
        """
        return self._on_algo_ineq_constraint

    @property
    def var_start(self) -> np.ndarray:
        """returns initial variable for the algorithm.

        Returns
        -------
        np.ndarray
            initial variable for the algorithm.
        """
        return self._var_start


class MinimizationAlgorithm:
    def __init__(self):
        """Constructor.

        Subclasses have a responsibility to set the following variables.

        - ``_is_gradient_required``: whether or not to require gradient.
        - ``_is_hessian_required``: whether or not to require Hessian.

        """
        self._is_gradient_required: bool = False
        self._is_hessian_required: bool = False
        self._loss: LossFunction = None
        self._option: MinimizationAlgorithmOption = None

    @property
    def is_gradient_required(self) -> bool:
        """returns whether or not to require gradient.

        Returns
        -------
        bool
            whether or not to require gradient.
        """
        return self._is_gradient_required

    @property
    def is_hessian_required(self) -> bool:
        """returns whether or not to require Hessian.

        Returns
        -------
        bool
            whether or not to require Hessian.
        """
        return self._is_hessian_required

    @property
    def loss(self) -> LossFunction:
        """returns loss function.

        Returns
        -------
        LossFunction
            loss function.
        """
        return self._loss

    def set_from_loss(self, loss: LossFunction) -> None:
        """sets from LossFunction and calls ``is_loss_sufficient`` function.

        Parameters
        ----------
        loss : MinimizationAlgorithmOption
            loss to set.
        """
        self._loss = loss
        self.is_loss_sufficient()

    def is_loss_sufficient(self) -> bool:
        """returns whether the loss is sufficient.

        In the default implementation, this function returns True.
        Override with subclasses as needed.

        Returns
        -------
        bool
            whether the loss is sufficient.
        """
        return True

    @property
    def option(self) -> MinimizationAlgorithmOption:
        """returns algorithm option.

        Returns
        -------
        LossFunctionOption
            algorithm option.
        """
        return self._option

    def set_from_option(self, option: MinimizationAlgorithmOption) -> None:
        """sets option from MinimizationAlgorithmOption and calls ``is_option_sufficient`` function.

        Parameters
        ----------
        option : MinimizationAlgorithmOption
            option to set.
        """
        self._option = option
        self.is_option_sufficient()

    @abstractmethod
    def set_constraint_from_standard_qt_and_option(
        self, qt: StandardQTomography, option: MinimizationAlgorithmOption
    ) -> None:
        """sets constraint from StandardQTomography and Algorithm Option.

        Parameters
        ----------
        qt : StandardQTomography
            StandardQTomography to set constraint.
        option : MinimizationAlgorithmOption
            Algorithm Option.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    def is_option_sufficient(self) -> bool:
        """returns whether the option is sufficient.

        In the default implementation, this function returns True.
        Override with subclasses as needed.

        Returns
        -------
        bool
            whether the option is sufficient.
        """
        return True

    def is_loss_and_option_sufficient(self) -> bool:
        """returns whether the loss and the option are sufficient.

        In the default implementation, this function returns True.
        Override with subclasses as needed.

        Returns
        -------
        bool
            whether the loss and the option are sufficient.
        """
        return True

    @abstractmethod
    def optimize(
        self,
        loss_function: LossFunction,
        loss_function_option: LossFunctionOption,
        algorithm_option: MinimizationAlgorithmOption,
        on_iteration_history: bool = False,
    ) -> MinimizationResult:
        """optimizes using specified parameters.

        Parameters
        ----------
        loss_function : LossFunction
            Loss Function
        loss_function_option : LossFunctionOption
            Loss Function Option
        algorithm_option : MinimizationAlgorithmOption
            Minimization Algorithm Option
        on_iteration_history : bool, optional
            whether to return iteration history, by default False

        Returns
        -------
        MinimizationResult
            the result of the optimization.
        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()
