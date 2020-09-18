from abc import abstractmethod

import numpy as np

from quara.data_analysis.loss_function import LossFunction, LossFunctionOption


class MinimizationResult:
    def __init__(self, value: np.array, computation_time: float = None):
        """Constructor

        Parameters
        ----------
        value : np.array
            the result of the minimization.
        computation_time : float, optional
            computation time for the minimization, by default None
        """
        self._value: np.array = value
        self._computation_time: float = computation_time

    @property
    def value(self) -> np.array:
        """returns the result of the minimization.

        Returns
        -------
        np.array
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
        var_start: np.array,
        is_gradient_required: bool,
        is_hessian_required: bool,
    ):
        """Constructor

        Parameters
        ----------
        var_start : np.array
            initial variable for the algorithm.
        is_gradient_required : bool
            whether or not to support gradient.
        is_hessian_required : bool
            whether or not to support Hessian.
        """
        self._var_start: np.array = var_start
        self._is_gradient_required: bool = is_gradient_required
        self._is_hessian_required: bool = is_hessian_required

    @property
    def var_start(self) -> np.array:
        """returns initial variable for the algorithm.

        Returns
        -------
        np.array
            initial variable for the algorithm.
        """
        return self._var_start

    @property
    def is_gradient_required(self) -> bool:
        """returns whether or not to support gradient.

        Returns
        -------
        bool
            whether or not to support gradient.
        """
        return self._is_gradient_required

    @property
    def is_hessian_required(self) -> bool:
        """returns whether or not to support Hessian.

        Returns
        -------
        bool
            whether or not to support Hessian.
        """
        return self._is_hessian_required


class MinimizationAlgorithm:
    def __init__(self):
        pass

    @abstractmethod
    def optimize(
        self,
        loss_function: LossFunction,
        loss_function_option: LossFunctionOption,
        algorithm_option: MinimizationAlgorithmOption,
        on_iteration_history: bool = False,
    ) -> MinimizationResult:
        """optimize using specified parameters.

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
