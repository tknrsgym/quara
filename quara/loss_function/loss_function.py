from abc import abstractmethod
from typing import List, Tuple

import numpy as np

from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography


class LossFunctionOption:
    def __init__(
        self, mode_weight: str = None, weights: List = None, weight_name: str = None
    ):
        """Constructor

        Parameters
        ----------
        mode_weight : str, optional
            mode weight, by default None
        weights : List, optional
            list of weight, by default None
        weight_name : str, optional
            weight name for reporting, by default None
        """
        self._mode_weight: str = mode_weight
        self._weights: List = weights
        self._weight_name: str = weight_name

    @property
    def mode_weight(self) -> str:
        """returns mode weight.

        Returns
        -------
        str
            mode weight.
        """
        return self._mode_weight

    @property
    def weights(self) -> List:
        """returns weights.

        Returns
        -------
        List
            list of weight.
        """
        return self._weights

    @property
    def weight_name(self) -> str:
        """returns weight name for reporting.

        Returns
        -------
        str
            weight name for reporting.
        """
        return self._weight_name


class LossFunction:
    def __init__(self, num_var: int = None):
        """Constructor

        Subclasses have a responsibility to set ``on_value``, ``on_gradient``, ``on_hessian``.

        Parameters
        ----------
        num_var : int, optional
            number of variables, by default None
        """
        self._num_var: int = num_var
        self._on_value: bool = False
        self._on_gradient: bool = False
        self._on_hessian: bool = False
        self._option: LossFunctionOption = None

    @property
    def num_var(self) -> int:
        """returns number of variables.

        Returns
        -------
        int
            number of variables.
        """
        return self._num_var

    @property
    def on_value(self) -> bool:
        """returns whether or not to support value.

        Returns
        -------
        bool
            whether or not to support value.
        """
        return self._on_value

    @abstractmethod
    def _update_on_value_true(self) -> bool:
        """validates and updates ``on_value`` to True.

        Returns
        -------
        bool
            True when success.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    def _reset_on_value(self) -> None:
        """resets ``on_value`` to False.

        This function is intended to be called by subclasses.
        Subclasses have a responsibility to validate.
        """
        self._on_value = False

    def _set_on_value(self, on_value: bool) -> None:
        """sets ``on_value``.

        This function is intended to be called by subclasses.
        Subclasses have a responsibility to validate.

        Parameters
        ----------
        on_value : bool
            value of ``on_value``.
        """
        self._on_value = on_value

    @property
    def on_gradient(self) -> bool:
        """returns whether or not to support gradient.

        Returns
        -------
        bool
            whether or not to support gradient.
        """
        return self._on_gradient

    @abstractmethod
    def _update_on_gradient_true(self) -> bool:
        """validates and updates ``on_gradient`` to True.

        Returns
        -------
        bool
            True when success.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    def _reset_on_gradient(self) -> None:
        """resets ``on_gradient`` to False.

        This function is intended to be called by subclasses.
        Subclasses have a responsibility to validate.
        """
        self._on_gradient = False

    def _set_on_gradient(self, on_gradient: bool) -> None:
        """sets ``on_gradient``.

        This function is intended to be called by subclasses.
        Subclasses have a responsibility to validate.

        Parameters
        ----------
        on_gradient : bool
            value of ``on_gradient``.
        """
        self._on_gradient = on_gradient

    @property
    def on_hessian(self) -> bool:
        """returns whether or not to support Hessian.

        Returns
        -------
        bool
            whether or not to support Hessian.
        """
        return self._on_hessian

    @abstractmethod
    def _update_on_hessian_true(self) -> bool:
        """validates and updates ``on_hessian`` to True.

        Returns
        -------
        bool
            True when success.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    def _reset_on_hessian(self) -> None:
        """resets ``on_hessian`` to False.

        This function is intended to be called by subclasses.
        Subclasses have a responsibility to validate.
        """
        self._on_hessian = False

    def _set_on_hessian(self, on_hessian: bool) -> None:
        """sets ``on_hessian``.

        This function is intended to be called by subclasses.
        Subclasses have a responsibility to validate.

        Parameters
        ----------
        on_hessian : bool
            value of ``on_hessian``.
        """
        self._on_hessian = on_hessian

    def _validate_var_shape(self, var: np.ndarray) -> None:
        """validates whether the shape of variable is ``(num_var,)``.

        Parameters
        ----------
        var : np.ndarray
            np.ndarray of variables.

        Raises
        ------
        ValueError
            the shape of variable is not ``(num_var,)``.
        """
        if var.shape != (self.num_var,):
            raise ValueError(
                "the shape of variable must be ({self.num_var},). the shape of variable is {var.shape}"
            )

    @property
    def option(self) -> LossFunctionOption:
        """returns loss function option.

        Returns
        -------
        LossFunctionOption
            loss function option.
        """
        return self._option

    def set_from_option(self, option: LossFunctionOption) -> None:
        """sets option from LossFunctionOption and calls ``is_option_sufficient`` function.

        Parameters
        ----------
        option : LossFunctionOption
            option to set.
        """
        self._option = option
        self.is_option_sufficient()

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

    def set_from_standard_qtomography_option_data(
        self,
        qtomography: StandardQTomography,
        option: LossFunctionOption,
        data: List[Tuple[int, np.ndarray]],
        is_gradient_required: bool,
        is_hessian_required: bool,
    ) -> None:
        """sets settings of loss function.

        This implementation is empty by default.
        If necessary, implement it in subclasses.

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
        pass

    @abstractmethod
    def value(self, var: np.ndarray) -> np.float64:
        """returns the value of the loss function.

        Parameters
        ----------
        var : np.ndarray
            np.ndarray of variables.

        Returns
        -------
        np.float64
            the value of the loss function.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def gradient(self, var: np.ndarray) -> np.ndarray:
        """returns the gradient of the loss function.

        Parameters
        ----------
        var : np.ndarray
            np.ndarray of variables.

        Returns
        -------
        np.ndarray
            the gradient of the loss function. dtype=np.float64

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def hessian(self, var: np.ndarray) -> np.ndarray:
        """returns the Hessian of the loss function.

        Parameters
        ----------
        var : np.ndarray
            np.ndarray of variables.

        Returns
        -------
        np.ndarray
            the Hessian of the loss function. dtype=np.float64

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()
