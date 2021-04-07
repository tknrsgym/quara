from abc import abstractmethod

import numpy as np

from quara.loss_function.loss_function import LossFunction, LossFunctionOption


class SimpleQuadraticLossFunctionOption(LossFunctionOption):
    def __init__(self):
        super().__init__()


class SimpleQuadraticLossFunction(LossFunction):
    def __init__(self, var_ref: np.array):
        """Constructor

        this class has following properties.
        - ``on_value = True``
        - ``on_gradient = True``
        - ``on_hessian = True``

        Parameters
        ----------
        var_ref : np.array
            variables
        """
        super().__init__(var_ref.size)
        self._on_value = True
        self._on_gradient = True
        self._on_hessian = True
        self._var_ref: np.array = var_ref

    def _update_on_value_true(self) -> bool:
        """validates and updates ``on_value`` to True.

        see :func:`~quara.data_analysis.loss_function.LossFunction._update_on_value_true`
        """
        self._set_on_value(True)
        return True

    def _update_on_gradient_true(self) -> bool:
        """validates and updates ``on_gradient`` to True.

        see :func:`~quara.data_analysis.loss_function.LossFunction._update_on_gradient_true`
        """
        self._set_on_gradient(True)
        return True

    def _update_on_hessian_true(self) -> bool:
        """validates and updates ``on_hessian`` to True.

        see :func:`~quara.data_analysis.loss_function.LossFunction._update_on_hessian_true`
        """
        self._set_on_hessian(True)
        return True

    def value(self, var: np.array) -> np.float64:
        """returns the value of the loss function.

        the value of the loss function is ``|| var - var_ref ||^2_2``.

        Parameters
        ----------
        var : np.array
            np.array of variables.

        Returns
        -------
        np.float64
            the value of the loss function.

        Raises
        ------
        ValueError
            the shape of variable is not ``(num_var,)``.
        """
        self._validate_var_shape(var)
        val = np.sum((var - self._var_ref) ** 2)
        return val

    def gradient(self, var: np.array) -> np.array:
        """returns the gradient of the loss function.

        the value of the loss function is ``2 * (var - var_ref)``.

        Parameters
        ----------
        var : np.array
            np.array of variables.

        Returns
        -------
        np.array
            the gradient of the loss function. dtype=np.float64

        Raises
        ------
        ValueError
            the shape of variable is not ``(num_var,)``.
        """
        self._validate_var_shape(var)
        val = 2 * (var - self._var_ref)
        return val

    def hessian(self, var: np.array) -> np.array:
        """returns the Hessian of the loss function.

        the value of the loss function is ``2I``, where ``I`` is the identity matrix of size ``num_var``.

        Parameters
        ----------
        var : np.array
            np.array of variables.

        Returns
        -------
        np.array
            the Hessian of the loss function. dtype=np.float64

        Raises
        ------
        ValueError
            the shape of variable is not ``(num_var,)``.
        """
        self._validate_var_shape(var)
        val = 2 * np.eye(self.num_var, dtype=np.float64)
        return val
