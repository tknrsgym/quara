from abc import abstractmethod

import numpy as np

from quara.loss_function.loss_function import LossFunction, LossFunctionOption


class SimpleQuadraticLossFunctionOption(LossFunctionOption):
    def __init__(self):
        super().__init__()


class SimpleQuadraticLossFunction(LossFunction):
    def __init__(self, var_ref: np.ndarray):
        """Constructor

        this class has following properties.

        - ``on_value = True``
        - ``on_gradient = True``
        - ``on_hessian = True``

        Parameters
        ----------
        var_ref : np.ndarray
            variables
        """
        super().__init__(var_ref.size)
        self._on_value = True
        self._on_gradient = True
        self._on_hessian = True
        self._var_ref: np.ndarray = var_ref

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

    def value(self, var: np.ndarray) -> np.float64:
        """returns the value of the loss function.

        the value of the loss function is :math:`|| \\text{var} - \\text{var_ref} ||^2_2`.

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
        ValueError
            the shape of variable is not ``(num_var,)``.
        """
        self._validate_var_shape(var)
        val = np.sum((var - self._var_ref) ** 2)
        return val

    def gradient(self, var: np.ndarray) -> np.ndarray:
        """returns the gradient of the loss function.

        the value of the loss function is :math:`2(\\text{var} - \\text{var_ref})`.

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
        ValueError
            the shape of variable is not ``(num_var,)``.
        """
        self._validate_var_shape(var)
        val = 2 * (var - self._var_ref)
        return val

    def hessian(self, var: np.ndarray) -> np.ndarray:
        """returns the Hessian of the loss function.

        the value of the loss function is ``2I``, where ``I`` is the identity matrix of size ``num_var``.

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
        ValueError
            the shape of variable is not ``(num_var,)``.
        """
        self._validate_var_shape(var)
        val = 2 * np.eye(self.num_var, dtype=np.float64)
        return val
