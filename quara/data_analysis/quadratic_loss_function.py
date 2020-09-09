from abc import abstractmethod

import numpy as np

from quara.data_analysis.loss_function import LossFunction


class QuadraticLossFunction(LossFunction):
    def __init__(self, var_ref: np.array):
        """Constructor

        this class has following properties.
        - ``on_gradient = True``
        - ``on_hessian = True``

        Parameters
        ----------
        var_ref : np.array
            [description]
        """
        super().__init__(var_ref.size, True, True)
        self._var_ref: np.array = var_ref

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
