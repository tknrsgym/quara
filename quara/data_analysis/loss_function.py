from abc import abstractmethod

import numpy as np


class LossFunction:
    def __init__(
        self, num_var: int, on_gradient: bool, on_hessian: bool,
    ):
        """Constructor

        Parameters
        ----------
        num_var : int
            number of variables.
        on_gradient : bool
            whether or not to support gradient.
        on_hessian : bool
            whether or not to support Hessian.
        """
        self._num_var: int = num_var
        self._on_gradient: bool = on_gradient
        self._on_hessian: bool = on_hessian

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
    def on_gradient(self) -> bool:
        """returns whether or not to support gradient.

        Returns
        -------
        bool
            whether or not to support gradient.
        """
        return self._on_gradient

    @property
    def on_hessian(self) -> bool:
        """returns whether or not to support Hessian.

        Returns
        -------
        bool
            whether or not to support Hessian.
        """
        return self._on_hessian

    def _validate_var_shape(self, var: np.array) -> None:
        """validates whether the shape of variable is ``(num_var,)``.

        Parameters
        ----------
        var : np.array
            np.array of variables.

        Raises
        ------
        ValueError
            the shape of variable is not ``(num_var,)``.
        """
        if var.shape != (self.num_var,):
            raise ValueError(
                "the shape of variable must be ({self.num_var},). the shape of variable is {var.shape}"
            )

    @abstractmethod
    def value(self, var: np.array) -> np.float64:
        """returns the value of the loss function.

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
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def gradient(self, var: np.array) -> np.array:
        """returns the gradient of the loss function.

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
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def hessian(self, var: np.array) -> np.array:
        """returns the Hessian of the loss function.

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
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()
