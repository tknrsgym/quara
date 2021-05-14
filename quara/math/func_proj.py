from typing import Callable

import numpy as np


def proj_to_self() -> Callable[[np.ndarray], np.ndarray]:
    """return the function of projection that maps to self.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        the function of projection that maps to self.
    """

    def _proj_to_self(var: np.ndarray) -> np.ndarray:
        return var

    return _proj_to_self


def proj_to_hyperplane(var_a: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """return the function of projection that maps to the hyperplane :math:`x \\cdot a = ||a||^2`.

    Parameters
    ----------
    var_a : np.ndarray
        the point that determines the hyperplane.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        the function of projection that maps to the hyperplane :math:`x \\cdot a = ||a||^2`.
    """

    def _proj_to_hyperplane(var: np.ndarray) -> np.ndarray:
        proj_value = var - np.dot(var_a, var) * var_a / np.dot(var_a, var_a) + var_a
        return proj_value

    return _proj_to_hyperplane


def proj_to_nonnegative() -> Callable[[np.ndarray], np.ndarray]:
    """return the function of projection that maps to the non-negative region.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        the function of projection that maps to the non-negative region.
    """

    def _proj_to_nonnegative(var: np.ndarray) -> np.ndarray:
        zero = np.zeros(var.shape)
        proj_value = np.maximum(var, zero)
        return proj_value

    return _proj_to_nonnegative
