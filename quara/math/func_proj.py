from typing import Callable

import numpy as np


def proj_to_self() -> Callable[[np.array], np.array]:
    """return the function of projection that maps to self.

    Returns
    -------
    Callable[[np.array], np.array]
        the function of projection that maps to self.
    """

    def _proj_to_self(var: np.array) -> np.array:
        return var

    return _proj_to_self


def proj_to_hyperplane(var_a: np.array) -> Callable[[np.array], np.array]:
    """return the function of projection that maps to the hyperplane x \cdot a = ||a||^2.

    Parameters
    ----------
    var_a : np.array
        the point that determines the hyperplane.

    Returns
    -------
    Callable[[np.array], np.array]
        the function of projection that maps to the hyperplane x \cdot a = ||a||^2.
    """

    def _proj_to_hyperplane(var: np.array) -> np.array:
        proj_value = var - np.dot(var_a, var) * var_a / np.dot(var_a, var_a) + var_a
        return proj_value

    return _proj_to_hyperplane


def proj_to_nonnegative() -> Callable[[np.array], np.array]:
    """return the function of projection that maps to the non-negative region.

    Returns
    -------
    Callable[[np.array], np.array]
        the function of projection that maps to the non-negative region.
    """

    def _proj_to_nonnegative(var: np.array) -> np.array:
        zero = np.zeros(var.shape)
        proj_value = np.maximum(var, zero)
        return proj_value

    return _proj_to_nonnegative
