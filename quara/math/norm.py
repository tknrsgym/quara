import numpy as np


def l2_norm(x: np.array, y: np.array) -> np.float64:
    """calculates L2 norm.

    Parameters
    ----------
    x : np.array
        vector.(1-dim numpy array)
    y : np.array
        vector.(1-dim numpy array)

    Returns
    -------
    np.float64
        L^2 norm of x and y.
    """
    norm = np.linalg.norm(x - y)
    return norm
