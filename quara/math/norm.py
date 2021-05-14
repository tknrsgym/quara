import numpy as np


def l2_norm(x: np.ndarray, y: np.ndarray) -> np.float64:
    """calculates L2 norm.

    Parameters
    ----------
    x : np.ndarray
        vector.(1-dim numpy ndarray)
    y : np.ndarray
        vector.(1-dim numpy ndarray)

    Returns
    -------
    np.float64
        L2 norm of x and y.
    """
    norm = np.linalg.norm(x - y)
    return norm
