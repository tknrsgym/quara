import numpy as np


def multiply_veca_vecb_matc(vec_a: np.array, vec_b: np.array, mat_c: np.array) -> float:
    """returns ``a \cdot C b``. '\cdot' means inner product.

    Parameters
    ----------
    vec_a : np.array
        vector a.
    vec_b : np.array
        vector b.
    mat_c : np.array
        matrix C.

    Returns
    -------
    float
        value of ``a \cdot C b``.

    Raises
    ------
    ValueError
        dimension of vec_a is not 1.(not vector)
    ValueError
        dimension of vec_b is not 1.(not vector)
    ValueError
        dimension of mat_c is not 2.(not matrix)
    """
    if vec_a.ndim != 1:
        raise ValueError(
            f"dimension of vec_a must be 1. dimension of vec_a is {vec_a.ndim}"
        )
    if vec_b.ndim != 1:
        raise ValueError(
            f"dimension of vec_b must be 1. dimension of vec_b is {vec_b.ndim}"
        )
    if mat_c.ndim != 2:
        raise ValueError(
            f"dimension of mat_c must be 2. dimension of mat_c is {mat_c.ndim}"
        )

    val = np.dot(vec_a, mat_c @ vec_b)
    return val
