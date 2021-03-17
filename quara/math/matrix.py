import numpy as np
import math


def multiply_veca_vecb(vec_a: np.array, vec_b: np.array) -> float:
    """returns ``a \cdot b``. '\cdot' means inner product.

    Parameters
    ----------
    vec_a : np.array
        vector a.
    vec_b : np.array
        vector b.

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
    """
    if vec_a.ndim != 1:
        raise ValueError(
            f"dimension of vec_a must be 1. dimension of vec_a is {vec_a.ndim}"
        )
    if vec_b.ndim != 1:
        raise ValueError(
            f"dimension of vec_b must be 1. dimension of vec_b is {vec_b.ndim}"
        )

    val = np.dot(vec_a, vec_b)
    return val


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


def project_to_traceless_matrix(A: np.array) -> np.array:
    """returns a matrix projected to a trace-less matrix subspace.

    Parameters
    ----------
    A : np.array
        Square matrix

    Returns
    ----------
    B : np.array
        Square matrix, B = A - Tr[b0 @ A] b0, b0 = I/sqrt(d).

    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square matrix.")

    d = A.shape[0]
    b0 = np.eye(d) / math.sqrt(d)
    tr = np.trace(b0 @ A)
    B = A - tr * b0
    return B
