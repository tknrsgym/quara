import numpy as np
import math


def multiply_veca_vecb(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """returns :math:`a \\cdot b`. this dot means inner product.

    Parameters
    ----------
    vec_a : np.ndarray
        vector a.
    vec_b : np.ndarray
        vector b.

    Returns
    -------
    float
        value of :math:`a \\cdot b`.

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


def multiply_veca_vecb_matc(
    vec_a: np.ndarray, vec_b: np.ndarray, mat_c: np.ndarray
) -> float:
    """returns :math:`a \\cdot C b`. this dot means inner product.

    Parameters
    ----------
    vec_a : np.ndarray
        vector a.
    vec_b : np.ndarray
        vector b.
    mat_c : np.ndarray
        matrix C.

    Returns
    -------
    float
        value of :math:`a \\cdot C b`.

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


def project_to_traceless_matrix(A: np.ndarray) -> np.ndarray:
    """returns a matrix projected to a trace-less matrix subspace.

    Parameters
    ----------
    A : np.ndarray
        Square matrix

    Returns
    ----------
    B : np.ndarray
        Square matrix, :math:`B = A - \\mathrm{Tr}[b_0 A] b_0`, where :math:`b_0 = \\frac{I}{\\sqrt{d}}`.

    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square matrix.")

    d = A.shape[0]
    b0 = np.eye(d) / math.sqrt(d)
    tr = np.trace(b0 @ A)
    B = A - tr * b0
    return B
