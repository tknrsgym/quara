"""utility package about matrix."""
from functools import reduce
from operator import add

import numpy as np


def is_hermitian(matrix: np.ndarray, atol: float = 1e-14) -> bool:
    """returns whether the matrix is Hermitian.

    Parameters
    ----------
    matrix : np.ndarray
        input matrix.

    Returns
    -------
    bool
        True where ``matrix`` is Hermitian, False otherwise.
    """
    rows, columns = matrix.shape
    if rows != columns:
        return False

    adjoint = matrix.conj().T
    return np.allclose(matrix, adjoint, atol=atol, rtol=0.0)


def is_positive_semidefinite(matrix: np.ndarray, atol: float = None) -> bool:
    """Returns whether the matrix is positive semidifinite.

    Parameters
    ----------
    matrix : np.ndarray
        input matrix.
    atol : float, optional
        by default None.
        If atol is None, the default value ``1e-14`` is used.

    Returns
    -------
    bool
        True where ``matrix`` is positive semidifinite, False otherwise.
    """

    atol = atol if atol else 1e-14
    if is_hermitian(matrix, atol):
        return np.all(np.linalg.eigvals(matrix) >= 0)
    else:
        return False


def partial_trace1(matrix: np.ndarray, dim_Y: int) -> np.array:
    """calculates partial trace ``Tr_1[X \otimes Y] := Tr[X]Y``.

    Parameters
    ----------
    matrix : np.ndarray
        input matrix.
    dim_Y : int
        dim of ``Y``.

    Returns
    -------
    np.array
        partial trace.
    """
    # split input matrix to diagonal blocks
    block_list = []
    dim_X = int(matrix.shape[0] / dim_Y)
    for block_index in range(dim_X):
        block = matrix[
            block_index * dim_Y : (block_index + 1) * dim_Y,
            block_index * dim_Y : (block_index + 1) * dim_Y,
        ]
        block_list.append(block)

    # sum diagonal blocks
    P_trace = reduce(add, block_list)
    return P_trace


def is_tp(matrix: np.ndarray, dim: int, atol: float = 1e-13) -> bool:
    """returns whether the matrix is TP.
    if ``Tr_1[matrix] = I_2``, we think the matrix is TP.
    ``dim`` is a size of ``I_2``.

    Parameters
    ----------
    matrix : np.ndarray
        input matrix.
    dim : int
        dim of partial trace.
    atol : float, optional
        the absolute tolerance parameter, by default 1e-13.
        returns True, if ``absolute(identity matrix - partial trace) <= atol``.
        otherwise returns False.

    Returns
    -------
    bool
        True where ``matrix`` is TP, False otherwise.
    """
    p_trace = partial_trace1(matrix, dim)
    identity = np.eye(dim, dtype=np.complex128).reshape(dim, dim)
    return np.allclose(p_trace, identity, atol=atol, rtol=0.0)


def inner_product(left: np.ndarray, right: np.ndarray) -> np.complex128:
    """calculates Hilbert-Schmidt inner product ``<left, right> := Tr(left^{\dagger} @ right)`` = <<left|right>>.

    Parameters
    ----------
    left : np.ndarray
        left argument of inner product
    right : np.ndarray
        right argument of inner product

    Returns
    -------
    np.complex128
        Hilbert-Schmidt inner product
    """
    # calculate <<left|right>>
    i_product = np.inner(left.conj().flatten(), right.flatten())
    return i_product
