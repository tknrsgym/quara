"""utility package about matrix."""
from functools import reduce
from operator import add

import numpy as np

from quara.settings import Settings


def is_hermitian(matrix: np.ndarray, atol: float = None) -> bool:
    """returns whether the matrix is Hermitian.

    Parameters
    ----------
    matrix : np.ndarray
        input matrix.
    atol : float, optional
        the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.

    Returns
    -------
    bool
        True where ``matrix`` is Hermitian, False otherwise.
    """
    atol = atol if atol else Settings.get_atol()

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
        the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.

    Returns
    -------
    bool
        True where ``matrix`` is positive semidifinite, False otherwise.
    """
    atol = atol if atol else Settings.get_atol()

    if is_hermitian(matrix, atol):
        # ignore eigvals close zero
        eigvals_array = np.linalg.eigvals(matrix)
        close_zero = np.where(np.isclose(eigvals_array, 0, atol=atol, rtol=0.0))
        eigvals_not_close_zero = np.delete(eigvals_array, close_zero)
        return np.all(eigvals_not_close_zero >= 0)
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


def is_tp(matrix: np.ndarray, dim: int, atol: float = None) -> bool:
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
        the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.
        returns True, if ``absolute(identity matrix - partial trace) <= atol``.
        otherwise returns False.

    Returns
    -------
    bool
        True where ``matrix`` is TP, False otherwise.
    """
    atol = atol if atol else Settings.get_atol()
    p_trace = partial_trace1(matrix, dim)
    identity = np.eye(dim, dtype=np.complex128).reshape(dim, dim)
    return np.allclose(p_trace, identity, atol=atol, rtol=0.0)
