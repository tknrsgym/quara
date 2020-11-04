"""utility package about matrix."""
from typing import List, Tuple
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


def calc_se(xs: List[np.array], ys: List[np.array]) -> np.float64:
    """calculates Squared Error of ``xs`` and ``ys``.

    Parameters
    ----------
    xs : List[np.array]
        a list of array.
    ys : List[np.array]
        a list of array.

    Returns
    -------
    np.float64
        Squared Error of ``xs`` and ``ys``.
    """
    squared_errors = []
    for x, y in zip(xs, ys):
        squared_error = np.vdot(x - y, x - y)
        squared_errors.append(squared_error)

    se = np.sum(squared_errors, dtype=np.float64)
    return se


def calc_mse_prob_dists(
    xs_list: List[List[np.array]], ys_list: List[List[np.array]]
) -> np.float64:
    """calculates MSE(Mean Squared Error) of 'list of xs' and 'list of ys'.

    MSE is a sum of each MSE.
    Assume xs_list = [xs0, xs1] and ys_list = [ys0, ys1], returns 'MSE of xs0 and ys0' + 'MSE of xs1 and ys1'. 

    Parameters
    ----------
    xs_list : List[List[np.array]]
        a list of list of array.
    ys_list : List[List[np.array]]
        a list of list of array.

    Returns
    -------
    np.float64
        MSE of ``xs_list`` and ``ys_list``.
    """
    se_total = 0.0
    for xs, ys in zip(xs_list, ys_list):
        se_total += calc_se(xs, ys)
    # print(f"割る前: {se_total}")
    # print(f"len={len(xs_list)}")
    # print(f"割った後: {se_total / len(xs_list)}")
    mse = se_total / len(xs_list)
    print(f"calc_mse_prob_dists: {mse}")
    return mse


def calc_covariance_mat(q: np.array, n: int) -> np.array:
    """calculates covariance matrix of vector ``q``.

    Parameters
    ----------
    q : np.array
        vector.
    n : int
        number of data.

    Returns
    -------
    np.array
        covariance matrix = 1/n (diag(q) - q \cdot q^T)
    """
    mat = np.diag(q) - np.array([q]).T @ np.array([q])
    return mat / n


def calc_covariance_mat_total(empi_dists: List[Tuple[int, np.array]]) -> np.array:
    """calculates covariance matrix of total empirical distributions.

    Parameters
    ----------
    empi_dists : List[Tuple[int, np.array]]
        list of empirical distributions.
        each empirical distribution is a tuple of (data_num, empirical distribution).

    Returns
    -------
    np.array
        covariance matrix of total empirical distributions.
    """
    matrices = []
    for empi_dist in empi_dists:
        mat_single = calc_covariance_mat(empi_dist[1], empi_dist[0])
        matrices.append(mat_single)

    val = calc_direct_sum(matrices)
    return val


def calc_direct_sum(matrices: List[np.array]) -> np.array:
    """calculates direct sum of matrices.

    Parameters
    ----------
    matrices : List[np.array]
        matrices to calculate direct sum.

    Returns
    -------
    np.array
        direct sum.

    Raises
    ------
    ValueError
        ``matrices`` don't consist of matrices(dim=2).
    ValueError
        ``matrices`` don't consist of square matrices.
    """
    matrix_size = 0
    for i, diag in enumerate(matrices):
        if diag.ndim != 2:
            raise ValueError(
                "``matrices`` must consist of matrices(dim=2). dim of matrices[{i}] is {diag.ndim}"
            )
        if diag.shape[0] != diag.shape[0]:
            raise ValueError(
                "``matrices`` must consist of square matrices. shape of matrices[{i}] is {diag.shape}"
            )
        matrix_size += diag.shape[0]

    matrix = np.zeros((matrix_size, matrix_size))
    index = 0
    for diag in matrices:
        size = diag.shape[0]
        matrix[index : index + size, index : index + size] = diag
        index += size

    return matrix


def calc_conjugate(x: np.array, v: np.array) -> np.array:
    """calculates conjugate of matrices.

    Parameters
    ----------
    x : np.array
        parameter ``x``.
    v : np.array
        parameter ``v``.

    Returns
    -------
    np.array
        x @ v @ x^T
    """
    return x @ v @ x.T


def calc_left_inv(matrix: np.array) -> np.array:
    """calculates left inverse matrix.

    Parameters
    ----------
    matrix : np.array
        matrix to calculate left inverse matrix.

    Returns
    -------
    np.array
        left inverse matrix.

    Raises
    ------
    ValueError
        ``matrix`` is not full rank.
    """
    # check full rank
    rank = np.linalg.matrix_rank(matrix)
    size = min(matrix.shape)
    if size != rank:
        raise ValueError("``matrix`` must be full rank. size={size}, rank={rank}")

    # calculate left inverse
    left_inv = np.linalg.pinv(matrix.T @ matrix) @ matrix.T
    return left_inv
