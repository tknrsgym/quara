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
    se_list = []
    for xs, ys in zip(xs_list, ys_list):
        se = calc_se(xs, ys)
        se_list.append(se)
    mse = np.mean(se_list, dtype=np.float64)
    std = np.std(se_list, dtype=np.float64, ddof=1)
    return mse, std


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


def calc_fisher_matrix(
    prob_dist: np.array, grad_prob_dist: List[np.array], eps: float = None
) -> np.array:
    """calculates Fisher matrix.

    Parameters
    ----------
    prob_dist : np.array
        probability distribution.
    grad_prob_dist : List[np.array]
        list of gradient of probability distribution.
        the length of list is the size of probability distribution.
        the size of np.array is the number of variables.
    eps : float, optional
        a parameter to avoid divergence about the inverse of probability, by default 1e-8

    Returns
    -------
    np.array
        Fisher matrix.

    Raises
    ------
    ValueError
        some elements of prob_dist are not between 0 and 1.
    ValueError
        the sum of prob_dist is not 1.
    ValueError
        the size of prob_dist and grad_prob_dist are not equal.
    ValueError
        eps is not a positive number.
    """
    eps = eps if eps is not None else 1e-8

    ### validate
    # each element of prob_dist must be between 0 and 1
    for index, entry in enumerate(prob_dist):
        if not (0.0 <= entry <= 1.0):
            raise ValueError(
                f"each element of prob_dist must be between 0 and 1. the sum of prob_dist[{index}]={entry}"
            )

    # the sum of prob_dist must be 1
    sum = np.sum(prob_dist)
    if not np.isclose(sum, 1.0, atol=eps, rtol=0.0):
        raise ValueError(f"the sum of prob_dist must be 1. the sum of prob_dist={sum}")

    # the size of prob_dist and grad_prob_dist must be equal
    size_prob_dist = prob_dist.shape[0]
    size_grad_prob_dist = len(grad_prob_dist)
    if size_prob_dist != size_grad_prob_dist:
        raise ValueError(
            f"the size of prob_dist and grad_prob_dist must be equal. the sum of prob_dist={size_prob_dist}, the sum of grad_prob_dist={size_grad_prob_dist}"
        )

    # eps must be a positive number
    if eps <= 0:
        raise ValueError(f"eps must be a positive number. eps={eps}")

    # replace
    replaced_prob_dist = replace_prob_dist(prob_dist, eps)

    ### calculate
    size_var = grad_prob_dist[0].shape[0]
    matrix = np.zeros((size_var, size_var))
    for prob, prob_dist in zip(replaced_prob_dist, grad_prob_dist):
        matrix += np.array([prob_dist]).T @ np.array([prob_dist]) / prob

    return matrix


def replace_prob_dist(prob_dist: np.array, eps: float = None) -> np.array:
    eps = eps if eps is not None else 1e-8

    size_prob_dist = prob_dist.shape[0]
    count_replace = np.count_nonzero(prob_dist < eps)
    replaced = np.zeros(size_prob_dist)

    for index, prob in enumerate(prob_dist):
        if prob < eps:
            replaced[index] = eps
        else:
            replaced[index] = prob_dist[index] - (eps * count_replace) / (
                size_prob_dist - count_replace
            )

    return replaced


def calc_fisher_matrix_total(
    prob_dists: List[np.array],
    grad_prob_dists: List[List[np.array]],
    weights: List[float],
    eps: float = None,
) -> np.array:
    """calculates total Fisher matrix.

    Parameters
    ----------
    prob_dists : List[np.array]
        list of probability distribution.
    grad_prob_dists : List[List[np.array]]
        list of list of gradient of probability distribution.
    weights : List[float]
        list of weight.
    eps : float, optional
        a parameter to avoid divergence about the inverse of probability, by default 1e-8

    Returns
    -------
    np.array
        total Fisher matrix.

    Raises
    ------
    ValueError
        size of prob_dists, grad_prob_dists and weights are not equal
    ValueError
        some weights are not non-nagative number.
    """
    eps = eps if eps is not None else 1e-8

    ### validate
    # size of prob_dists, grad_prob_dists and weights must be equal
    size_prob_dists = len(prob_dists)
    size_grad_prob_dists = len(grad_prob_dists)
    size_weights = len(weights)
    if size_prob_dists != size_grad_prob_dists:
        raise ValueError(
            f"size of prob_dists and grad_prob_dists must be equal. size of prob_dists={size_prob_dists}, size of grad_prob_dists={size_grad_prob_dists}"
        )
    if size_prob_dists != size_grad_prob_dists:
        raise ValueError(
            f"size of prob_dists and weights must be equal. size of prob_dists={size_prob_dists}, size of weights={size_weights}"
        )

    # each weight must be non-nagative number
    for index, weight in enumerate(weights):
        if weight < 0:
            raise ValueError(
                f"each weight must be non-negative number. weights[{index}]={weight}"
            )

    ### calculate
    matrix_size = prob_dists[0].shape[0]
    matrix = np.zeros((matrix_size, matrix_size))
    for index in range(size_prob_dists):
        matrix += weights[index] * calc_fisher_matrix(
            prob_dists[index], grad_prob_dists[index], eps=eps
        )

    return matrix
