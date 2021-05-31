"""utility package about matrix."""
import copy
from typing import List, Tuple, Union
from functools import reduce
from operator import add

import numpy as np

from quara.settings import Settings


def is_unitary(matrix: np.ndarray, atol: float = None) -> bool:
    """returns whether the matrix is unitary.

    Parameters
    ----------
    matrix : np.ndarray
        input matrix.
    atol : float, optional
        the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.

    Returns
    -------
    bool
        True where ``matrix`` is unitary, False otherwise.
    """
    atol = Settings.get_atol() if atol is None else atol
    rows, columns = matrix.shape

    if rows != columns:
        return False

    adjoint = matrix.conj().T
    I = np.eye(rows)
    return np.allclose(matrix @ adjoint, I, atol=atol, rtol=0.0)


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
    atol = Settings.get_atol() if atol is None else atol
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
    atol = Settings.get_atol() if atol is None else atol

    if is_hermitian(matrix, atol):
        # ignore eigvals close zero
        eigvals_array = np.linalg.eigvalsh(matrix)
        close_zero = np.where(np.isclose(eigvals_array, 0, atol=atol, rtol=0.0))
        eigvals_not_close_zero = np.delete(eigvals_array, close_zero)
        return np.all(eigvals_not_close_zero >= 0)
    else:
        return False


def partial_trace1(matrix: np.ndarray, dim_Y: int) -> np.ndarray:
    """calculates partial trace :math:`\\mathrm{Tr}_1[X \\otimes Y] := \\mathrm{Tr}[X]Y`.

    Parameters
    ----------
    matrix : np.ndarray
        input matrix.
    dim_Y : int
        dim of ``Y``.

    Returns
    -------
    np.ndarray
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
    if :math:`\\mathrm{Tr}_1[\\text{matrix}] = I_2`, we think the matrix is TP.
    ``dim`` is a size of :math:`I_2`.

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
    atol = Settings.get_atol() if atol is None else atol
    p_trace = partial_trace1(matrix, dim)
    identity = np.eye(dim, dtype=np.complex128).reshape(dim, dim)
    return np.allclose(p_trace, identity, atol=atol, rtol=0.0)


def truncate_imaginary_part(matrix: np.ndarray, eps: float = None) -> np.float64:
    """truncates the imaginary part of the matrix entries.

    Parameters
    ----------
    matrix : np.ndarray
        matrix to truncate the imaginary part.
    eps : float, optional
        threshold to truncate, by default :func:`~quara.settings.Settings.get_atol`

    Returns
    -------
    np.float64
        truncated matrix.
    """
    eps = Settings.get_atol() if eps is None else eps

    return np.where(np.abs(matrix.imag) < eps, matrix.real, matrix)


def truncate_computational_fluctuation(
    matrix: np.ndarray, eps: float = None
) -> np.float64:
    """truncates the computational fluctuation (real part) of the matrix entries.

    Parameters
    ----------
    matrix : np.ndarray
        matrix to truncate the computational fluctuation.
    eps : float, optional
        threshold to truncate, by default :func:`~quara.settings.Settings.get_atol`

    Returns
    -------
    np.float64
        truncated matrix.
    """
    eps = Settings.get_atol() if eps is None else eps
    return np.where(np.abs(matrix) < eps, 0.0, matrix)


def truncate_hs(
    hs: np.ndarray,
    eps_proj_physical: float = None,
    is_zero_imaginary_part_required: bool = True,
) -> np.ndarray:
    """truncate HS matrix to a real matrix.

    Parameters
    ----------
    hs : np.ndarray
        HS matrix to truncate.
    eps_proj_physical : float, optional
        threshold to truncate, by default :func:`~quara.settings.Settings.get_atol`
    is_zero_imaginary_part_required : bool, optional
        whether the imaginary part should be truncated to zero, by default True

    Returns
    -------
    np.ndarray
        truncated real matrix.

    Raises
    ------
    ValueError
        `is_zero_imaginary_part_required` == True and some imaginary parts of entries of matrix != 0.
    """
    tmp_hs = truncate_imaginary_part(hs, eps_proj_physical)

    if is_zero_imaginary_part_required == True and np.any(tmp_hs.imag != 0):
        raise ValueError(
            f"some imaginary parts of entries of matrix != 0. converted hs={tmp_hs}"
        )

    if is_zero_imaginary_part_required == True:
        tmp_hs = tmp_hs.real.astype(np.float64)

    truncated_hs = truncate_computational_fluctuation(tmp_hs, eps_proj_physical)
    return truncated_hs


def truncate_and_normalize(matrix: np.ndarray, eps: float = None) -> np.array:
    """truncates entries smaller than eps and normalizes to the matrix whose sum of each row is 1.

    Parameters
    ----------
    matrix : np.ndarray
        the matrix to be truncated and normalized.
    eps : float, optional
        threshold to truncate, uses :func:`~quara.settings.Settings.get_atol` by default.

    Returns
    -------
    np.array
        truncated and normalized matrix
    """
    eps = Settings.get_atol() if eps is None else eps

    # truncate entries smaller than eps and normalize matrix along rows
    matrix = np.where(matrix < eps, 0, matrix)
    if matrix.ndim == 1:
        matrix = matrix / np.sum(matrix)
    else:
        for index, row in enumerate(matrix):
            matrix[index] = row / np.sum(row)
    return matrix


def calc_se(xs: List[np.ndarray], ys: List[np.ndarray]) -> np.float64:
    """calculates Squared Error of ``xs`` and ``ys``.

    Parameters
    ----------
    xs : List[np.ndarray]
        a list of ndarray.
    ys : List[np.ndarray]
        a list of ndarray.

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
    xs_list: List[List[np.ndarray]], ys_list: List[List[np.ndarray]]
) -> np.float64:
    """calculates MSE(Mean Squared Error) of 'list of xs' and 'list of ys'.

    MSE is a sum of each MSE.
    Assume xs_list = [xs0, xs1] and ys_list = [ys0, ys1], returns 'MSE of xs0 and ys0' + 'MSE of xs1 and ys1'.

    Parameters
    ----------
    xs_list : List[List[np.ndarray]]
        a list of list of ndarray.
    ys_list : List[List[np.ndarray]]
        a list of list of ndarray.

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


def calc_covariance_mat(q: np.ndarray, n: int) -> np.ndarray:
    """calculates covariance matrix of vector ``q``.

    Parameters
    ----------
    q : np.ndarray
        vector.
    n : int
        number of data.

    Returns
    -------
    np.ndarray
        covariance matrix is :math:`\\frac{1}{n} (diag(q) - q \\cdot q^T)`
    """
    mat = np.diag(q) - np.array([q]).T @ np.array([q])
    return mat / n


def calc_covariance_mat_total(empi_dists: List[Tuple[int, np.ndarray]]) -> np.ndarray:
    """calculates covariance matrix of total empirical distributions.

    Parameters
    ----------
    empi_dists : List[Tuple[int, np.ndarray]]
        list of empirical distributions.
        each empirical distribution is a tuple of (data_num, empirical distribution).

    Returns
    -------
    np.ndarray
        covariance matrix of total empirical distributions.
    """
    matrices = []
    for empi_dist in empi_dists:
        mat_single = calc_covariance_mat(empi_dist[1], empi_dist[0])
        matrices.append(mat_single)

    val = calc_direct_sum(matrices)
    return val


def calc_direct_sum(matrices: List[np.ndarray]) -> np.ndarray:
    """calculates direct sum of matrices.

    Parameters
    ----------
    matrices : List[np.ndarray]
        matrices to calculate direct sum.

    Returns
    -------
    np.ndarray
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


def calc_conjugate(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """calculates conjugate of matrices.

    Parameters
    ----------
    x : np.ndarray
        parameter ``x``.
    v : np.ndarray
        parameter ``v``.

    Returns
    -------
    np.ndarray
        :math:`x v x^T`
    """
    return x @ v @ x.T


def calc_left_inv(matrix: np.ndarray) -> np.ndarray:
    """calculates left inverse matrix.

    Parameters
    ----------
    matrix : np.ndarray
        matrix to calculate left inverse matrix.

    Returns
    -------
    np.ndarray
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
    prob_dist: np.ndarray, grad_prob_dist: List[np.ndarray], eps: float = None
) -> np.ndarray:
    """calculates Fisher matrix.

    Parameters
    ----------
    prob_dist : np.ndarray
        probability distribution.
    grad_prob_dist : List[np.ndarray]
        list of gradient of probability distribution.
        the length of list is the size of probability distribution.
        the size of np.ndarray is the number of variables.
    eps : float, optional
        a parameter to avoid divergence about the inverse of probability, by default 1e-8

    Returns
    -------
    np.ndarray
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


def replace_prob_dist(prob_dist: np.ndarray, eps: float = None) -> np.ndarray:
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
    prob_dists: List[np.ndarray],
    grad_prob_dists: List[List[np.ndarray]],
    weights: List[float],
    eps: float = None,
) -> np.ndarray:
    """calculates total Fisher matrix.

    Parameters
    ----------
    prob_dists : List[np.ndarray]
        list of probability distribution.
    grad_prob_dists : List[List[np.ndarray]]
        list of list of gradient of probability distribution.
    weights : List[float]
        list of weight.
    eps : float, optional
        a parameter to avoid divergence about the inverse of probability, by default 1e-8

    Returns
    -------
    np.ndarray
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


def convert_list_by_permutation_matrix(
    old_list: List, permutation_matrix: np.ndarray
) -> List:
    """converts list by permutation_matrix.

    this function executes "permutation_matrix @ old_list"-like operation.
    for example, if old_list = [a, b] and permutation_matrix = np.array([[0, 1], [1, 0]]), then this function returns [b, a].

    Parameters
    ----------
    old_list : List
        a list before permutation.
    permutation_matrix : np.ndarray
        permutation_matrix to permutate a list.

    Returns
    -------
    List
        [description]
    """
    # this function executes "permutation_matrix @ old_list"-like operation.
    # for example, if old_list = [a, b] and permutation_matrix = np.array([[0, 1], [1, 0]]), then this function returns [b, a].
    row_size, col_size = permutation_matrix.shape
    new_list = [True] * row_size
    for row in range(row_size):
        # find new_list[row]
        for col in range(col_size):
            if permutation_matrix[row, col] == 1:
                new_list[row] = old_list[col]
                break
    # print(f"new_list={new_list}")
    return new_list


def _U(dim1, dim2, i, j):
    matrix = np.zeros((dim1, dim2))
    matrix[i, j] = 1
    return matrix


def _K(dim1: int, dim2: int) -> np.ndarray:
    matrix = np.zeros((dim1 * dim2, dim1 * dim2))
    for row in range(dim1):
        for col in range(dim2):
            matrix += np.kron(_U(dim1, dim2, row, col), _U(dim2, dim1, col, row))

    return matrix


def _left_permutation_matrix(position: int, size_list: List[int]) -> np.ndarray:
    # identity matrix for head of permutation matrix
    if position < 2:
        I_head = np.eye(1)
    else:
        size = reduce(add, size_list[: position - 1])
        I_head = np.eye(size)

    # create matrix K
    K_matrix = _K(size_list[position], size_list[position - 1])

    # identity matrix for tail of permutation matrix
    if position < len(size_list) - 1:
        size = reduce(add, size_list[position + 1 :])
        I_tail = np.eye(size)
    else:
        I_tail = np.eye(1)

    # calculate permutation matrix
    left_perm_matrix = np.kron(np.kron(I_head, K_matrix), I_tail)
    return left_perm_matrix


def _check_cross_system_position(
    system_order: List[int],
) -> Union[int, None]:
    # check cross system position
    # for example, if [0, 10, 5] is a list of names of ElementalSystem, then this functions returns 2(position of value 5)
    former_name = None
    for current_position, system_name in enumerate(system_order):
        current_name = system_name
        if not former_name is None and former_name > current_name:
            return current_position
        else:
            former_name = current_name

    # if cross ElementalSystem position does not exist, returns None
    return None


def calc_permutation_matrix(
    system_order: List[int], size_list: List[int]
) -> np.ndarray:
    """calculate permutation matrix.

    permutation matrix can reorder the system order to [0, 1,..., n].

    Parameters
    ----------
    system_order : List[int]
        system_order before permutation.
    size_list : List[int]
        size of systems.

    Returns
    -------
    np.ndarray
        permutation matrix.
    """
    tmp_system_order = copy.copy(system_order)
    tmp_size_list = copy.copy(size_list)
    total_dim = np.prod(size_list)
    perm_matrix = np.eye(total_dim)

    # calc permutation matrix
    position = _check_cross_system_position(tmp_system_order)
    while not position is None:
        left_perm = _left_permutation_matrix(position, tmp_size_list)
        perm_matrix = left_perm @ perm_matrix
        # swap tmp_system_order
        tmp_system_order[position - 1], tmp_system_order[position] = (
            tmp_system_order[position],
            tmp_system_order[position - 1],
        )
        # swap size_list
        tmp_size_list[position - 1], tmp_size_list[position] = (
            tmp_size_list[position],
            tmp_size_list[position - 1],
        )
        position = _check_cross_system_position(tmp_system_order)

    return perm_matrix


def calc_mat_from_vector_adjoint(vec: np.ndarray) -> np.ndarray:
    return np.array([vec]).T @ np.array([vec]).conjugate()
