"""utility package about matrix."""
import numpy as np


def is_hermitian(matrix: np.ndarray, atol: float = 1e-14) -> bool:
    """returns whether the matrix is Hermitian.

    Parameters
    ----------
    matrix : np.ndarray
        input matrix.
    atol : float, optional
        the absolute tolerance parameter, by default 1e-14.
        returns True, if ``absolute(matrix - conjugate of matrix) <= atol``.
        otherwise returns False.
    
    Returns
    -------
    bool
        True where ``matrix`` is Hermitian, False otherwise.
    """
    rows, columns = matrix.shape
    if rows != columns:
        return False

    adjoint = matrix.T.conj()
    return np.allclose(matrix, adjoint, atol=atol, rtol=0.0)


def is_positive_semidefinite(matrix: np.ndarray, atol: float = 1e-14) -> bool:
    """returns whether the matrix is positive semidifinite.

    Parameters
    ----------
    matrix : np.ndarray
        input matrix.
    atol : float, optional
        the absolute tolerance parameter, by default 1e-14.
        this function checks if ``matrix`` is Hermitian using :func:`~quara.utils.matrix_util.is_hermitian`.
        Use ``atol`` as arguments of :func:`~quara.utils.matrix_util.is_hermitian`.
    
    Returns
    -------
    bool
        True where ``matrix`` is positive semidifinite, False otherwise.
    """
    if is_hermitian(matrix, atol):
        return np.all(np.linalg.eigvals(matrix) >= 0)
    else:
        return False


def partial_trace(matrix: np.ndarray, dim: int) -> np.array:
    """calculates partial trace.
    
    Parameters
    ----------
    matrix : np.ndarray
        input matrix.
    dim : int
        dim of partial trace.
    
    Returns
    -------
    np.array
        partial trace.
    """
    # split input matrix to blocks
    entry = []
    for row_block in range(dim):
        for col_block in range(dim):
            block = matrix[
                row_block * dim : (row_block + 1) * dim,
                col_block * dim : (col_block + 1) * dim,
            ]
            entry.append(np.trace(block))

    # reshape
    p_trace = np.array(entry).reshape(dim, dim)
    return p_trace


def is_tp(matrix: np.ndarray, dim: int, atol: float = 1e-02) -> bool:
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
        the absolute tolerance parameter, by default 1e-02.
        returns True, if ``absolute(identity matrix - partial trace) <= atol``.
        otherwise returns False.
    
    Returns
    -------
    bool
        True where ``matrix`` is TP, False otherwise.
    """
    p_trace = partial_trace(matrix, dim)
    identity = np.eye(dim, dtype=np.complex128).reshape(dim, dim)
    return np.allclose(p_trace, identity, atol=atol, rtol=0.0)
