import copy
import itertools
from typing import List, Tuple

import numpy as np

from quara.settings import Settings
import quara.utils.matrix_util as mutil


class Basis:
    def __init__(self, basis: List[np.ndarray]):
        self._basis: Tuple[np.ndarray, ...] = tuple(copy.deepcopy(basis))

    @property
    def basis(self):  # read only
        return self._basis

    def __getitem__(self, key: int) -> np.ndarray:
        # return B_{\alpha}
        return self._basis[key]

    def __len__(self):
        """Returns number of basis.
        Returns
        -------
        int
            number of basis.
        """
        return len(self._basis)

    def __iter__(self):
        return iter(self._basis)

    def __str__(self):
        return str(self._basis)


class MatrixBasis(Basis):
    def __init__(self, basis: List[np.ndarray]):
        # make _basis immutable
        self._basis: Tuple[np.ndarray, ...] = tuple(copy.deepcopy(basis))
        for b in self._basis:
            b.setflags(write=False)

        self._dim = self[0].shape[0]

        # Validate
        if not self._is_squares():
            raise ValueError("Invalid argument. There is a non-square matrix.")

        if not self._is_same_size():
            raise ValueError(
                "Invalid argument. The sizes of the matrices are different."
            )

        if not self._is_basis():
            raise ValueError("Invalid argument. `basis` is not basis.")

    @property
    def dim(self) -> int:
        """Returns dim of matrix.

        Returns
        -------
        int
            dim of matrix.
        """
        return self._dim

    def to_vect(self) -> "VectorizedMatrixBasis":
        """Returns the class that vectorizes itself.

        Returns
        -------
        VectorizedMatrixBasis
            the class that vectorizes itself.
        """
        return to_vect(self)

    def _is_squares(self) -> bool:
        """Returns whether all matrices are square.

        Returns
        -------
        bool
            True where all matrices are square, False otherwise.
        """
        for mat in self:
            row, column = mat.shape
            if row != column:
                return False
        return True

    def _is_same_size(self) -> bool:
        """Returns whether all matrices are the same size.

        Returns
        -------
        bool
            True where all matrices are the same size, False otherwise.
        """
        for index in range(len(self) - 1):
            if self[index].shape != self[index + 1].shape:
                return False
        return True

    def _is_basis(self) -> bool:
        """Returns whether matrices are basis.

        Returns
        -------
        bool
            True where matrices are basis, False otherwise.
        """
        row_list = [mat.flatten() for mat in self]
        rank = np.linalg.matrix_rank(row_list)
        return rank >= self.dim ** 2

    def is_orthogonal(self) -> bool:
        """Returns whether matrices are orthogonal.

        Returns
        -------
        bool
            True where matrices are orthogonal, False otherwise.
        """
        for index, left in enumerate(self.basis[:-1]):
            for right in self.basis[index + 1 :]:
                i_product = np.vdot(left, right)
                if not np.isclose(i_product, 0, atol=Settings.get_atol()):
                    return False
        return True

    def is_normal(self) -> bool:
        """Returns whether matrices are normalized.

        Returns
        -------
        bool
            True where matrices are normalized, False otherwise.
        """
        for mat in self:
            i_product = np.vdot(mat, mat)
            if not np.isclose(i_product, 1, atol=Settings.get_atol()):
                return False
        return True

    def is_hermitian(self) -> bool:
        """Returns whether matrices are Hermitian.

        Returns
        -------
        bool
            True where matrices are Hermitian, False otherwise.
        """
        for mat in self:
            if not mutil.is_hermitian(mat):
                return False
        return True

    def is_0thpropI(self) -> bool:
        """Returns whether first matrix is constant multiple of identity matrix.

        Returns
        -------
        bool
            True where first matrix is constant multiple of identity matrix, False otherwise.
        """
        mat = self[0]
        scalar = mat[0, 0]
        identity = np.identity(self._dim, dtype=np.complex128)
        return np.allclose(scalar * identity, mat)

    def is_trace_less(self) -> bool:
        """Returns whether matrices are traceless except for first matrix.

        Returns
        -------
        bool
            True where matrices are traceless except for first matrix, False otherwise.
        """
        for index in range(1, len(self)):
            mat = self[index]
            tr = np.trace(mat)
            if tr != 0:
                return False
        return True

    def size(self) -> Tuple[int, int]:
        """Returns shape(=size) of basis.

        Returns
        -------
        Tuple[int, int]
            shape(=size) of basis.
        """
        return self[0].shape

    def __repr__(self):
        return f"{self.__class__.__name__}(basis={repr(list(self._basis))})"


class VectorizedMatrixBasis(Basis):
    def __init__(self, source: MatrixBasis):
        # Currently, only the MatrixBasis parameter is assumed.
        # It is not currently assumed that a vectorized np.ndarray is passed as a parameter.
        self._org_basis: MatrixBasis = source

        # vectorize
        temp_basis: List[np.ndarray] = []
        for b in self._org_basis:
            # "ravel" doesn't make copies, so it performs better than "flatten".
            # But, use "flatten" at this time to avoid breaking the original data(self._org_basis).
            # When performance issues arise, reconsider.
            vectorized_b = b.flatten()
            vectorized_b.setflags(write=False)
            temp_basis.append(vectorized_b)
        self._basis: Tuple[np.ndarray, ...] = tuple(temp_basis)

    @property
    def org_basis(self) -> MatrixBasis:  # read only
        """Original matrix basis.

        Returns
        -------
        MatrixBasis
            Original matrix basis.
        """
        return self._org_basis

    @property
    def dim(self) -> int:
        """Returns dim of matrix.

        Returns
        -------
        int
            dim of matrix
        """
        return self._org_basis.dim

    def is_hermitian(self) -> bool:
        """Returns whether matrices are Hermitian.

        Returns
        -------
        bool
            True where matrices are Hermitian, False otherwise.
        """
        return self._org_basis.is_hermitian()

    def is_orthogonal(self) -> bool:
        """Returns whether matrices are orthogonal.

        Returns
        -------
        bool
            True where matrices are orthogonal, False otherwise.
        """
        return self._org_basis.is_orthogonal()

    def is_normal(self) -> bool:
        """Returns whether matrices are normalized.

        Returns
        -------
        bool
            True where matrices are normalized, False otherwise.
        """
        return self._org_basis.is_normal()

    def is_scalar_mult_of_identity(self) -> bool:
        """Returns whether first matrix is constant multiple of identity matrix.

        Returns
        -------
        bool
            True where first matrix is constant multiple of identity matrix, False otherwise.
        """
        return self._org_basis.is_scalar_mult_of_identity()

    def is_trace_less(self) -> bool:
        """Returns whether matrices are traceless except for first matrix.

        Returns
        -------
        bool
            True where matrices are traceless except for first matrix, False otherwise.
        """
        return self._org_basis.is_trace_less()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source=MatrixBasis(basis={repr(list(self._org_basis))}))"


def calc_matrix_expansion_coefficient(
    from_mat: np.ndarray, basis: MatrixBasis
) -> np.ndarray:
    """return expansion coefficients of a matrix w.r.t. the matrix basis.

    Parameters
    ----------
    from_mat : np.ndarray((dim, dim), np.complex128)
        A square complex matrix

    basis: MatrixBasis
        A orthonormal matrix basis

    Returns
    ----------
    np.ndarray((dim * dim, 1), np.complex)
    """
    shape = from_mat.shape
    assert shape[0] == shape[1]
    assert basis.dim == shape[0]
    assert basis.is_normal
    assert basis.is_orthogonal

    l = []
    for bi in basis:
        c = np.trace(np.conjugate(np.transpose(bi)) @ from_mat)
        l.append(c)
    coeff = np.array(l, np.complex128)
    return coeff


def calc_hermitian_matrix_expansion_coefficient_hermitian_basis(
    from_mat: np.ndarray, basis: MatrixBasis
) -> np.ndarray:
    """return expansion coefficients of an Hermitian matrix w.r.t. the Hermitian matrix basis.

    Parameters
    ----------
    from_mat : np.ndarray((dim, dim), np.complex128)
        An Hermitian matrix

    basis: MatrixBasis
        An Hermitian orthonormal matrix basis

    Returns
    ----------
    np.ndarray((dim * dim, 1), np.float)
    """
    assert mutil.is_hermitian(from_mat)
    assert basis.is_hermitian

    coeff_comp = calc_matrix_expansion_coefficient(from_mat, basis)
    coeff_real = mutil.truncate_hs(coeff_comp)
    return coeff_real


def calc_mat_from_coefficient_basis(
    coeff: np.ndarray, basis: MatrixBasis
) -> np.ndarray:
    """return a matrix corresponding to the coefficient and matrix basis.

    Parameters
    ----------
    coeff : np.ndarray
        A coefficient vector.
        The shape of np.ndarray is ``(dim * dim, 1)``.

    basis : MatrixBasis
        A square matrix basis with dimension dim.

    Returns
    ----------
    np.ndarray
        A square matrix
        The shape of np.ndarray is ``(dim, dim)``.
    """
    dim = basis.dim
    assert len(coeff.shape) == 1
    assert coeff.shape[0] == dim * dim
    assert basis.is_orthogonal

    mat = np.zeros((dim, dim), dtype=np.complex128)
    for i, bi in enumerate(basis):
        ci = coeff[i]
        mat += ci * bi

    return mat


def to_vect(source: MatrixBasis) -> VectorizedMatrixBasis:
    """Convert MatrixBasis to VectorizedMatrixBasis

    Returns
    -------
    VectorizedMatrixBasis
        VectorizedMatrixBasis converted from MatrixBasis.
    """
    return VectorizedMatrixBasis(source)


def _calc_tensor_product_from_1q_basis(n_qubit: int, basis_1q: List[np.ndarray]):
    basis = basis_1q
    for _ in range(1, n_qubit):
        basis = [
            np.kron(val1, val2) for val1, val2 in itertools.product(basis, basis_1q)
        ]
    return basis


def get_comp_basis(dim: int = 2, mode: str = "row_major") -> MatrixBasis:
    """Returns computational basis.

    Parameters
    ----------
    dim : int, optional
        dim of computational basis, by default 2.
    mode : str, optional
        specify whether the order of basis is "row_major" or "column_major", by default "row_major".

    Returns
    -------
    MatrixBasis
        computational basis with specific dim.

    Raises
    ------
    ValueError
        ``mode`` is unsupported.
    """
    comp_basis_list = []
    if mode == "row_major":
        # row-major
        for row in range(dim):
            for col in range(dim):
                tmp_basis = np.zeros((dim, dim), dtype=np.complex128)
                tmp_basis[row, col] = 1
                comp_basis_list.append(tmp_basis)
    elif mode == "column_major":
        # column-major
        for col in range(dim):
            for row in range(dim):
                tmp_basis = np.zeros((dim, dim), dtype=np.complex128)
                tmp_basis[row, col] = 1
                comp_basis_list.append(tmp_basis)
    else:
        raise ValueError(f"unsupported mode={mode}")

    comp_basis = MatrixBasis(comp_basis_list)
    return comp_basis


def get_pauli_basis(n_qubit: int = 1) -> MatrixBasis:
    """Returns Pauli basis.

    Parameters
    ----------
    n_qubit : int, optional
        number of qubit for Pauli basis, by default 1.

    Returns
    -------
    MatrixBasis
        Pauli basis :math:`[I, X, Y, Z]`
    """
    identity = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    basis_1q = [identity, pauli_x, pauli_y, pauli_z]

    basis = _calc_tensor_product_from_1q_basis(n_qubit, basis_1q)
    matrix_basis = MatrixBasis(basis)
    return matrix_basis


def get_normalized_pauli_basis(n_qubit: int = 1) -> MatrixBasis:
    """Returns normalized Pauli basis.

    Parameters
    ----------
    n_qubit : int, optional
        number of qubit for normalized Pauli basis, by default 1.

    Returns
    -------
    MatrixBasis
        ``n_qubit`` of Pauli basis :math:`\\frac{1}{\\sqrt{2}}[I, X, Y, Z]`
    """
    identity = 1 / np.sqrt(2) * np.array([[1, 0], [0, 1]], dtype=np.complex128)
    pauli_x = 1 / np.sqrt(2) * np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = 1 / np.sqrt(2) * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli_z = 1 / np.sqrt(2) * np.array([[1, 0], [0, -1]], dtype=np.complex128)
    basis_1q = [identity, pauli_x, pauli_y, pauli_z]

    basis = _calc_tensor_product_from_1q_basis(n_qubit, basis_1q)
    matrix_basis = MatrixBasis(basis)
    return matrix_basis


def get_hermitian_basis(dim: int = 2) -> MatrixBasis:
    """Returns Hermitian basis.

    Parameters
    ----------
    dim : int, optional
        dim of Hermitian basis, by default 2.

    Returns
    -------
    MatrixBasis
        Hermitian basis.
    """
    basis = []
    for col in range(dim):
        # not diagonal
        for row in range(col):
            matrix_real = np.zeros((dim, dim), dtype=np.complex128)
            matrix_real[row, col] = 1
            matrix_real[col, row] = 1
            basis.append(matrix_real)

            matrix_imag = np.zeros((dim, dim), dtype=np.complex128)
            matrix_imag[row, col] = -1j
            matrix_imag[col, row] = 1j
            basis.append(matrix_imag)

        # diagonal
        matrix_diag = np.zeros((dim, dim), dtype=np.complex128)
        matrix_diag[col, col] = 1
        basis.append(matrix_diag)

    pauli_basis = MatrixBasis(basis)
    return pauli_basis


def get_normalized_hermitian_basis(dim: int = 2) -> MatrixBasis:
    """Returns normalized Hermitian basis.

    Parameters
    ----------
    dim : int, optional
        dim of normalized Hermitian basis, by default 2.

    Returns
    -------
    MatrixBasis
        normalized Hermitian basis.
    """
    basis = []
    for col in range(dim):
        # not diagonal
        for row in range(col):
            matrix_real = np.zeros((dim, dim), dtype=np.complex128)
            matrix_real[row, col] = 1 / np.sqrt(2)
            matrix_real[col, row] = 1 / np.sqrt(2)
            basis.append(matrix_real)

            matrix_imag = np.zeros((dim, dim), dtype=np.complex128)
            matrix_imag[row, col] = -1j / np.sqrt(2)
            matrix_imag[col, row] = 1j / np.sqrt(2)
            basis.append(matrix_imag)

        # diagonal
        matrix_diag = np.zeros((dim, dim), dtype=np.complex128)
        matrix_diag[col, col] = 1
        basis.append(matrix_diag)

    pauli_basis = MatrixBasis(basis)
    return pauli_basis


def get_gell_mann_basis() -> MatrixBasis:
    """Returns Gell-Mann matrices basis.

    Returns
    -------
    MatrixBasis
        Gell-Mann matrices basis
    """
    identity = np.sqrt(2 / 3) * np.eye(3, dtype=np.complex128)
    l_1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.complex128)
    l_2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=np.complex128)
    l_3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.complex128)
    l_4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.complex128)
    l_5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=np.complex128)
    l_6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.complex128)
    l_7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=np.complex128)
    l_8 = (
        1
        / np.sqrt(3)
        * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=np.complex128)
    )

    gell_mann_basis = MatrixBasis([identity, l_1, l_2, l_3, l_4, l_5, l_6, l_7, l_8])
    return gell_mann_basis


def get_normalized_gell_mann_basis() -> MatrixBasis:
    """Returns normalized Gell-Mann matrices basis.

    Returns
    -------
    MatrixBasis
        Normalized Gell-Mann matrices basis
    """
    identity = np.sqrt(2 / 3) * np.eye(3, dtype=np.complex128)
    l_1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.complex128)
    l_2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=np.complex128)
    l_3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.complex128)
    l_4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.complex128)
    l_5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=np.complex128)
    l_6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.complex128)
    l_7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=np.complex128)
    l_8 = (
        1
        / np.sqrt(3)
        * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=np.complex128)
    )
    source = [
        1 / np.sqrt(2) * x for x in [identity, l_1, l_2, l_3, l_4, l_5, l_6, l_7, l_8]
    ]
    gell_mann_basis = MatrixBasis(source)
    return gell_mann_basis


def get_generalized_gell_mann_basis(n_qubit: int = 1, dim: int = 2) -> MatrixBasis:
    """Returns Generalized Gell-Mann matrices basis.

    Parameters
    ----------
    n_qubit : int, optional
        number of qubit for Generalized Gell-Mann matrices basis, by default 1.
    dim : int, optional
        dim of Generalized Gell-Mann matrices basis, by default 2.

    Returns
    -------
    MatrixBasis
        Generalized Gell-Mann matrices basis.
        see https://mathworld.wolfram.com/GeneralizedGell-MannMatrix.html
    """
    basis_1q = []
    for col in range(dim):
        for row in range(col):
            # symmetric matrix
            matrix_real = np.zeros((dim, dim), dtype=np.complex128)
            matrix_real[row, col] = 1
            matrix_real[col, row] = 1
            basis_1q.append(matrix_real)

            # antisymmetric matrix
            matrix_imag = np.zeros((dim, dim), dtype=np.complex128)
            matrix_imag[row, col] = -1j
            matrix_imag[col, row] = 1j
            basis_1q.append(matrix_imag)

        # diagonal matrix
        if col == 0:
            matrix_diag = np.sqrt(2 / dim) * np.eye(dim, dtype=np.complex128)
        else:
            matrix_diag = np.zeros((dim, dim), dtype=np.complex128)
            for diag in range(col):
                matrix_diag[diag, diag] = 1
            matrix_diag[col, col] = -col
            matrix_diag = np.sqrt(2 / (col * (col + 1))) * matrix_diag
        basis_1q.append(matrix_diag)

    basis = _calc_tensor_product_from_1q_basis(n_qubit, basis_1q)
    matrix_basis = MatrixBasis(basis)
    return matrix_basis


def get_normalized_generalized_gell_mann_basis(
    n_qubit: int = 1, dim: int = 2
) -> MatrixBasis:
    """Returns Normalized Generalized Gell-Mann matrices basis.

    Parameters
    ----------
    n_qubit : int, optional
        number of qubit for Normalized Generalized Gell-Mann matrices basis, by default 1.
    dim : int, optional
        dim of Normalized Generalized Gell-Mann matrices basis, by default 2.

    Returns
    -------
    MatrixBasis
        Normalized Generalized Gell-Mann matrices basis.
        see https://mathworld.wolfram.com/GeneralizedGell-MannMatrix.html
    """
    basis_1q = []
    for col in range(dim):
        for row in range(col):
            # symmetric matrix
            matrix_real = np.zeros((dim, dim), dtype=np.complex128)
            matrix_real[row, col] = 1 / np.sqrt(2)
            matrix_real[col, row] = 1 / np.sqrt(2)
            basis_1q.append(matrix_real)

            # antisymmetric matrix
            matrix_imag = np.zeros((dim, dim), dtype=np.complex128)
            matrix_imag[row, col] = -1j / np.sqrt(2)
            matrix_imag[col, row] = 1j / np.sqrt(2)
            basis_1q.append(matrix_imag)

        # diagonal matrix
        if col == 0:
            matrix_diag = np.sqrt(1 / dim) * np.eye(dim, dtype=np.complex128)
        else:
            matrix_diag = np.zeros((dim, dim), dtype=np.complex128)
            for diag in range(col):
                matrix_diag[diag, diag] = 1
            matrix_diag[col, col] = -col
            matrix_diag = np.sqrt(1 / (col * (col + 1))) * matrix_diag
        basis_1q.append(matrix_diag)

    basis = _calc_tensor_product_from_1q_basis(n_qubit, basis_1q)
    matrix_basis = MatrixBasis(basis)
    return matrix_basis


def convert_vec(
    from_vec: np.ndarray, from_basis: MatrixBasis, to_basis: MatrixBasis
) -> np.ndarray:
    """converts vector representation from ``from_basis`` to ``to_basis``.

    Parameters
    ----------
    from_vec : np.ndarray
        vector representation before converts vector representation.
    from_basis : MatrixBasis
        basis before converts vector representation.
    to_basis : MatrixBasis
        basis after converts vector representation.

    Returns
    -------
    np.ndarray
        vector representation after converts vector representation.
        ``dtype`` is ``float64``.
    """
    # whether length of from_basis equals length of to_basis
    if len(from_basis) != len(to_basis):
        raise ValueError(
            f"length of from_basis must equal length of to_basis. length of from_basis={len(from_basis)}. length of to_basis is {len(to_basis)}"
        )
    len_basis = len(from_basis)

    # whether dim of from_basis equals dim of to_basis
    if from_basis.dim != to_basis.dim:
        raise ValueError(
            f"dim of from_basis must equal dim of to_basis. dim of from_basis={from_basis.dim}. dim of to_basis is {to_basis.dim}"
        )

    # "converted_vec"_{\alpha} = \sum_{\beta} Tr["to_basis"_{\beta}^{\dagger} "from_basis"_{\alpha}] "from_vec"_{\alpha}
    representation_matrix = [
        np.vdot(val1, val2) for val1, val2 in itertools.product(to_basis, from_basis)
    ]
    rep_mat = np.array(representation_matrix).reshape(len_basis, len_basis)
    converted_vec = rep_mat @ from_vec

    return converted_vec
