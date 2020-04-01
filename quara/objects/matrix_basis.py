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
            dim of matrix
        """
        return self._dim

    def to_vect(self) -> "VectorizedMatrixBasis":
        """Returns the class that vectorizes itself.

        Returns
        -------
        VectorizedMatrixBasis
            the class that vectorizes itself
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

    def is_scalar_mult_of_identity(self) -> bool:
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


def to_vect(source: MatrixBasis) -> VectorizedMatrixBasis:
    """Convert MatrixBasis to VectorizedMatrixBasis

    Returns
    -------
    VectorizedMatrixBasis
        VectorizedMatrixBasis converted from MatrixBasis
    """
    return VectorizedMatrixBasis(source)


def get_comp_basis() -> MatrixBasis:
    """Returns computational basis.

    Returns
    -------
    MatrixBasis
        computational basis ``[|0><0|, |0><1|, |1><0|, |1><1|]``
    """
    array00 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    array01 = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    array10 = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    array11 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    comp_basis = MatrixBasis([array00, array01, array10, array11])

    return comp_basis


def get_pauli_basis() -> MatrixBasis:
    """Returns Pauli basis.

    Returns
    -------
    MatrixBasis
        Pauli basis ``[I, X, Y, Z]``
    """
    identity = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    pauli_basis = MatrixBasis([identity, pauli_x, pauli_y, pauli_z])

    return pauli_basis


def get_normalized_pauli_basis() -> MatrixBasis:
    """Returns normalized Pauli basis.

    Returns
    -------
    MatrixBasis
        Pauli basis ``\\frac{1}{\\sqrt{2}}[I, X, Y, Z]``
    """
    identity = 1 / np.sqrt(2) * np.array([[1, 0], [0, 1]], dtype=np.complex128)
    pauli_x = 1 / np.sqrt(2) * np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = 1 / np.sqrt(2) * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli_z = 1 / np.sqrt(2) * np.array([[1, 0], [0, -1]], dtype=np.complex128)
    pauli_basis = MatrixBasis([identity, pauli_x, pauli_y, pauli_z])

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


def convert_vec(
    from_vec: np.array, from_basis: MatrixBasis, to_basis: MatrixBasis
) -> np.array:
    """converts vector representation from ``from_basis`` to ``to_basis``.

    Parameters
    ----------
    from_vec : np.array
        vector representation before converts vector representation
    from_basis : MatrixBasis
        basis before converts vector representation
    to_basis : MatrixBasis
        basis after converts vector representation

    Returns
    -------
    np.array
        vector representation after converts vector representation
        ``dtype`` is ``float64``
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
