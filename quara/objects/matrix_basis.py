import numpy as np
import quara.utils.matrix_util as mutil

from typing import List
from typing import Tuple


def to_vect(source: "MatrixBasis") -> "VectorizedMatrixBasis":
    return VectorizedMatrixBasis(source)


class MatrixBasis:
    def __init__(self, basis: List[np.ndarray]):
        self._basis = basis
        self._dim = self[0].shape[0]
        assert self.is_squares()
        assert self.is_same_size()

    @property
    def dim(self):
        # dimを返す
        return self._dim

    def to_vect(self) -> "VectorizedMatrixBasis":
        # 自分自身をベクトル化したクラスを返す
        return to_vect(self)

    def is_squares(self) -> bool:
        # すべての行列が正方行列かどうかのチェック
        for mat in self:
            row, column = mat.shape
            if row != column:
                return False
        return True

    def is_same_size(self) -> bool:
        # すべての行列がサイズが一致しているかどうかのチェック
        for index in range(len(self) - 1):
            if self[index].shape != self[index + 1].shape:
                return False
        return True

    def is_basis(self) -> bool:
        # 基底になっているかどうかのチェック
        row_list = [mat.reshape(1, -1)[0] for mat in self]
        rank = np.linalg.matrix_rank(row_list)
        return rank >= self.dim ** 2

    def is_orthogonal(self) -> bool:
        # 直交性のチェック
        for index, left in enumerate(self[:-1]):
            for right in self[index + 1 :]:
                i_product = mutil.inner_product(left, right)
                if not np.isclose(i_product, 0):
                    return False
        return True

    def is_normal(self) -> bool:
        # 規格化されているかどうかのチェック
        for mat in self:
            i_product = mutil.inner_product(mat, mat)
            if not np.isclose(i_product, 1):
                return False
        return True

    def is_hermitian(self) -> bool:
        # エルミート行列になっているかどうかのチェック
        for mat in self:
            if not mutil.is_hermitian(mat):
                return False
        return True

    def is_scalar_mult_of_identity(self) -> bool:
        # 最初の要素(\alpha=0)が恒等行列の定数倍になっているかどうかのチェック
        mat = self[0]
        scalar = mat[0, 0]
        identity = np.identity(2, dtype=np.complex128)
        return np.allclose(scalar * identity, mat)

    def is_trace_less(self) -> bool:
        # 最初以外の要素(\alpha >= 1)がトレースレスになっているかどうかのチェック
        for index in range(1, len(self)):
            mat = self[index]
            tr = np.trace(mat)
            if tr != 0:
                return False
        return True

    def __getitem__(self, index: int) -> np.ndarray:
        """returns ``index``-th element of basis.
        
        Parameters
        ----------
        index : int
            index of the desired element of basis.
        
        Returns
        -------
        np.ndarray
            ``index``-th element of basis.
        """
        return np.copy(self._basis[index])

    def size(self) -> Tuple[int, int]:
        """returns shape(=size) of basis.
        
        Returns
        -------
        Tuple[int, int]
            shape(=size) of basis.
        """
        return self[0].shape

    def __len__(self) -> int:
        """returns number of basis.
        
        Returns
        -------
        int
            number of basis.
        """
        return len(self._basis)

    def __iter__(self):
        return iter(self._basis)

    def __str__(self) -> str:
        return str(self._basis)


class VectorizedMatrixBasis:
    def __init__(self, source: List[np.ndarray]):
        self._org_basis: MatrixBasis = source
        self._basis: List[np.ndarray] = []

        # vectorize
        # これはutilに置いてもいいかもしれない
        for b in self._org_basis:
            b_vect = b.copy()
            self._basis.append(b_vect.flatten())

    def __str__(self):
        return str(self._basis)

    @property
    def org_basis(self):  # read only
        return self._org_basis

    @property
    def basis(self):  # read only
        return self._basis

    def is_hermitian(self) -> bool:
        return self._org_basis.is_hermitian()

    def is_orthogonal(self) -> bool:
        return self._org_basis.is_orthogonal()

    def is_normal(self) -> bool:
        return self._org_basis.is_normal()

    def is_scalar_mult_of_identity(self) -> bool:
        return self._org_basis.is_scalar_mult_of_identity()

    def is_trace_less(self) -> bool:
        return self._org_basis.is_trace_less()


def get_comp_basis() -> MatrixBasis:
    """returns computational basis.
    
    Returns
    -------
    MatrixBasis
        computational basis ``[|00><00|, |01><01|, |10><10|, |11><11|]``
    """
    array00 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    array01 = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    array10 = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    array11 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    comp_basis = MatrixBasis([array00, array01, array10, array11])

    return comp_basis


def get_pauli_basis() -> MatrixBasis:
    """returns Pauli basis.
    
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
    """returns normalized Pauli basis.
    
    Returns
    -------
    MatrixBasis
        Pauli basis ``\frac{1}{\sqrt{2}}[I, X, Y, Z]``
    """
    identity = 1 / np.sqrt(2) * np.array([[1, 0], [0, 1]], dtype=np.complex128)
    pauli_x = 1 / np.sqrt(2) * np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = 1 / np.sqrt(2) * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli_z = 1 / np.sqrt(2) * np.array([[1, 0], [0, -1]], dtype=np.complex128)
    pauli_basis = MatrixBasis([identity, pauli_x, pauli_y, pauli_z])

    return pauli_basis

