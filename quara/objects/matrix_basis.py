import numpy as np
import quara.utils.matrix_util as mutil

from typing import List


class Basis:
    def __init__(self, basis: List[np.ndarray]):
        self._basis = basis

    @property
    def basis(self):  # read only
        return self._basis

    def __getitem__(self, index: int) -> np.ndarray:
        # return B_{\alpha}
        return np.copy(self._basis[index])

    def __len__(self):
        return len(self._basis)

    def __iter__(self):
        return iter(self._basis)

    def __str__(self):
        return str(self._basis)


class MatrixBasis(Basis):
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
        return VectorizedMatrixBasis(self)

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

    def size(self):
        # 行列サイズを返す
        return self[0].shape


class VectorizedMatrixBasis(Basis):
    def __init__(self, source: MatrixBasis):
        # 現状は、一旦MatrixBasisでくることだけが想定されている
        # もともとベクトル化されたnp.ndarrayがくることは想定されていない
        # TODO: sourceは、MatrixBasisがくるかすでにベクトル化されたnp.ndarrayがくるかで判断ですべきでは？
        self._org_basis: MatrixBasis = source
        self._basis: List[np.ndarray] = []

        # vectorize
        # これはutilに置いてもいいかもしれない
        for b in self._org_basis:
            b_vect = b.copy()
            self._basis.append(b_vect.flatten())

    @property
    def org_basis(self):  # read only
        return self._org_basis

    @property
    def dim(self):
        return self._org_basis.dim

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
    # computational basis
    array00 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    array01 = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    array10 = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    array11 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    comp_basis = MatrixBasis([array00, array01, array10, array11])

    return comp_basis


def get_pauli_basis() -> MatrixBasis:
    # Pauli basis
    identity = 1 / np.sqrt(2) * np.array([[1, 0], [0, 1]], dtype=np.complex128)
    pauli_x = 1 / np.sqrt(2) * np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = 1 / np.sqrt(2) * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli_z = 1 / np.sqrt(2) * np.array([[1, 0], [0, -1]], dtype=np.complex128)
    pauli_basis = MatrixBasis([identity, pauli_x, pauli_y, pauli_z])

    return pauli_basis
