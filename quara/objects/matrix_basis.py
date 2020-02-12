import numpy as np
import quara.utils.matrix_util as mutil


class VectorizedMatrixBasis:
    def __init__(self, array):
        self._org_basis: MatrixBasis = array  # TODO

    def is_hermitian(self) -> bool:
        # 元の行列がエルミート行列になっているかどうかのチェック
        return self._org_basis.is_hermitian()


class MatrixBasis:
    def __init__(self, array):
        self.array: np.ndarray = array

    def is_hermitian(self) -> bool:
        # エルミート行列になっているかどうかのチェック
        return mutil.is_hermitian(self.array)

    def to_vect(self) -> VectorizedMatrixBasis:
        # 自分自身をベクトル化したクラスを返す
        return VectorizedMatrixBasis(self)

    def size(self):
        # 行列サイズを返す
        return self.array.size()
