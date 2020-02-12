from typing import List

import numpy as np
import quara.utils.matrix_util as mutil


class VectorizedMatrixBasis:
    def __init__(self, source):
        self._org_basis: MatrixBasis = source
        # self._basis:

    def is_hermitian(self) -> bool:
        # 元の行列がエルミート行列になっているかどうかのチェック
        return self._org_basis.is_hermitian()


class MatrixBasis:
    def __init__(self, basis: List[np.ndarray]):
        self.basis = basis

    # def is_hermitian(self) -> bool:
    #     # エルミート行列になっているかどうかのチェック
    #     return mutil.is_hermitian(self.array)

    def to_vect(self) -> VectorizedMatrixBasis:
        # 自分自身をベクトル化したクラスを返す
        return VectorizedMatrixBasis(self)

    def __getitem__(self, index: int) -> np.ndarray:
        # 各B_{\alpha}を返す
        assert 0 <= index
        assert index <= len(self.basis)
        return np.copy(self.basis[index])

    def size(self):
        # 行列サイズを返す
        return self[0].shape

    def __len__(self):
        # 行列の個数を返す
        return len(self.basis)
