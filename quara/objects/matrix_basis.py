from typing import List

import numpy as np
import quara.utils.matrix_util as mutil


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
        # 元の行列がエルミート行列になっているかどうかのチェック
        return self._org_basis.is_hermitian()


class MatrixBasis:
    def __init__(self, basis: List[np.ndarray]):
        self._basis = basis

    # def is_hermitian(self) -> bool:
    #     # エルミート行列になっているかどうかのチェック
    #     return mutil.is_hermitian(self.array)

    @property
    def basis(self):  # read only
        # 外から書き換え可能だとバグの温床になりそうなので
        # とりあえずread onlyにしておく
        return self._basis

    def to_vect(self) -> VectorizedMatrixBasis:
        # 自分自身をベクトル化したクラスを返す
        return VectorizedMatrixBasis(self)

    def __getitem__(self, index: int) -> np.ndarray:
        # 各B_{\alpha}を返す
        assert 0 <= index
        assert index <= len(self._basis)
        return self._basis[index]

    def size(self):
        # 行列サイズを返す
        return self[0].shape

    def __len__(self):
        # 行列の個数を返す
        return len(self._basis)

    def __iter__(self):
        return iter(self._basis)

    def __str__(self):
        return str(self._basis)
