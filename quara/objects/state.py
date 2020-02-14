from typing import List

import numpy as np

from quara.objects.composite_system import CompositeSystem
import quara.utils.matrix_util as mutil


class State:
    def __init__(self, c_sys: CompositeSystem, vec: np.ndarray):
        self._composite_system: CompositeSystem = c_sys
        self._vec: np.ndarray = vec

        size = self._vec.shape
        # 1次元配列(=ベクトル)なのかチェック
        assert len(size) == 1
        # サイズが平方数なのかチェック
        self._dim = int(np.sqrt(size[0]))
        assert self._dim ** 2 == size[0]
        # TODO 実数であることのチェック(dtypeを見る？)
        # TODO CompositeSystemの次元と、vecの次元が一致していること

    @property
    def dim(self):
        # dimを返す
        return self._dim

    def get_density_matrix(self) -> np.ndarray:
        # 密度行列を返す
        density = np.zeros((self._dim, self._dim), dtype=np.complex128)
        for entry, basis in zip(self._vec, self._composite_system.basis):
            density += entry * basis
        return density

    def is_trace_one(self) -> bool:
        # トレースが1になっているか確認する
        tr = np.trace(self.get_density_matrix())
        return tr == 1

    def is_hermitian(self) -> bool:
        # エルミート行列になっているかどうかのチェック
        return mutil.is_hermitian(self.get_density_matrix())

    def is_positive_semidefinite(self) -> bool:
        # 半正定値行列になっているかどうかのチェック
        return mutil.is_positive_semidefinite(self.get_density_matrix())

    def get_eigen_values(self) -> List:
        # 固有値を返す(順不同)
        # see: https://numpy.org/doc/1.18/reference/generated/numpy.linalg.eigvals.html
        return np.linalg.eigvals(self.get_density_matrix())
