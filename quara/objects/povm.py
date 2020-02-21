from typing import List, Union

import numpy as np

import quara.utils.matrix_util as mutil
from quara.objects.composite_system import CompositeSystem


class Povm:
    """
    Positive Operator-Valued Measure
    """

    def __init__(self):
        self.composite_system: CompositeSystem = None  # TODO
        self.w_list: list = None  # TODO: 取りうる測定値のリスト
        self._vec: List[np.ndarray] = []  # TODO: エルミート行列基底であることを要請する

    def __getitem__(self, key: int):
        return self._vec[key]

    def is_positive_semidefinite(self, atol: float = None) -> bool:
        # 各要素が半正定値か確認する

        # TODO: ここはもっとうまいやり方があるかもしれない
        if atol:
            for v in self._vec:
                if not mutil.is_positive_semidefinite(v, atol):
                    return False
        else:
            for v in self._vec:
                if not mutil.is_positive_semidefinite(v):
                    return False
        return True

    def is_identity(self):
        # 要素の総和が恒等行列になっているか確認する
        size = self._vec[0].shape
        sum_matrix = np.zeros(size)
        for v in self._vec:
            sum_matrix += v

        identity = np.identity(len(self._vec), dtype=np.complex128)
        return np.allclose(sum_matrix, identity)

    def eig(self, index: int = None) -> Union[List[np.ndarray], np.ndarray]:
        # 各要素の固有値を返す
        if index:
            target = self._vec[index]
            w = np.linalg.eigvals(target)
            return w
        else:
            w_list = []
            for target in self._vec:
                w = np.linalg.eigvals(target)
                w_list.append(w)
            return w_list
