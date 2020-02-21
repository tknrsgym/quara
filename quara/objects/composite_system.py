import itertools
from typing import List, Union, Tuple

from quara.objects import elemental_system
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import MatrixBasis

import numpy as np


class CompositeSystem:
    """合成系を記述するためのクラス"""

    # E1 \otimes E2のCompositeSystemがある場合には、E2 \otimes E1は実行できない

    def __init__(self, systems: List[ElementalSystem]):
        self._elemental_systems: List[ElementalSystem] = systems
        # calculate tensor product of ElamentalSystem list for getting new MatrixBasis
        if len(self._elemental_systems) == 1:
            self._basis: MatrixBasis = self._elemental_systems[0].hemirtian_basis
        else:
            basis_list = [e_sys.hemirtian_basis for e_sys in self._elemental_systems]
            temp = basis_list[0]
            for elem in basis_list[1:]:
                temp = [
                    np.kron(val1, val2) for val1, val2 in itertools.product(temp, elem)
                ]
            self._basis: MatrixBasis = MatrixBasis(temp)
        self._dim: int = self._basis[0].shape[0]

    @property
    def basis(self):
        return self._basis

    @property
    def dim(self):
        """returns dim of CompositeSystem.

        Returns
        -------
        int
            dim of CompositeSystem
        """
        return self._dim

    def get_basis(self, index: Union[int, tuple]) -> MatrixBasis:
        # 基底のテンソル積を返す
        # TODO read onlyであるべき
        if type(index) == tuple:
            # tupleのサイズは_elemental_systemsのリスト長と一致していること
            assert len(index) == len(self._elemental_systems)

            # _basisでの位置を計算(tupleの中を後ろから走査)
            # ElementalSystemのリスト長=3で、次元をそれぞれdim1,dim2,dim3とする
            # このとき、tuple(x1, x2, x3)が返す_basisのインデックスは、次の式で表される
            # x1 * (dim1 ** 2) * (dim2 ** 2) * (dim3 ** 2) + x2 * (dim2 ** 2) * (dim3 ** 2) + x3 * (dim3 ** 2)
            temp_grobal_index = 0
            temp_dim = 1
            for e_sys_position, local_index in enumerate(reversed(index)):
                temp_grobal_index += local_index * temp_dim
                temp_dim = temp_dim * (self._elemental_systems[e_sys_position].dim ** 2)
            return self._basis[temp_grobal_index]
        else:
            return self._basis[index]

    def __len__(self) -> int:
        return len(self._elemental_systems)

    def __getitem__(self, key: int) -> ElementalSystem:
        return self._elemental_systems[key]

    def __iter__(self):
        return iter(self._elemental_systems)

    def __eq__(self, other) -> bool:
        if not isinstance(other, CompositeSystem):
            return False

        if len(self) != len(other):
            return False

        for s, o in zip(self, other):
            if s is not o:
                return False
        return True
