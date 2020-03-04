import copy
import itertools
from typing import List, Tuple, Union

import numpy as np

from quara.objects import elemental_system
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import MatrixBasis


class CompositeSystem:
    """合成系を記述するためのクラス"""

    # E1 \otimes E2のCompositeSystemがある場合には、E2 \otimes E1は実行できない

    def __init__(self, systems: List[ElementalSystem]):

        # Validation
        # Check for duplicate ElementalSystem
        names: List[str] = []
        e_sys_ids: List[int] = []

        for e_sys in systems:
            if e_sys.system_id in e_sys_ids:
                raise ValueError(
                    f"Duplicate ElementalSystem. \n system_id={e_sys.system_id}, name={e_sys.name}"
                )
            e_sys_ids.append(e_sys.system_id)

            if e_sys.name in names:
                raise ValueError(f"Duplicate ElementalSystem name. name={e_sys.name}")
            names.append(e_sys.name)

        # Sort by name of ElementalSystem
        ## Copy to avoid affecting the original source.
        ## ElementalSystem should remain the same instance as the original source
        ## to check if the instances are the same in the tensor product calculation.
        ## Therefore, use `copy.copy` instead of `copy.deepcopy`.
        sored_e_syses = copy.copy(systems)
        sored_e_syses.sort(key=lambda x: x.name)

        self._elemental_systems: Tuple[ElementalSystem, ...] = tuple(sored_e_syses)

        # Calculate tensor product of ElamentalSystem list for getting new MatrixBasis
        self._basis: MatrixBasis

        if len(self._elemental_systems) == 1:
            self._basis = self._elemental_systems[0].hemirtian_basis
        else:
            basis_list = [e_sys.hemirtian_basis for e_sys in self._elemental_systems]
            temp = basis_list[0]
            for elem in basis_list[1:]:
                temp = [
                    np.kron(val1, val2) for val1, val2 in itertools.product(temp, elem)
                ]
            self._basis = MatrixBasis(temp)
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

    @property
    def elemental_systems(self):  # read only
        return self._elemental_systems

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

    def __str__(self):
        desc = "elemental_systems:\n"
        for i, e_sys in enumerate(self._elemental_systems):
            desc += f"[{i}] {e_sys.name} (system_id={e_sys.system_id})\n"

        desc += "\n"
        desc += f"dim: {self._dim}\n"
        desc += f"basis:\n"
        desc += str(self._basis)
        return desc

    def __repr__(self):
        return f"{self.__class__.__name__}(systems={repr(self.elemental_systems)})"
