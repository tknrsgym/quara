from typing import List
from quara.objects import elemental_system
from quara.objects import operator
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import MatrixBasis


class CompositeSystem:
    """合成系を記述するためのクラス"""

    def __init__(self, systems: List[ElementalSystem]):
        self._elemental_systems: List[ElementalSystem] = systems
        # 合成後の基底をMatrixBasisの形で持っておく
        if len(self._elemental_systems) == 1:
            self._basis: MatrixBasis = self._elemental_systems[0].hemirtian_basis
        else:
            basis_list = [e_sys.hemirtian_basis for e_sys in self._elemental_systems]
            self._basis: MatrixBasis = operator.tensor_product(basis_list)
        self._dim: int = self._basis[0].shape[0]

    @property
    def basis(self):
        # TODO read onlyであるべき
        # CompositeSystemのbasisを返す
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

    def get_basis(self, index) -> MatrixBasis:
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


# TODO この関数は不要？
def get_with_normalized_pauli_basis() -> CompositeSystem:
    e_sys = elemental_system.get_with_normalized_pauli_basis()
    c_sys = CompositeSystem([e_sys])
    return c_sys


"""
e_sys1 = elemental_system.get_with_normalized_pauli_basis()
e_sys2 = elemental_system.get_with_normalized_pauli_basis()
c_sys = CompositeSystem([e_sys1, e_sys2])
print(len(c_sys.basis))
# print(c_sys.basis)
print(c_sys.get_basis(4))
print(c_sys.get_basis((1, 0)))
"""
