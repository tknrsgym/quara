from typing import List
from quara.objects import elemental_system
from quara.objects.elemental_system import ElementalSystem


class CompositeSystem:
    def __init__(self, systems: List[ElementalSystem]):
        self._elemental_systems = systems
        # TODO 合成後の基底をMatrixBasisの形で持っておくこと
        if len(self._elemental_systems) == 1:
            self._basis = self._elemental_systems[0].hemirtian_basis

    @property
    def basis(self):
        # CompositeSystemのbasisを返す
        return self._basis


def get_with_normalized_pauli_basis() -> CompositeSystem:
    elem_system = elemental_system.get_with_normalized_pauli_basis()
    system = CompositeSystem([elem_system])
    return system

