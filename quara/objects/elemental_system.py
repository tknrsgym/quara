from typing import Dict

from quara.objects import matrix_basis
from quara.objects.matrix_basis import MatrixBasis


class ElementalSystem:
    """個々の量子系を記述するためのクラス"""

    __esys_names: Dict[str, "ElementalSystem"] = {}

    @classmethod
    def esys_names(cls):
        return list(cls.__esys_names.keys())

    def __init__(self, name: str, basis: MatrixBasis):
        self._name: str = name
        self.__class__._regist_esys_name(self)
        # system_idを持たせなくても同値ではなく同一インスタンスであるかどうかの判定はisで可能だが、
        # system_idで持たせておくとうっかり同値判定（==）で比較しても異なる値として判断されるので、
        # とりあえず持たせておく
        self._system_id: int = id(self)
        self._dim: int = basis.dim
        # self._computational_basis: MatrixBasis = None  # TODO
        self._hemirtian_basis: MatrixBasis = basis

    @classmethod
    def _regist_esys_name(cls, new_elemental_system: "ElementalSystem"):
        name = new_elemental_system.name
        if name in cls.__esys_names:
            raise ValueError(
                f"An ElementalSystem with the name '{name}' already exists."
            )

        cls.__esys_names[name] = new_elemental_system

    @property
    def name(self):  # read only
        return self._name

    @property
    def system_id(self):  # read only
        return self._system_id

    @property
    def dim(self):
        # dimを返す
        return self._dim

    @property
    def computational_basis(self):  # read only
        return self._computational_basis

    @property
    def hemirtian_basis(self):  # read only?
        return self._hemirtian_basis

    def __eq__(self, other):
        if not isinstance(other, ElementalSystem):
            return False
        return self.name == other.name


# TODO この関数は不要？
def get_with_normalized_pauli_basis(name: str) -> ElementalSystem:
    basis = matrix_basis.get_normalized_pauli_basis()
    system = ElementalSystem(name, basis)
    return system
