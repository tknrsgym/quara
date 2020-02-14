from quara.objects import matrix_basis
from quara.objects.matrix_basis import MatrixBasis


class ElementalSystem:
    """個々の量子系を記述するためのクラス"""

    def __init__(self, basis: MatrixBasis):
        # system_idを持たせなくても同値ではなく同一インスタンスであるかどうかの判定はisで可能だが、
        # system_idで持たせておくとうっかり同値判定（==）で比較しても異なる値として判断されるので、
        # とりあえず持たせておく
        self._system_id: int = id(self)
        self._dim: int = basis.dim
        self._computational_basis: MatrixBasis = None  # TODO
        self._hemirtian_basis: MatrixBasis = basis

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


# TODO この関数は不要？
def get_with_normalized_pauli_basis() -> ElementalSystem:
    basis = matrix_basis.get_normalized_pauli_basis()
    system = ElementalSystem(basis)
    return system
