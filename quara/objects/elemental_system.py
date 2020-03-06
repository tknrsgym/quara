from quara.objects import matrix_basis
from quara.objects.matrix_basis import MatrixBasis


class ElementalSystem:
    """個々の量子系を記述するためのクラス"""

    def __init__(self, name: int, basis: MatrixBasis):
        # Validate
        if type(name) != int:
            raise TypeError("Type of 'name' must be int.")

        # Set
        self._name: int = name
        # system_idを持たせなくても同値ではなく同一インスタンスであるかどうかの判定はisで可能だが、
        # system_idで持たせておくとうっかり同値判定（==）で比較しても異なる値として判断されるので、
        # とりあえず持たせておく
        self._system_id: int = id(self)
        self._dim: int = basis.dim
        # self._computational_basis: MatrixBasis = None  # TODO
        self._hemirtian_basis: MatrixBasis = basis

    @property
    def name(self):  # read only
        return self._name

    @property
    def system_id(self):  # read only
        return self._system_id

    @property
    def dim(self):  # read only
        return self._dim

    # @property
    # def computational_basis(self):  # read only
    #     return self._computational_basis

    @property
    def hemirtian_basis(self):  # read only
        return self._hemirtian_basis

    def __str__(self):
        desc = f"name: {self.name} \n"
        desc += f"system_id: {self.system_id} \n"
        desc += f"hemirtian_basis: {self._hemirtian_basis}"
        return desc

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)}, basis={repr(self._hemirtian_basis)}))"
