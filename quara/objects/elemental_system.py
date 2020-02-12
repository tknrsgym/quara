from quara.objects.matrix_basis import MatrixBasis


class ElementalSystem:
    def __init__(self):
        # system_idを持たせなくても同値ではなく同一インスタンスであるかどうかの判定はisで可能だが、
        # system_idで持たせておくとうっかり同値判定（==）で比較しても異なる値として判断されるので、
        # とりあえず持たせておく
        self._system_id: int = id(self)
        self.dimension: int = None  # TODO
        self._computational_basis: MatrixBasis = None  # TODO
        self._hemirtian_basis: MatrixBasis = None  # TODO

    @property
    def system_id(self):  # read only
        return self._system_id

    @property
    def computational_basis(self):  # read only?
        return self._computational_basis

    @property
    def hemirtian_basis_basis(self):  # read only?
        return self._hemirtian_basis
