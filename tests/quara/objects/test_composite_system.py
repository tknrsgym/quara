from quara.objects.matrix_basis import get_pauli_basis
from quara.objects.elemental_system import ElementalSystem
from quara.objects.composite_system import CompositeSystem


class TestCompositeSystem:
    def test_eq(self):
        e1 = ElementalSystem("pauli", get_pauli_basis())
        e2 = ElementalSystem("pauli", get_pauli_basis())

        c1 = CompositeSystem([e1, e2])
        c2 = CompositeSystem([e1, e2])
        assert (c1 == c2) is True

        c1 = CompositeSystem([e2, e1])
        c2 = CompositeSystem([e1, e2])
        assert (c1 == c2) is False

        c1 = CompositeSystem([e1, e1])
        c2 = CompositeSystem([e1, e1, e1])
        assert (c1 == c2) is False

        assert (c1 == c1) is True
