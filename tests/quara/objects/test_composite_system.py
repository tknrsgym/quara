import pytest

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_pauli_basis


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

        assert (c1 == "string") is False
        assert (c1 == None) is False
        assert (c1 == 1) is False

    def test_getitem(self):
        e1 = ElementalSystem("pauli_1", get_pauli_basis())
        e2 = ElementalSystem("pauli_2", get_pauli_basis())

        c1 = CompositeSystem([e1, e2])
        assert c1[0] is e1
        assert c1[1] is not e1

        # Test that element_system list cannot be updated
        with pytest.raises(TypeError):
            c1[0] = e2
