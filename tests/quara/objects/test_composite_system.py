import pytest

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_pauli_basis


class TestCompositeSystem:
    def test_eq(self):
        e1 = ElementalSystem("pauli_1", get_pauli_basis())
        e2 = ElementalSystem("pauli_2", get_pauli_basis())
        e3 = ElementalSystem("pauli_3", get_pauli_basis())

        c1 = CompositeSystem([e1, e2])
        c2 = CompositeSystem([e1, e2])
        assert (c1 == c2) is True

        c1 = CompositeSystem([e2, e1])
        c2 = CompositeSystem([e1, e2])
        assert (c1 == c2) is False

        c1 = CompositeSystem([e1, e2])
        c2 = CompositeSystem([e1, e2, e3])
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

    def test_duplicate_elemental_system(self):
        e1 = ElementalSystem("pauli_1", get_pauli_basis())

        with pytest.raises(ValueError):
            _ = CompositeSystem([e1, e1])

    def test_duplicate_elemental_system_name(self):
        e1 = ElementalSystem("pauli_1", get_pauli_basis())
        e2 = ElementalSystem("pauli_1", get_pauli_basis())

        with pytest.raises(ValueError):
            _ = CompositeSystem([e1, e2])


class TestCompositeSystemImmutable:
    def test_deney_update_elemental_systems_item(self):
        m_basis_1 = get_pauli_basis()
        m_basis_2 = get_pauli_basis()
        e1 = ElementalSystem("pauli_1", m_basis_1)
        e2 = ElementalSystem("pauli_2", m_basis_2)
        c_sys = CompositeSystem([e1, e2])

        m_basis_3 = get_pauli_basis()
        e3 = ElementalSystem("pauli_3", m_basis_3)

        with pytest.raises(TypeError):
            # TypeError: 'CompositeSystem' object does not support item assignment
            c_sys[0] = e3

        assert c_sys[0] is e1
        assert c_sys[1] is e2

        with pytest.raises(TypeError):
            # TypeError: 'tuple' object does not support item assignment
            c_sys.elemental_systems[0] = e3
        assert c_sys[0] is e1
        assert c_sys[1] is e2
