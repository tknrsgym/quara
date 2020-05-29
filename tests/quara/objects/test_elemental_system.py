import numpy as np
import pytest

import quara.objects.elemental_system as esys
from quara.objects.matrix_basis import (
    get_comp_basis,
    get_gell_mann_basis,
    get_pauli_basis,
)


class TestElementalSystem:
    def test_init_raise_name_is_not_int(self):
        # Arrange
        m_basis = get_comp_basis()

        # Act & Assert
        with pytest.raises(TypeError):
            # TypeError: Type of 'name' must be int.
            _ = esys.ElementalSystem("str is invalid type", m_basis)

    def test_access_name(self):
        m_basis = get_comp_basis()
        e1 = esys.ElementalSystem(1, m_basis)

        assert e1.name == 1

        # Test that "name" cannot be updated
        with pytest.raises(AttributeError):
            e1.name = 2  # New name

    def test_access_systemid(self):
        m_basis = get_comp_basis()
        e1 = esys.ElementalSystem(1, m_basis)
        actual = e1.system_id
        expected = id(e1)
        assert actual == expected

        # Test that "system_id" cannot be updated
        with pytest.raises(AttributeError):
            e1.system_id = "new_id"

    def test_access_dim(self):
        m_basis = get_comp_basis()
        e1 = esys.ElementalSystem(1, m_basis)
        actual = e1.dim
        expected = m_basis.dim
        assert actual == expected

        # Test that "dim" cannot be updated
        with pytest.raises(AttributeError):
            e1.dim = 100

    def test_access_comp_basis(self):
        # case: dim = 2
        m_basis = get_pauli_basis()
        e1 = esys.ElementalSystem(1, m_basis)
        actual = e1.comp_basis
        expected = get_comp_basis()
        assert len(actual) == 4
        assert np.all(actual[0] == expected[0])
        assert np.all(actual[1] == expected[1])
        assert np.all(actual[2] == expected[2])
        assert np.all(actual[3] == expected[3])

        # case: dim = 3
        m_basis = get_gell_mann_basis()
        e1 = esys.ElementalSystem(2, m_basis)
        actual = e1.comp_basis
        expected = get_comp_basis(3)
        assert len(actual) == 9
        assert np.all(actual[0] == expected[0])
        assert np.all(actual[1] == expected[1])
        assert np.all(actual[2] == expected[2])
        assert np.all(actual[3] == expected[3])
        assert np.all(actual[4] == expected[4])
        assert np.all(actual[5] == expected[5])
        assert np.all(actual[6] == expected[6])
        assert np.all(actual[7] == expected[7])
        assert np.all(actual[8] == expected[8])

    def test_access_hemirtian_basis(self):
        m_basis = get_comp_basis()
        e1 = esys.ElementalSystem(1, m_basis)

        assert id(m_basis) == id(e1.hemirtian_basis)

        # Test that "hemirtian_basis" cannot be updated
        with pytest.raises(AttributeError):
            e1.hemirtian_basis = 1

        with pytest.raises(AttributeError):
            e1.hemirtian_basis.basis = 1
