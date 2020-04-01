import numpy as np
import pytest

import quara.objects.elemental_system as esys
from quara.objects.matrix_basis import get_comp_basis


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

    def test_access_hemirtian_basis(self):
        m_basis = get_comp_basis()
        e1 = esys.ElementalSystem(1, m_basis)

        assert id(m_basis) == id(e1.hemirtian_basis)

        # Test that "hemirtian_basis" cannot be updated
        with pytest.raises(AttributeError):
            e1.hemirtian_basis = 1

        with pytest.raises(AttributeError):
            e1.hemirtian_basis.basis = 1
