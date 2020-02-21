import numpy as np
import pytest

import quara.objects.elemental_system as esys
from quara.objects.matrix_basis import get_comp_basis


class TestElementalSystem:
    def test_access_name(self):
        m_basis = get_comp_basis()
        e1 = esys.ElementalSystem("e1", m_basis)

        assert e1.name == "e1"

        # Test that the name cannot be updated
        with pytest.raises(AttributeError):
            e1.name = "e2"
