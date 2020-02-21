from quara.objects.povm import Povm
import quara.objects.elemental_system as esys
import quara.objects.composite_system as csys
from quara.objects.matrix_basis import get_comp_basis


class TestPovm:
    def test_access_composite_system(self):
        m_basis = get_comp_basis()
        e1 = esys.ElementalSystem("e1", m_basis)
        e2 = esys.ElementalSystem("e2", m_basis)
        c1 = csys.CompositeSystem([e1, e2])

