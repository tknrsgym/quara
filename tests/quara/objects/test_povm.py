import numpy as np
import pytest

import quara.objects.composite_system as csys
import quara.objects.elemental_system as esys
from quara.objects.matrix_basis import get_comp_basis, get_pauli_basis
from quara.objects.povm import Povm


class TestPovm:
    def test_validate_set_of_hermitian_matrices_ok(self):
        # Arrange
        # TODO: Is dtype complex or real?
        a1 = np.array([1, 0, 0, 1], dtype=np.complex128)
        a2 = np.array([0, 1, 1, 0], dtype=np.complex128)
        a3 = np.array([0, -1j, 1j, 0], dtype=np.complex128)
        a4 = np.array([1, 0, 0, -1], dtype=np.complex128)
        vecs = [a1, a2, a3, a4]

        e_sys = esys.ElementalSystem("q1", get_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)

        # Assert
        expected = [a1, a2, a3, a4]
        assert (povm[0] == expected[0]).all()
        assert (povm[1] == expected[1]).all()
        assert (povm[2] == expected[2]).all()
        assert (povm[3] == expected[3]).all()
        assert povm.composite_system is c_sys

    def test_validate_set_of_hermitian_matrices_ng(self):
        # Arrange
        # TODO: Is dtype complex or real?
        a1 = np.array([1, 0, 0, 0], dtype=np.complex128)
        a2 = np.array([0, 1, 0, 0], dtype=np.complex128)
        a3 = np.array([0, 0, 1, 0], dtype=np.complex128)
        a4 = np.array([0, 0, 0, 1], dtype=np.complex128)
        vecs = [a1, a2, a3, a4]

        e_sys = esys.ElementalSystem("q1", get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        with pytest.raises(ValueError):
            # ValueError: povm must be a set of Hermitian matrices
            _ = Povm(c_sys=c_sys, vecs=vecs)
