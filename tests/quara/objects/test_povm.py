import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

import quara.objects.composite_system as csys
import quara.objects.elemental_system as esys
from quara.objects.matrix_basis import (
    get_comp_basis,
    get_normalized_pauli_basis,
    get_pauli_basis,
)
from quara.objects.povm import Povm
from quara.protocol import simple_io as s_io


class TestPovm:
    def test_validate_set_of_hermitian_matrices_ok(self):
        # Arrange
        p1 = np.array(
            [0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j], dtype=np.complex128
        )
        p2 = np.array(
            [0.5 + 0.0j, -0.5 + 0.0j, -0.5 + 0.0j, 0.5 + 0.0j], dtype=np.complex128
        )
        vecs = [p1, p2]

        e_sys = esys.ElementalSystem(1, get_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)

        # Assert
        expected = [p1, p2]
        assert (povm[0] == expected[0]).all()
        assert (povm[1] == expected[1]).all()
        assert povm.composite_system is c_sys

    def test_validate_set_of_hermitian_matrices_ng(self):
        # Arrange
        p1 = np.array([1, 0, 0, 0], dtype=np.complex128)
        p2 = np.array([0, 1, 0, 0], dtype=np.complex128)
        vecs = [p1, p2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        with pytest.raises(ValueError):
            # ValueError: povm must be a set of Hermitian matrices
            _ = Povm(c_sys=c_sys, vecs=vecs)

    def test_validate_sum_is_identity_ok(self):
        # Arrange
        p1 = np.array(
            [0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j], dtype=np.complex128
        )
        p2 = np.array(
            [0.5 + 0.0j, -0.5 + 0.0j, -0.5 + 0.0j, 0.5 + 0.0j], dtype=np.complex128
        )
        vecs = [p1, p2]

        e_sys = esys.ElementalSystem(1, get_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)
        actual = povm._is_identity()

        # Assert
        assert actual is True

    def test_validate_sum_is_identity_ng(self):
        # Arrange
        p1 = np.array(
            [0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j], dtype=np.complex128
        )
        p2 = np.array(
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128
        )
        vecs = [p1, p2]

        e_sys = esys.ElementalSystem(1, get_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        with pytest.raises(ValueError):
            # ValueError: The sum of the elements of POVM must be an identity matrix.
            _ = Povm(c_sys=c_sys, vecs=vecs)

    def test_validate_is_positive_semidefinite_ok(self):
        # Arrange
        ps_1 = np.array([1, 0, 0, 0], dtype=np.complex128)
        ps_2 = np.array([0, 0, 0, 1], dtype=np.complex128)
        vecs = [ps_1, ps_2]

        e_sys = esys.ElementalSystem(1, get_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)
        actual = povm._is_positive_semidefinite()

        # Assert
        assert actual is True

    def test_validate_is_positive_semidefinite_ng(self):
        # Arrange
        ps = np.array([1, 0, 0, 0], dtype=np.complex128)
        not_ps = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        vecs = [ps, not_ps]

        e_sys = esys.ElementalSystem(1, get_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        with pytest.raises(ValueError):
            _ = Povm(c_sys=c_sys, vecs=vecs)

    def test_calc_eigenvalues_all(self):
        # Arrange
        vec_1 = np.array([1, 0, 0, 0], dtype=np.complex128)
        vec_2 = np.array([0, 0, 0, 1], dtype=np.complex128)
        vecs = [vec_1, vec_2]

        e_sys = esys.ElementalSystem(1, get_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)
        actual = povm.calc_eigenvalues()

        # Assert
        expected = [
            np.array([1, 0], dtype=np.complex128),
            np.array([0, 1], dtype=np.complex128),
        ]

        assert len(actual) == len(expected)
        npt.assert_almost_equal(actual[0], expected[0], decimal=15)
        npt.assert_almost_equal(actual[1], expected[1], decimal=15)

    def test_calc_eigenvalues_one(self):
        # Arrange
        vec_1 = np.array([1, 0, 0, 0], dtype=np.complex128)
        vec_2 = np.array([0, 0, 0, 1], dtype=np.complex128)
        vecs = [vec_1, vec_2]

        e_sys = esys.ElementalSystem(1, get_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)
        actual = povm.calc_eigenvalues(0)

        # Assert
        expected = np.array([1, 0], dtype=np.complex128)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)
        actual = povm.calc_eigenvalues(1)

        # Assert
        expected = np.array([0, 1], dtype=np.complex128)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_validate_dim_ng(self):
        # Arrange
        test_root_dir = Path(os.path.dirname(__file__)).parent.parent
        data_dir = test_root_dir / "data"

        dim = 2 ** 2  # 2 qubits
        num_state = 16
        num_povm = 9
        num_outcome = 4

        povms = s_io.load_povm_list(
            data_dir / "tester_2qubit_povm.csv",
            dim=dim,
            num_povm=num_povm,
            num_outcome=num_outcome,
        )
        vecs = list(povms[0])  # 2qubit

        e_sys = esys.ElementalSystem(1, get_pauli_basis())  # 1qubit
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        with pytest.raises(ValueError):
            _ = Povm(c_sys=c_sys, vecs=vecs)

    def test_convert_basis(self):
        # Arrange
        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])
        ps_1 = np.array([1, 0, 0, 0], dtype=np.complex128)
        ps_2 = np.array([0, 0, 0, 1], dtype=np.complex128)
        vecs = [ps_1, ps_2]
        povm = Povm(c_sys=c_sys, vecs=vecs)
        to_basis = get_normalized_pauli_basis()

        # Act
        actual = povm.convert_basis(to_basis)

        # Assert
        expected = [
            1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.complex128),
            1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.complex128),
        ]
        assert len(actual) == len(expected)
        for i, a in enumerate(actual):
            assert np.all(a == expected[i])
