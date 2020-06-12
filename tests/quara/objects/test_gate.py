import numpy as np
import numpy.testing as npt
import pytest

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import (
    Gate,
    convert_var_index_to_gate_index,
    convert_gate_index_to_var_index,
    convert_var_to_gate,
    convert_gate_to_var,
    calc_gradient_from_gate,
    calc_agf,
    convert_hs,
    get_cnot,
    get_cz,
    get_h,
    get_i,
    get_root_x,
    get_root_y,
    get_s,
    get_sdg,
    get_swap,
    get_t,
    get_x,
    get_y,
    get_z,
    is_hp,
)
from quara.objects.operators import composite, tensor_product
from quara.objects.state import get_y0_1q, get_y1_1q, get_z0_1q, get_z1_1q
from quara.settings import Settings


class TestGate:
    def test_init_error(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # Test that HS must be square matrix
        hs = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.float64,
        )
        with pytest.raises(ValueError):
            Gate(c_sys, hs)

        # Test that dim of HS must be square number
        hs = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
        with pytest.raises(ValueError):
            Gate(c_sys, hs)

        # Test that HS must be real matrix
        hs = np.array(
            [[1j, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.complex128,
        )
        with pytest.raises(ValueError):
            Gate(c_sys, hs)

        # Test that dim of HS equals dim of CompositeSystem
        hs = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        with pytest.raises(ValueError):
            Gate(c_sys, hs)

    def test_init_is_physical(self):
        e_sys = ElementalSystem(1, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])

        # gate is not TP
        hs_not_tp = np.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        with pytest.raises(ValueError):
            Gate(c_sys, hs_not_tp)
        with pytest.raises(ValueError):
            Gate(c_sys, hs_not_tp, is_physical=True)

        # gate is not CP
        hs_not_cp = np.array(
            [[-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        with pytest.raises(ValueError):
            Gate(c_sys, hs_not_cp)
        with pytest.raises(ValueError):
            Gate(c_sys, hs_not_cp, is_physical=True)

        # case: when is_physical is False, it is not happened ValueError
        Gate(c_sys, hs_not_tp, is_physical=False)
        Gate(c_sys, hs_not_cp, is_physical=False)

    def test_access_dim(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        hs = np.array(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        gate = Gate(c_sys, hs)

        actual = gate.dim
        expected = int(np.sqrt(hs.shape[0]))
        assert actual == expected

        # Test that "dim" cannot be updated
        with pytest.raises(AttributeError):
            gate.dim = 100

    def test_access_hs(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        hs = np.array(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        gate = Gate(c_sys, hs)

        actual = gate.hs
        expected = hs
        assert np.all(actual == expected)

        # Test that "hs" cannot be updated
        with pytest.raises(AttributeError):
            gate.hs = hs

    def test_access_is_physical(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        hs = np.array(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        gate = Gate(c_sys, hs)
        assert gate.is_physical == True

        # Test that "is_physical" cannot be updated
        with pytest.raises(AttributeError):
            gate.is_physical = False

    def test_get_basis(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        z = get_z(c_sys)
        actual = z.get_basis().basis
        expected = matrix_basis.get_normalized_pauli_basis().basis

        assert len(actual) == 4
        assert np.all(actual[0] == expected[0])
        assert np.all(actual[1] == expected[1])
        assert np.all(actual[2] == expected[2])
        assert np.all(actual[3] == expected[3])

    def test_is_tp(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # case: TP
        z = get_z(c_sys)
        assert z.is_tp() == True

        # case: not TP
        hs = np.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        gate = Gate(c_sys, hs, is_physical=False)
        assert gate.is_tp() == False

    def test_is_cp(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # case: CP
        x = get_x(c_sys)
        assert x.is_cp() == True
        y = get_y(c_sys)
        assert y.is_cp() == True
        z = get_z(c_sys)
        assert z.is_cp() == True

        # case: not CP
        hs = np.array(
            [[-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        gate = Gate(c_sys, hs, is_physical=False)
        assert gate.is_cp() == False

    def test_convert_basis(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        i = get_i(c_sys)
        x = get_x(c_sys)
        y = get_y(c_sys)
        z = get_z(c_sys)

        # for I
        actual = i.convert_basis(matrix_basis.get_comp_basis())
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for X
        actual = x.convert_basis(matrix_basis.get_comp_basis())
        expected = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for Y
        actual = y.convert_basis(matrix_basis.get_comp_basis())
        expected = np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for Z
        actual = z.convert_basis(matrix_basis.get_comp_basis())
        expected = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_convert_to_comp_basis(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        i = get_i(c_sys)
        x = get_x(c_sys)
        y = get_y(c_sys)
        z = get_z(c_sys)

        # for I
        actual = i.convert_to_comp_basis()
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for X
        actual = x.convert_to_comp_basis()
        expected = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for Y
        actual = y.convert_to_comp_basis()
        expected = np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for Z
        actual = z.convert_to_comp_basis()
        expected = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_to_choi_matrix(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # for I
        actual = get_i(c_sys).to_choi_matrix()
        expected = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for X
        actual = get_x(c_sys).to_choi_matrix()
        expected = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for Y
        actual = get_y(c_sys).to_choi_matrix()
        expected = np.array([[0, 0, 0, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for Z
        actual = get_z(c_sys).to_choi_matrix()
        expected = np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for H
        actual = get_h(c_sys).to_choi_matrix()
        expected = (
            1
            / 2
            * np.array([[1, 1, 1, -1], [1, 1, 1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]])
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    @classmethod
    def calc_sum_of_kraus(cls, kraus):
        # calc \sum_{\alpha} K__{\alpha} K_{\alpha}^{\dagger}
        sum = np.zeros(kraus[0].shape, dtype=np.complex128)
        for matrix in kraus:
            sum += matrix @ matrix.conj().T
        return sum

    def test_to_kraus_matrices(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        eye2 = np.eye(2, dtype=np.complex128)

        # for I
        actual = get_i(c_sys).to_kraus_matrices()
        expected = [np.array([[1, 0], [0, 1]], dtype=np.complex128)]
        assert len(actual) == 1
        npt.assert_almost_equal(actual[0], expected[0], decimal=15)
        npt.assert_almost_equal(TestGate.calc_sum_of_kraus(actual), eye2, decimal=14)

        # for X
        actual = get_x(c_sys).to_kraus_matrices()
        expected = [np.array([[0, 1], [1, 0]], dtype=np.complex128)]
        assert len(actual) == 1
        npt.assert_almost_equal(actual[0], expected[0], decimal=15)
        npt.assert_almost_equal(TestGate.calc_sum_of_kraus(actual), eye2, decimal=14)

        # for Y
        actual = get_y(c_sys).to_kraus_matrices()
        expected = [np.array([[0, 1], [-1, 0]], dtype=np.complex128)]
        assert len(actual) == 1
        npt.assert_almost_equal(actual[0], expected[0], decimal=15)
        npt.assert_almost_equal(TestGate.calc_sum_of_kraus(actual), eye2, decimal=14)

        # for Z
        actual = get_z(c_sys).to_kraus_matrices()
        expected = [np.array([[-1, 0], [0, 1]], dtype=np.complex128)]
        assert len(actual) == 1
        npt.assert_almost_equal(actual[0], expected[0], decimal=15)
        npt.assert_almost_equal(TestGate.calc_sum_of_kraus(actual), eye2, decimal=14)

        # for H
        actual = get_h(c_sys).to_kraus_matrices()
        expected = [1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)]
        assert len(actual) == 1
        npt.assert_almost_equal(actual[0], expected[0], decimal=15)
        npt.assert_almost_equal(TestGate.calc_sum_of_kraus(actual), eye2, decimal=14)

        # Kraus matrix does not exist(HS is not CP)
        hs = np.array(
            [[-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        actual = Gate(c_sys, hs, is_physical=False).to_kraus_matrices()
        assert len(actual) == 0

    def test_is_hp(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # case: HP
        x = get_x(c_sys)
        assert is_hp(x.hs, x.get_basis()) == True
        y = get_y(c_sys)
        assert is_hp(y.hs, y.get_basis()) == True
        z = get_z(c_sys)
        assert is_hp(z.hs, z.get_basis()) == True

        # case: not HP
        hs = np.array([[1j, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        assert is_hp(hs, matrix_basis.get_comp_basis()) == False

    def test_to_process_matrix(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        i = get_i(c_sys)
        x = get_x(c_sys)
        y = get_y(c_sys)
        z = get_z(c_sys)

        # for I
        actual = i.to_process_matrix()
        expected = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for X
        actual = x.to_process_matrix()
        expected = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for Y
        actual = y.to_process_matrix()
        expected = np.array([[0, 0, 0, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for Z
        actual = z.to_process_matrix()
        expected = np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]])
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_to_var(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hs = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
        )

        # Case 1: default
        # Act
        gate = Gate(c_sys=c_sys, hs=hs)
        actual = gate.to_var()

        # Assert
        expected = np.array([0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 2:
        # Arrange
        gate = Gate(c_sys=c_sys, hs=hs, on_para_eq_constraint=True)

        # Act
        actual = gate.to_var()

        # Assert
        expected = np.array([0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 3:
        # Arrange
        gate = Gate(c_sys=c_sys, hs=hs, on_para_eq_constraint=False)

        # Act
        actual = gate.to_var()

        # Assert
        expected = np.array(
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64
        )
        npt.assert_almost_equal(actual, expected, decimal=15)


def test_convert_var_index_to_gate_index():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # default
    actual = convert_var_index_to_gate_index(c_sys, 11)
    assert actual == (3, 3)

    # on_eq_constraint=True
    actual = convert_var_index_to_gate_index(c_sys, 11, on_eq_constraint=True)
    assert actual == (3, 3)

    # on_eq_constraint=False
    actual = convert_var_index_to_gate_index(c_sys, 15, on_eq_constraint=False)
    assert actual == (3, 3)


def test_convert_gate_index_to_var_index():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # default
    actual = convert_gate_index_to_var_index(c_sys, (3, 3))
    assert actual == 11

    # on_eq_constraint=True
    actual = convert_gate_index_to_var_index(c_sys, (3, 3), on_eq_constraint=True)
    assert actual == 11

    # on_eq_constraint=False
    actual = convert_gate_index_to_var_index(c_sys, (3, 3), on_eq_constraint=False)
    assert actual == 15


def test_convert_var_to_gate():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # default
    hs = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64)
    actual = convert_var_to_gate(c_sys, hs)
    expected = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # on_eq_constraint=True
    hs = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64)
    actual = convert_var_to_gate(c_sys, hs, on_eq_constraint=True)
    expected = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # on_eq_constraint=False
    hs = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    actual = convert_var_to_gate(c_sys, hs, on_eq_constraint=False)
    npt.assert_almost_equal(actual.hs, hs, decimal=15)


def test_convert_gate_to_var():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # default
    hs = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    actual = convert_gate_to_var(c_sys, hs)
    expected = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64)
    npt.assert_almost_equal(actual, expected.flatten(), decimal=15)

    # on_eq_constraint=True
    hs = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    actual = convert_gate_to_var(c_sys, hs, on_eq_constraint=True)
    expected = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64)
    npt.assert_almost_equal(actual, expected.flatten(), decimal=15)

    # on_eq_constraint=False
    hs = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    actual = convert_gate_to_var(c_sys, hs, on_eq_constraint=False)
    npt.assert_almost_equal(actual, hs.flatten(), decimal=15)


def test_calc_gradient_from_gate():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # default
    hs = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    actual = calc_gradient_from_gate(c_sys, hs, 1)
    expected = np.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
    )
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # on_eq_constraint=True
    hs = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    actual = calc_gradient_from_gate(c_sys, hs, 1, on_eq_constraint=True)
    expected = np.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
    )
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # on_eq_constraint=False
    hs = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    actual = calc_gradient_from_gate(c_sys, hs, 1, on_eq_constraint=False)
    expected = np.array(
        [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
    )
    npt.assert_almost_equal(actual.hs, expected, decimal=15)


def test_calc_agf():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    i = get_i(c_sys)
    x = get_x(c_sys)
    z = get_z(c_sys)

    # case: g=u
    actual = calc_agf(i, i)
    expected = 1
    assert np.isclose(actual, expected, atol=Settings.get_atol())

    # case: g is not u
    actual = calc_agf(z, x)
    expected = 1.0 / 3.0
    assert np.isclose(actual, expected, atol=Settings.get_atol())

    # case: u is not Hermitian
    hs = np.array(
        [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    gate = Gate(c_sys, hs, is_physical=False)
    with pytest.raises(ValueError):
        calc_agf(z.hs, gate)


def test_convert_hs():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    i = get_i(c_sys)
    x = get_x(c_sys)
    y = get_y(c_sys)
    z = get_z(c_sys)

    # for I
    actual = convert_hs(i.hs, i.get_basis(), matrix_basis.get_comp_basis())
    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    npt.assert_almost_equal(actual, expected, decimal=15)

    # for X
    actual = convert_hs(x.hs, x.get_basis(), matrix_basis.get_comp_basis())
    expected = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    npt.assert_almost_equal(actual, expected, decimal=15)

    # for Y
    actual = convert_hs(y.hs, y.get_basis(), matrix_basis.get_comp_basis())
    expected = np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0]])
    npt.assert_almost_equal(actual, expected, decimal=15)

    # for Z
    actual = convert_hs(z.hs, z.get_basis(), matrix_basis.get_comp_basis())
    expected = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_get_i():
    # case: dim = 2
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    i = get_i(c_sys)
    x = get_x(c_sys)
    y = get_y(c_sys)
    z = get_z(c_sys)
    npt.assert_almost_equal(x.hs @ i.hs, x.hs, decimal=15)
    npt.assert_almost_equal(i.hs @ x.hs, x.hs, decimal=15)
    npt.assert_almost_equal(y.hs @ i.hs, y.hs, decimal=15)
    npt.assert_almost_equal(i.hs @ y.hs, y.hs, decimal=15)
    npt.assert_almost_equal(z.hs @ i.hs, z.hs, decimal=15)
    npt.assert_almost_equal(i.hs @ z.hs, z.hs, decimal=15)

    # case: dim = 3
    e_sys = ElementalSystem(1, matrix_basis.get_gell_mann_basis())
    c_sys = CompositeSystem([e_sys])
    actual = get_i(c_sys)
    expected = np.eye(9, dtype=np.float64)
    assert np.all(actual.hs == expected)


def test_get_x():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    i = get_i(c_sys)
    x = get_x(c_sys)
    y = get_y(c_sys)
    z = get_z(c_sys)
    npt.assert_almost_equal(x.hs @ x.hs, i.hs, decimal=15)
    npt.assert_almost_equal(x.hs @ y.hs, z.hs, decimal=15)

    # Test that not 1qubit ElementalSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        get_x(c_sys)


def test_get_y():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    i = get_i(c_sys)
    x = get_x(c_sys)
    y = get_y(c_sys)
    z = get_z(c_sys)
    npt.assert_almost_equal(y.hs @ y.hs, i.hs, decimal=15)
    npt.assert_almost_equal(y.hs @ z.hs, x.hs, decimal=15)

    # Test that not 1qubit ElementalSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        get_y(c_sys)


def test_get_z():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    i = get_i(c_sys)
    x = get_x(c_sys)
    y = get_y(c_sys)
    z = get_z(c_sys)
    npt.assert_almost_equal(z.hs @ z.hs, i.hs, decimal=15)
    npt.assert_almost_equal(z.hs @ x.hs, y.hs, decimal=15)

    # Test that not 1qubit ElementalSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        get_z(c_sys)


def test_get_root_x():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    x = get_x(c_sys)
    rootx = get_root_x(c_sys)
    npt.assert_almost_equal(rootx.hs @ rootx.hs, x.hs, decimal=15)

    # Test that not 1qubit ElementalSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        get_root_x(c_sys)


def test_get_root_y():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    y = get_y(c_sys)
    rooty = get_root_y(c_sys)
    npt.assert_almost_equal(rooty.hs @ rooty.hs, y.hs, decimal=10)

    # Test that not 1qubit ElementalSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        get_root_y(c_sys)


def test_get_s():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    z = get_z(c_sys)
    s = get_s(c_sys)
    npt.assert_almost_equal(s.hs @ s.hs, z.hs, decimal=15)

    # Test that not 1qubit ElementalSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        get_s(c_sys)


def test_get_sdg():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    s = get_s(c_sys)
    sdg = get_sdg(c_sys)
    i = get_i(c_sys)
    npt.assert_almost_equal(s.hs @ sdg.hs, i.hs, decimal=15)

    # Test that not 1qubit ElementalSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        get_sdg(c_sys)


def test_get_t():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    s = get_s(c_sys)
    t = get_t(c_sys)
    npt.assert_almost_equal(t.hs @ t.hs, s.hs, decimal=15)

    # Test that not 1qubit ElementalSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        get_t(c_sys)


def test_get_cnot():
    # prepare gate
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys0 = CompositeSystem([e_sys0])
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys1 = CompositeSystem([e_sys1])

    c_sys01 = CompositeSystem([e_sys0, e_sys1])

    # prepare states
    z0_c_sys0 = get_z0_1q(c_sys0)
    z1_c_sys0 = get_z1_1q(c_sys0)
    z0_c_sys1 = get_z0_1q(c_sys1)
    z1_c_sys1 = get_z1_1q(c_sys1)
    z0_z0 = tensor_product(z0_c_sys0, z0_c_sys1)
    z0_z1 = tensor_product(z0_c_sys0, z1_c_sys1)
    z1_z0 = tensor_product(z1_c_sys0, z0_c_sys1)
    z1_z1 = tensor_product(z1_c_sys0, z1_c_sys1)

    ### gete: control bit is 1st qubit
    gate = get_cnot(c_sys01, e_sys0)

    # |00> -> |00>
    state = composite(gate, z0_z0)
    npt.assert_almost_equal(
        state.to_density_matrix(), z0_z0.to_density_matrix(), decimal=15
    )

    # |01> -> |01>
    state = composite(gate, z0_z1)
    npt.assert_almost_equal(
        state.to_density_matrix(), z0_z1.to_density_matrix(), decimal=15
    )

    # |10> -> |11>
    state = composite(gate, z1_z0)
    npt.assert_almost_equal(
        state.to_density_matrix(), z1_z1.to_density_matrix(), decimal=15
    )

    # |11> -> |10>
    state = composite(gate, z1_z1)
    npt.assert_almost_equal(
        state.to_density_matrix(), z1_z0.to_density_matrix(), decimal=15
    )

    ### gete: control bit is 2st qubit
    gate = get_cnot(c_sys01, e_sys1)

    # |00> -> |00>
    state = composite(gate, z0_z0)
    npt.assert_almost_equal(
        state.to_density_matrix(), z0_z0.to_density_matrix(), decimal=15
    )

    # |01> -> |11>
    state = composite(gate, z0_z1)
    npt.assert_almost_equal(
        state.to_density_matrix(), z1_z1.to_density_matrix(), decimal=15
    )

    # |10> -> |10>
    state = composite(gate, z1_z0)
    npt.assert_almost_equal(
        state.to_density_matrix(), z1_z0.to_density_matrix(), decimal=15
    )

    # |11> -> |01>
    state = composite(gate, z1_z1)
    npt.assert_almost_equal(
        state.to_density_matrix(), z0_z1.to_density_matrix(), decimal=15
    )

    # Test that not 2qubits ElementalSystem
    with pytest.raises(ValueError):
        get_cnot(c_sys0, e_sys0)

    # Test that not 4 dim ElementalSystem
    e_sys2 = ElementalSystem(2, matrix_basis.get_gell_mann_basis())
    e_sys3 = ElementalSystem(3, matrix_basis.get_gell_mann_basis())
    c_sys23 = CompositeSystem([e_sys2, e_sys3])
    with pytest.raises(ValueError):
        get_cnot(c_sys23, e_sys2)


def test_get_cz():
    # prepare gate
    e_sys0 = ElementalSystem(0, matrix_basis.get_comp_basis())
    c_sys0 = CompositeSystem([e_sys0])
    e_sys1 = ElementalSystem(1, matrix_basis.get_comp_basis())
    c_sys1 = CompositeSystem([e_sys1])

    c_sys01 = CompositeSystem([e_sys0, e_sys1])

    # prepare states
    y0_c_sys0 = get_y0_1q(c_sys0)
    y1_c_sys0 = get_y1_1q(c_sys0)
    y0_c_sys1 = get_y0_1q(c_sys1)
    y1_c_sys1 = get_y1_1q(c_sys1)
    y0_y0 = tensor_product(y0_c_sys0, y0_c_sys1)
    y0_y1 = tensor_product(y0_c_sys0, y1_c_sys1)
    y1_y0 = tensor_product(y1_c_sys0, y0_c_sys1)
    y1_y1 = tensor_product(y1_c_sys0, y1_c_sys1)

    ### gete
    gate = get_cz(c_sys01)

    # |i,i> -> |i,i>
    state = composite(gate, y0_y0)
    assert np.all(state.to_density_matrix() == y0_y0.to_density_matrix())

    # |i,-i> -> |i,-i>
    state = composite(gate, y0_y1)
    assert np.all(state.to_density_matrix() == y0_y1.to_density_matrix())

    # |-i,i> -> |-i,-i>
    state = composite(gate, y1_y0)
    assert np.all(state.to_density_matrix() == y1_y1.to_density_matrix())

    # |-i,-i> -> |-i,i>
    state = composite(gate, y1_y1)
    assert np.all(state.to_density_matrix() == y1_y0.to_density_matrix())

    # Test that not 2qubits ElementalSystem
    with pytest.raises(ValueError):
        get_cnot(c_sys0, e_sys0)

    # Test that not 4 dim ElementalSystem
    e_sys2 = ElementalSystem(2, matrix_basis.get_gell_mann_basis())
    e_sys3 = ElementalSystem(3, matrix_basis.get_gell_mann_basis())
    c_sys23 = CompositeSystem([e_sys2, e_sys3])
    with pytest.raises(ValueError):
        get_cnot(c_sys23, e_sys2)


def test_get_swap():
    # prepare gate
    e_sys0 = ElementalSystem(0, matrix_basis.get_comp_basis())
    c_sys0 = CompositeSystem([e_sys0])
    e_sys1 = ElementalSystem(1, matrix_basis.get_comp_basis())
    c_sys1 = CompositeSystem([e_sys1])

    c_sys01 = CompositeSystem([e_sys0, e_sys1])

    # prepare states
    z0_c_sys0 = get_z0_1q(c_sys0)
    z1_c_sys0 = get_z1_1q(c_sys0)
    z0_c_sys1 = get_z0_1q(c_sys1)
    z1_c_sys1 = get_z1_1q(c_sys1)
    z0_z0 = tensor_product(z0_c_sys0, z0_c_sys1)
    z0_z1 = tensor_product(z0_c_sys0, z1_c_sys1)
    z1_z0 = tensor_product(z1_c_sys0, z0_c_sys1)
    z1_z1 = tensor_product(z1_c_sys0, z1_c_sys1)

    ### gete
    gate = get_swap(c_sys01)

    # |00> -> |00>
    state = composite(gate, z0_z0)
    assert np.all(state.to_density_matrix() == z0_z0.to_density_matrix())

    # |01> -> |10>
    state = composite(gate, z0_z1)
    assert np.all(state.to_density_matrix() == z1_z0.to_density_matrix())

    # |10> -> |01>
    state = composite(gate, z1_z0)
    assert np.all(state.to_density_matrix() == z0_z1.to_density_matrix())

    # |11> -> |11>
    state = composite(gate, z1_z1)
    assert np.all(state.to_density_matrix() == z1_z1.to_density_matrix())

    # Test that not 2qubits ElementalSystem
    with pytest.raises(ValueError):
        get_cnot(c_sys0, e_sys0)

    # Test that not 4 dim ElementalSystem
    e_sys2 = ElementalSystem(2, matrix_basis.get_gell_mann_basis())
    e_sys3 = ElementalSystem(3, matrix_basis.get_gell_mann_basis())
    c_sys23 = CompositeSystem([e_sys2, e_sys3])
    with pytest.raises(ValueError):
        get_cnot(c_sys23, e_sys2)
