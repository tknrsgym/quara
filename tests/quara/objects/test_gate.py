import numpy as np
import numpy.testing as npt
import pytest

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import (
    Gate,
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
    is_ep,
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
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        gate = Gate(c_sys, hs)
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
        gate = Gate(c_sys, hs)
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

    def test_calc_choi_matrix(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # for I
        actual = get_i(c_sys).calc_choi_matrix()
        expected = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for X
        actual = get_x(c_sys).calc_choi_matrix()
        expected = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for Y
        actual = get_y(c_sys).calc_choi_matrix()
        expected = np.array([[0, 0, 0, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for Z
        actual = get_z(c_sys).calc_choi_matrix()
        expected = np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for H
        actual = get_h(c_sys).calc_choi_matrix()
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

    def test_calc_kraus_matrices(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        eye2 = np.eye(2, dtype=np.complex128)

        # for I
        actual = get_i(c_sys).calc_kraus_matrices()
        expected = [np.array([[1, 0], [0, 1]], dtype=np.complex128)]
        assert len(actual) == 1
        npt.assert_almost_equal(actual[0], expected[0], decimal=15)
        npt.assert_almost_equal(TestGate.calc_sum_of_kraus(actual), eye2, decimal=14)

        # for X
        actual = get_x(c_sys).calc_kraus_matrices()
        expected = [np.array([[0, 1], [1, 0]], dtype=np.complex128)]
        assert len(actual) == 1
        npt.assert_almost_equal(actual[0], expected[0], decimal=15)
        npt.assert_almost_equal(TestGate.calc_sum_of_kraus(actual), eye2, decimal=14)

        # for Y
        actual = get_y(c_sys).calc_kraus_matrices()
        expected = [np.array([[0, 1], [-1, 0]], dtype=np.complex128)]
        assert len(actual) == 1
        npt.assert_almost_equal(actual[0], expected[0], decimal=15)
        npt.assert_almost_equal(TestGate.calc_sum_of_kraus(actual), eye2, decimal=14)

        # for Z
        actual = get_z(c_sys).calc_kraus_matrices()
        expected = [np.array([[-1, 0], [0, 1]], dtype=np.complex128)]
        assert len(actual) == 1
        npt.assert_almost_equal(actual[0], expected[0], decimal=15)
        npt.assert_almost_equal(TestGate.calc_sum_of_kraus(actual), eye2, decimal=14)

        # for H
        actual = get_h(c_sys).calc_kraus_matrices()
        expected = [1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)]
        assert len(actual) == 1
        npt.assert_almost_equal(actual[0], expected[0], decimal=15)
        npt.assert_almost_equal(TestGate.calc_sum_of_kraus(actual), eye2, decimal=14)

        # Kraus matrix does not exist(HS is not CP)
        hs = np.array(
            [[-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        actual = Gate(c_sys, hs).calc_kraus_matrices()
        assert len(actual) == 0

    def test_is_ep(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # case: EP
        x = get_x(c_sys)
        assert is_ep(x.hs, x.get_basis()) == True
        y = get_y(c_sys)
        assert is_ep(y.hs, y.get_basis()) == True
        z = get_z(c_sys)
        assert is_ep(z.hs, z.get_basis()) == True

        # case: not EP
        hs = np.array([[1j, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        assert is_ep(hs, matrix_basis.get_comp_basis()) == False

    def test_calc_process_matrix(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        i = get_i(c_sys)
        x = get_x(c_sys)
        y = get_y(c_sys)
        z = get_z(c_sys)

        # for I
        actual = i.calc_process_matrix()
        expected = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for X
        actual = x.calc_process_matrix()
        expected = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for Y
        actual = y.calc_process_matrix()
        expected = np.array([[0, 0, 0, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for Z
        actual = z.calc_process_matrix()
        expected = np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]])
        npt.assert_almost_equal(actual, expected, decimal=15)


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
    gate = Gate(c_sys, hs)
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

    # Test that not 1qubit ElementalSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        get_i(c_sys)


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
        get_i(c_sys)


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
        get_i(c_sys)


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
        get_i(c_sys)


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
        get_i(c_sys)


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
        get_i(c_sys)


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
        get_i(c_sys)


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
        get_i(c_sys)


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
        get_i(c_sys)


def test_get_cnot():
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

    ### gete: control bit is 1st qubit
    gate = get_cnot(c_sys01, e_sys0)

    # |00> -> |00>
    state = composite(gate, z0_z0)
    assert np.all(state.get_density_matrix() == z0_z0.get_density_matrix())

    # |01> -> |01>
    state = composite(gate, z0_z1)
    assert np.all(state.get_density_matrix() == z0_z1.get_density_matrix())

    # |10> -> |11>
    state = composite(gate, z1_z0)
    assert np.all(state.get_density_matrix() == z1_z1.get_density_matrix())

    # |11> -> |10>
    state = composite(gate, z1_z1)
    assert np.all(state.get_density_matrix() == z1_z0.get_density_matrix())

    ### gete: control bit is 2st qubit
    gate = get_cnot(c_sys01, e_sys1)

    # |00> -> |00>
    state = composite(gate, z0_z0)
    assert np.all(state.get_density_matrix() == z0_z0.get_density_matrix())

    # |01> -> |11>
    state = composite(gate, z0_z1)
    assert np.all(state.get_density_matrix() == z1_z1.get_density_matrix())

    # |10> -> |10>
    state = composite(gate, z1_z0)
    assert np.all(state.get_density_matrix() == z1_z0.get_density_matrix())

    # |11> -> |01>
    state = composite(gate, z1_z1)
    assert np.all(state.get_density_matrix() == z0_z1.get_density_matrix())

    # Test that not 2qubits ElementalSystem
    with pytest.raises(ValueError):
        get_cnot(c_sys0, e_sys0)


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
    assert np.all(state.get_density_matrix() == y0_y0.get_density_matrix())

    # |i,-i> -> |i,-i>
    state = composite(gate, y0_y1)
    assert np.all(state.get_density_matrix() == y0_y1.get_density_matrix())

    # |-i,i> -> |-i,-i>
    state = composite(gate, y1_y0)
    assert np.all(state.get_density_matrix() == y1_y1.get_density_matrix())

    # |-i,-i> -> |-i,i>
    state = composite(gate, y1_y1)
    assert np.all(state.get_density_matrix() == y1_y0.get_density_matrix())

    # Test that not 2qubits ElementalSystem
    with pytest.raises(ValueError):
        get_cnot(c_sys0, e_sys0)


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
    assert np.all(state.get_density_matrix() == z0_z0.get_density_matrix())

    # |01> -> |10>
    state = composite(gate, z0_z1)
    assert np.all(state.get_density_matrix() == z1_z0.get_density_matrix())

    # |10> -> |01>
    state = composite(gate, z1_z0)
    assert np.all(state.get_density_matrix() == z0_z1.get_density_matrix())

    # |11> -> |11>
    state = composite(gate, z1_z1)
    assert np.all(state.get_density_matrix() == z1_z1.get_density_matrix())

    # Test that not 2qubits ElementalSystem
    with pytest.raises(ValueError):
        get_cnot(c_sys0, e_sys0)
