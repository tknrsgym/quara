import numpy as np
import numpy.testing as npt
import pytest

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.effective_lindbladian import EffectiveLindbladian
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import (
    Gate,
    convert_var_index_to_gate_index,
    convert_gate_index_to_var_index,
    convert_var_to_gate,
    convert_hs_to_var,
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
    to_hs_from_choi,
    to_hs_from_choi_with_dict,
    to_hs_from_choi_with_sparsity,
    get_depolarizing_channel,
    get_x_rotation,
    get_amplitutde_damping_channel,
)
from quara.objects.operators import compose_qoperations, tensor_product
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

    def test_init_is_physicality_required(self):
        e_sys = ElementalSystem(1, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])

        # gate is not TP
        hs_not_tp = np.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        with pytest.raises(ValueError):
            Gate(c_sys, hs_not_tp)
        with pytest.raises(ValueError):
            Gate(c_sys, hs_not_tp, is_physicality_required=True)

        # gate is not CP
        hs_not_cp = np.array(
            [[-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        with pytest.raises(ValueError):
            Gate(c_sys, hs_not_cp)
        with pytest.raises(ValueError):
            Gate(c_sys, hs_not_cp, is_physicality_required=True)

        # case: when is_physicality_required is False, it is not happened ValueError
        Gate(c_sys, hs_not_tp, is_physicality_required=False)
        Gate(c_sys, hs_not_cp, is_physicality_required=False)

    def test_access_is_physicality_required(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        hs = np.array(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        gate = Gate(c_sys, hs)
        assert gate.is_physicality_required == True

        # Test that "is_physicality_required" cannot be updated
        with pytest.raises(AttributeError):
            gate.is_physicality_required = False

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

    def test_is_physical(self):
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])

        hs = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        gate = Gate(c_sys, hs)
        assert gate.is_physical() == True

        # gate is not TP
        hs_not_tp = np.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        gate = Gate(c_sys, hs_not_tp, is_physicality_required=False)
        assert gate.is_physical() == False

        # gate is not CP
        hs_not_cp = np.array(
            [[-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        gate = Gate(c_sys, hs_not_cp, is_physicality_required=False)
        assert gate.is_physical() == False

    def test_set_zero(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        gate = get_z(c_sys)
        gate.set_zero()

        expected = np.zeros((4, 4), dtype=np.float64)
        npt.assert_almost_equal(gate.hs, expected, decimal=15)
        assert gate.dim == 2
        assert gate.is_physicality_required == False
        assert gate.is_estimation_object == True
        assert gate.on_para_eq_constraint == True
        assert gate.on_algo_eq_constraint == True
        assert gate.on_algo_ineq_constraint == True
        assert gate.eps_proj_physical == Settings.get_atol() / 10.0

    def test_zero_obj(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        gate = get_z(c_sys)
        zero = gate.generate_zero_obj()

        expected = np.zeros((4, 4), dtype=np.float64)
        npt.assert_almost_equal(zero.hs, expected, decimal=15)
        assert zero.dim == gate.dim
        assert zero.is_physicality_required == False
        assert zero.is_estimation_object == False
        assert zero.on_para_eq_constraint == gate.on_para_eq_constraint
        assert zero.on_algo_eq_constraint == gate.on_algo_eq_constraint
        assert zero.on_algo_ineq_constraint == gate.on_algo_ineq_constraint
        assert zero.eps_proj_physical == gate.eps_proj_physical

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

    def test_to_stacked_vector(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        gate = get_z(c_sys)
        vector = gate.to_stacked_vector()

        expected = np.array(
            [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1], dtype=np.float64
        )
        npt.assert_almost_equal(vector, expected, decimal=15)

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
        assert z.is_eq_constraint_satisfied() == True

        # case: not TP
        hs = np.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        gate = Gate(c_sys, hs, is_physicality_required=False)
        assert gate.is_tp() == False
        assert gate.is_eq_constraint_satisfied() == False

    def test_is_cp(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # case: CP
        x = get_x(c_sys)
        assert x.is_cp() == True
        assert x.is_ineq_constraint_satisfied() == True

        y = get_y(c_sys)
        assert y.is_cp() == True
        assert y.is_ineq_constraint_satisfied() == True

        z = get_z(c_sys)
        assert z.is_cp() == True
        assert z.is_ineq_constraint_satisfied() == True

        # case: not CP
        hs = np.array(
            [[-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        gate = Gate(c_sys, hs, is_physicality_required=False)
        assert gate.is_cp() == False
        assert gate.is_ineq_constraint_satisfied() == False

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

    def test_to_choi_matrix_with_dict(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # for I
        actual = get_i(c_sys).to_choi_matrix_with_dict()
        expected = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for X
        actual = get_x(c_sys).to_choi_matrix_with_dict()
        expected = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for Y
        actual = get_y(c_sys).to_choi_matrix_with_dict()
        expected = np.array([[0, 0, 0, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for Z
        actual = get_z(c_sys).to_choi_matrix_with_dict()
        expected = np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for H
        actual = get_h(c_sys).to_choi_matrix_with_dict()
        expected = (
            1
            / 2
            * np.array([[1, 1, 1, -1], [1, 1, 1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]])
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_to_choi_matrix_with_sparsity(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # for I
        actual = get_i(c_sys).to_choi_matrix_with_sparsity()
        expected = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for X
        actual = get_x(c_sys).to_choi_matrix_with_sparsity()
        expected = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for Y
        actual = get_y(c_sys).to_choi_matrix_with_sparsity()
        expected = np.array([[0, 0, 0, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for Z
        actual = get_z(c_sys).to_choi_matrix_with_sparsity()
        expected = np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]])
        npt.assert_almost_equal(actual, expected, decimal=15)

        # for H
        actual = get_h(c_sys).to_choi_matrix_with_sparsity()
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
        expected = [np.array([[1, 0], [0, -1]], dtype=np.complex128)]
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
        actual = Gate(c_sys, hs, is_physicality_required=False).to_kraus_matrices()
        assert len(actual) == 0

        # for swap
        e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
        e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys1, e_sys2])
        eye4 = np.eye(4, dtype=np.complex128)

        actual = get_swap(c_sys).to_kraus_matrices()
        expected = [
            np.array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=np.complex128,
            )
        ]
        assert len(actual) == 1
        npt.assert_almost_equal(actual[0], expected[0], decimal=15)
        npt.assert_almost_equal(TestGate.calc_sum_of_kraus(actual), eye4, decimal=14)

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

    def test_generate_from_var(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        matrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
        )
        hs = convert_hs(
            matrix, matrix_basis.get_normalized_pauli_basis(), c_sys.basis()
        )
        hs = hs.real.astype(np.float64)
        init_is_physicality_required = True
        init_is_estimation_object = False
        init_on_para_eq_constraint = True
        init_on_algo_eq_constraint = False
        init_on_algo_ineq_constraint = True
        init_eps_proj_physical = 10 ** (-5)

        source_gate = Gate(
            c_sys,
            hs=hs,
            is_physicality_required=init_is_physicality_required,
            is_estimation_object=init_is_estimation_object,
            on_para_eq_constraint=init_on_para_eq_constraint,
            on_algo_eq_constraint=init_on_algo_eq_constraint,
            on_algo_ineq_constraint=init_on_algo_ineq_constraint,
            eps_proj_physical=init_eps_proj_physical,
        )

        # Case 1: default
        var = np.array([0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64)
        # Act
        actual = source_gate.generate_from_var(var)
        # Assert
        expected = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
        )
        npt.assert_almost_equal(actual.hs, expected, decimal=15)
        assert actual.composite_system is c_sys
        assert actual.is_physicality_required is init_is_physicality_required
        assert actual.is_estimation_object is init_is_estimation_object
        assert actual.on_para_eq_constraint is init_on_para_eq_constraint
        assert actual.on_algo_eq_constraint is init_on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is init_on_algo_ineq_constraint
        assert actual.eps_proj_physical is init_eps_proj_physical

        # Case 2:
        # Arrange
        var = np.array(
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64
        )
        source_is_physicality_required = False
        source_is_estimation_object = True
        source_on_para_eq_constraint = False
        source_on_algo_eq_constraint = True
        source_on_algo_ineq_constraint = False
        source_eps_proj_physical = 10 ** (-2)

        # Act
        actual = source_gate.generate_from_var(
            var,
            is_physicality_required=source_is_physicality_required,
            is_estimation_object=source_is_estimation_object,
            on_para_eq_constraint=source_on_para_eq_constraint,
            on_algo_eq_constraint=source_on_algo_eq_constraint,
            on_algo_ineq_constraint=source_on_algo_ineq_constraint,
            eps_proj_physical=source_eps_proj_physical,
        )

        # Assert
        expected = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
        )
        assert np.all(actual.hs == expected)
        assert actual.composite_system is c_sys
        assert actual.is_physicality_required is source_is_physicality_required
        assert actual.is_estimation_object is source_is_estimation_object
        assert actual.on_para_eq_constraint is source_on_para_eq_constraint
        assert actual.on_algo_eq_constraint is source_on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is source_on_algo_ineq_constraint
        assert actual.eps_proj_physical == source_eps_proj_physical

    def test_generate_origin_obj(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hs = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
        )

        gate = Gate(
            c_sys=c_sys,
            hs=hs,
            is_physicality_required=False,
            is_estimation_object=True,
            on_para_eq_constraint=False,
            on_algo_eq_constraint=True,
            on_algo_ineq_constraint=False,
            eps_proj_physical=0.2,
        )

        # Act
        actual = gate.generate_origin_obj()
        expected_hs = np.array(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        assert np.all(actual.hs == expected_hs)
        # `is_physicality_required` and `is_estimation_object` are always False
        assert actual.composite_system is c_sys
        assert actual.is_physicality_required is False
        assert actual.is_estimation_object is False
        assert actual.on_para_eq_constraint is False
        assert actual.on_algo_eq_constraint is True
        assert actual.on_algo_ineq_constraint is False
        assert actual.eps_proj_physical == 0.2

    def test_add(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        gate_1 = get_x(c_sys)
        gate_2 = get_y(c_sys)

        actual = gate_1 + gate_2

        # Assert
        expected_hs = np.array(
            [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -2]], dtype=np.float64
        )
        npt.assert_almost_equal(actual.hs, expected_hs, decimal=15)
        assert actual.composite_system is c_sys
        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint is gate_1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint is gate_1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is gate_1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == gate_1.eps_proj_physical

    def test_sub(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        gate_1 = get_x(c_sys)
        gate_2 = get_y(c_sys)

        actual = gate_1 - gate_2

        # Assert
        expected_hs = np.array(
            [[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, -2, 0], [0, 0, 0, -0]], dtype=np.float64
        )
        npt.assert_almost_equal(actual.hs, expected_hs, decimal=15)
        assert actual.composite_system is c_sys
        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint is gate_1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint is gate_1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is gate_1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == gate_1.eps_proj_physical

    def test_mul_rmul(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        gate_1 = get_x(c_sys)

        # Case 1: mul
        # Act
        actual = gate_1 * 2

        # Assert
        expected_hs = np.array(
            [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, -2, 0], [0, 0, 0, -2]], dtype=np.float64
        )
        npt.assert_almost_equal(actual.hs, expected_hs, decimal=15)
        assert actual.composite_system is c_sys
        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint is gate_1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint is gate_1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is gate_1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == gate_1.eps_proj_physical

        # Case 2: rmul
        # Act
        actual = 0.3 * gate_1

        # Assert
        expected_hs = np.array(
            [[0.3, 0, 0, 0], [0, 0.3, 0, 0], [0, 0, -0.3, 0], [0, 0, 0, -0.3]],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual.hs, expected_hs, decimal=15)
        assert actual.composite_system is c_sys
        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint is gate_1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint is gate_1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is gate_1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == gate_1.eps_proj_physical

    def test_truediv(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        gate_1 = get_x(c_sys)

        # Case 1:
        # Act
        actual = gate_1 / 2

        # Assert
        expected_hs = np.array(
            [[1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, -1 / 2, 0], [0, 0, 0, -1 / 2]],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual.hs, expected_hs, decimal=15)
        assert actual.composite_system is c_sys
        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint is gate_1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint is gate_1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is gate_1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == gate_1.eps_proj_physical

        # Case 2:
        # Act
        actual = gate_1 / 0.5

        # Assert
        expected_hs = np.array(
            [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, -2, 0], [0, 0, 0, -2]],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual.hs, expected_hs, decimal=15)
        assert actual.composite_system is c_sys
        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint is gate_1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint is gate_1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is gate_1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == gate_1.eps_proj_physical

    @pytest.mark.skip(reason="The result depends on the Python version and OS.")
    def test_truediv_zero_division(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        gate_1 = get_x(c_sys)

        # Act
        actual = gate_1 / 0

        # Assert
        expected_hs = np.array(
            [
                [np.inf, np.nan, np.nan, np.nan],
                [np.nan, np.inf, np.nan, np.nan],
                [np.nan, np.nan, -np.inf, np.nan],
                [np.nan, np.nan, np.nan, -np.inf],
            ],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual.hs, expected_hs, decimal=15)
        assert actual.composite_system is c_sys
        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint is gate_1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint is gate_1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is gate_1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == gate_1.eps_proj_physical

    def test_calc_gradient(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # Case 1: on_para_eq_constraint=True
        # Arrange
        hs = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
        )
        init_is_physicality_required = True
        init_is_estimation_object = False
        init_on_para_eq_constraint = True
        init_on_algo_eq_constraint = False
        init_on_algo_ineq_constraint = True
        init_eps_proj_physical = 10 ** (-5)

        gate = Gate(
            c_sys=c_sys,
            hs=hs,
            is_physicality_required=init_is_physicality_required,
            is_estimation_object=init_is_estimation_object,
            on_para_eq_constraint=init_on_para_eq_constraint,
            on_algo_eq_constraint=init_on_algo_eq_constraint,
            on_algo_ineq_constraint=init_on_algo_ineq_constraint,
            eps_proj_physical=init_eps_proj_physical,
        )

        # Act
        actual = gate.calc_gradient(var_index=1)

        # Assert
        expected = np.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        npt.assert_almost_equal(actual.hs, expected, decimal=15)
        assert actual.composite_system is c_sys
        assert actual.is_physicality_required is False
        assert actual.is_estimation_object is init_is_estimation_object
        assert actual.on_para_eq_constraint is init_on_para_eq_constraint
        assert actual.on_algo_eq_constraint is init_on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is init_on_algo_ineq_constraint
        assert actual.eps_proj_physical is init_eps_proj_physical

        # Case 2: on_para_eq_constraint=False
        hs = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
        )
        init_is_physicality_required = False
        init_is_estimation_object = True
        init_on_para_eq_constraint = False
        init_on_algo_eq_constraint = True
        init_on_algo_ineq_constraint = False
        init_eps_proj_physical = 10 ** (-2)

        # Act
        gate = Gate(
            c_sys=c_sys,
            hs=hs,
            is_physicality_required=init_is_physicality_required,
            is_estimation_object=init_is_estimation_object,
            on_para_eq_constraint=init_on_para_eq_constraint,
            on_algo_eq_constraint=init_on_algo_eq_constraint,
            on_algo_ineq_constraint=init_on_algo_ineq_constraint,
            eps_proj_physical=init_eps_proj_physical,
        )

        actual = gate.calc_gradient(var_index=1)
        expected = np.array(
            [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        npt.assert_almost_equal(actual.hs, expected, decimal=15)
        assert actual.composite_system is c_sys
        assert actual.is_physicality_required is False
        assert actual.is_estimation_object is init_is_estimation_object
        assert actual.on_para_eq_constraint is init_on_para_eq_constraint
        assert actual.on_algo_eq_constraint is init_on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is init_on_algo_ineq_constraint
        assert actual.eps_proj_physical is init_eps_proj_physical

    def test_calc_proj_eq_constraint(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hs = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float64,
        )
        gate = Gate(c_sys=c_sys, hs=hs, is_physicality_required=False)

        # Act
        actual = gate.calc_proj_eq_constraint()

        # Assert
        expected_hs = np.array(
            [[1, 0, 0, 0], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual.hs, expected_hs, decimal=15)
        assert actual.composite_system is c_sys
        assert actual.is_physicality_required is gate.is_physicality_required
        assert actual.is_estimation_object is gate.is_estimation_object
        assert actual.on_para_eq_constraint is gate.on_para_eq_constraint
        assert actual.on_algo_eq_constraint is gate.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is gate.on_algo_ineq_constraint
        assert actual.eps_proj_physical is gate.eps_proj_physical

    def test_calc_proj_eq_constraint_with_var(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hs = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float64,
        )
        gate = Gate(c_sys=c_sys, hs=hs, is_physicality_required=False)

        # case 1: on_para_eq_constraint: default(True)
        actual = gate.calc_proj_eq_constraint_with_var(c_sys, gate.to_var())
        expected = np.array(
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case 2: on_para_eq_constraint=True
        actual = gate.calc_proj_eq_constraint_with_var(
            c_sys, gate.to_var(), on_para_eq_constraint=True
        )
        expected = np.array(
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case 3: on_para_eq_constraint=False
        var = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            dtype=np.float64,
        )
        actual = gate.calc_proj_eq_constraint_with_var(
            c_sys, var, on_para_eq_constraint=False
        )
        expected = np.array(
            [1, 0, 0, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_proj_ineq_constraint(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        gate = get_x(c_sys)

        # Act
        actual = gate.calc_proj_ineq_constraint()

        # Assert
        expected_choi = np.array(
            [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
        )
        npt.assert_almost_equal(actual.to_choi_matrix(), expected_choi, decimal=14)
        expected_hs = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
        )
        npt.assert_almost_equal(actual.hs, expected_hs, decimal=15)
        assert actual.composite_system is c_sys
        assert actual.is_physicality_required is gate.is_physicality_required
        assert actual.is_estimation_object is gate.is_estimation_object
        assert actual.on_para_eq_constraint is gate.on_para_eq_constraint
        assert actual.on_algo_eq_constraint is gate.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is gate.on_algo_ineq_constraint
        assert actual.eps_proj_physical is gate.eps_proj_physical

    def test_calc_proj_ineq_constraint_with_var(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        gate = get_x(c_sys)

        # case 1: on_para_eq_constraint: default(True)
        actual = gate.calc_proj_ineq_constraint_with_var(c_sys, gate.to_var())
        expected = np.array([0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case 2: on_para_eq_constraint=True
        actual = gate.calc_proj_ineq_constraint_with_var(
            c_sys, gate.to_var(), on_para_eq_constraint=True
        )
        expected = np.array([0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case 3: on_para_eq_constraint=False
        var = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1])
        actual = gate.calc_proj_ineq_constraint_with_var(
            c_sys, var, on_para_eq_constraint=False
        )
        expected = np.array(
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_convert_var_to_stacked_vector(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        gate = get_x(c_sys)
        expected = np.array(
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64
        )

        # case 1: on_para_eq_constraint: default(True)
        var = np.array([0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64)
        actual = gate.convert_var_to_stacked_vector(c_sys, var)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 2: on_para_eq_constraint=True
        var = np.array([0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64)
        actual = gate.convert_var_to_stacked_vector(
            c_sys, var, on_para_eq_constraint=True
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 3: on_para_eq_constraint=False
        var = np.array(
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64
        )
        actual = gate.convert_var_to_stacked_vector(
            c_sys, var, on_para_eq_constraint=False
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_convert_stacked_vector_to_var(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        gate = get_x(c_sys)
        stacked_vector = np.array(
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64
        )

        # case 1: on_para_eq_constraint: default(True)
        actual = gate.convert_stacked_vector_to_var(c_sys, stacked_vector)
        expected = np.array([0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 2: on_para_eq_constraint=True
        actual = gate.convert_stacked_vector_to_var(
            c_sys, stacked_vector, on_para_eq_constraint=True
        )
        expected = np.array([0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 3: on_para_eq_constraint=False
        actual = gate.convert_stacked_vector_to_var(
            c_sys, stacked_vector, on_para_eq_constraint=False
        )
        expected = np.array(
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64
        )
        npt.assert_almost_equal(actual, expected, decimal=15)


def test_to_hs_from_choi():
    # Case 1:
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    gate = get_x(c_sys)
    source_choi = gate.to_choi_matrix()
    # Act
    actual = to_hs_from_choi(c_sys, source_choi)
    # Assert
    expected = gate.hs
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case 2:
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    gate = get_y(c_sys)
    source_choi = gate.to_choi_matrix()
    # Act
    actual = to_hs_from_choi(c_sys, source_choi)
    # Assert
    expected = gate.hs
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case 3:
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    gate = get_z(c_sys)
    source_choi = gate.to_choi_matrix()
    # Act
    actual = to_hs_from_choi(c_sys, source_choi)
    # Assert
    expected = gate.hs
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case 4:
    # Arrange
    hs = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        dtype=np.float64,
    )
    gate = Gate(c_sys=c_sys, hs=hs, is_physicality_required=False)
    source_choi = gate.to_choi_matrix()
    # Act
    actual = to_hs_from_choi(c_sys, source_choi)
    # Assert
    expected = gate.hs
    npt.assert_almost_equal(actual, expected, decimal=14)


def test_to_hs_from_choi_with_dict():
    # Case 1:
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    gate = get_x(c_sys)
    source_choi = gate.to_choi_matrix()
    # Act
    actual = to_hs_from_choi_with_dict(c_sys, source_choi)
    # Assert
    expected = gate.hs
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case 2:
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    gate = get_y(c_sys)
    source_choi = gate.to_choi_matrix()
    # Act
    actual = to_hs_from_choi_with_dict(c_sys, source_choi)
    # Assert
    expected = gate.hs
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case 3:
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    gate = get_z(c_sys)
    source_choi = gate.to_choi_matrix()
    # Act
    actual = to_hs_from_choi_with_dict(c_sys, source_choi)
    # Assert
    expected = gate.hs
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_to_hs_from_choi_with_sparsity():
    # Case 1:
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    gate = get_x(c_sys)
    source_choi = gate.to_choi_matrix()
    # Act
    actual = to_hs_from_choi_with_sparsity(c_sys, source_choi)
    # Assert
    expected = gate.hs
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case 2:
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    gate = get_y(c_sys)
    source_choi = gate.to_choi_matrix()
    # Act
    actual = to_hs_from_choi_with_sparsity(c_sys, source_choi)
    # Assert
    expected = gate.hs
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case 3:
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    gate = get_z(c_sys)
    source_choi = gate.to_choi_matrix()
    # Act
    actual = to_hs_from_choi_with_sparsity(c_sys, source_choi)
    # Assert
    expected = gate.hs
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_convert_var_index_to_gate_index():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # default
    actual = convert_var_index_to_gate_index(c_sys, 11)
    assert actual == (3, 3)

    # on_para_eq_constraint=True
    actual = convert_var_index_to_gate_index(c_sys, 11, on_para_eq_constraint=True)
    assert actual == (3, 3)

    # on_para_eq_constraint=False
    actual = convert_var_index_to_gate_index(c_sys, 15, on_para_eq_constraint=False)
    assert actual == (3, 3)


def test_convert_gate_index_to_var_index():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # default
    actual = convert_gate_index_to_var_index(c_sys, (3, 3))
    assert actual == 11

    # on_para_eq_constraint=True
    actual = convert_gate_index_to_var_index(c_sys, (3, 3), on_para_eq_constraint=True)
    assert actual == 11

    # on_para_eq_constraint=False
    actual = convert_gate_index_to_var_index(c_sys, (3, 3), on_para_eq_constraint=False)
    assert actual == 15


def test_convert_var_to_gate():
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # Case 1: default
    # Arrange
    var = np.array([0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64)
    # Act
    actual = convert_var_to_gate(c_sys, var)
    # Assert
    expected = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # Case 2: on_para_eq_constraint=True
    # Arrange
    var = np.array([0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64)
    # Act
    actual = convert_var_to_gate(c_sys, var, on_para_eq_constraint=True)
    # Assert
    expected = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # Case 3: on_para_eq_constraint=False
    # Arrange
    var = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64)
    # Act
    actual = convert_var_to_gate(c_sys, var, on_para_eq_constraint=False)
    # Assert
    expected = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    npt.assert_almost_equal(actual.hs, expected, decimal=15)


def test_convert_var_to_gate_2q():
    # Arrange
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys_2q = CompositeSystem([e_sys0, e_sys1])

    # Case 1:
    # Arrange
    source = np.array(list(range(16, 16 * 16)), dtype=np.float64)
    # Act
    actual = convert_var_to_gate(
        c_sys=c_sys_2q,
        var=source,
        on_para_eq_constraint=True,
        is_physicality_required=False,
    )
    # Assert
    expected = np.array(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + list(range(16, 16 * 16)),
        dtype=np.float64,
    )
    expected = expected.reshape(16, 16)
    npt.assert_almost_equal(actual.hs, expected, decimal=15)
    assert actual.composite_system is c_sys_2q

    # Case 2:
    # Arrange
    source = np.array(list(range(16 * 16)), dtype=np.float64)
    # Act
    actual = convert_var_to_gate(
        c_sys_2q,
        source,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    # Assert
    expected = np.array(range(16 * 16), dtype=np.float64)
    expected = expected.reshape(16, 16)
    npt.assert_almost_equal(actual.hs, expected, decimal=15)
    assert actual.composite_system is c_sys_2q


def test_convert_gate_to_var():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # default
    hs = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    actual = convert_hs_to_var(c_sys, hs)
    expected = np.array([0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # on_para_eq_constraint=True
    hs = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    actual = convert_hs_to_var(c_sys, hs, on_para_eq_constraint=True)
    expected = np.array([0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # on_para_eq_constraint=False
    hs = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    actual = convert_hs_to_var(c_sys, hs, on_para_eq_constraint=False)
    expected = np.array(
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64
    )
    npt.assert_almost_equal(actual, hs.flatten(), decimal=15)


def test_convert_gate_to_var_2q():
    # Arrange
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys_2q = CompositeSystem([e_sys0, e_sys1])
    hs = np.array(range(16 * 16), dtype=np.float64)
    hs = hs.reshape(16, 16)
    gate = Gate(c_sys_2q, hs=hs, is_physicality_required=False)
    hs = np.array(range(16 * 16), dtype=np.float64)
    hs = hs.reshape(16, 16)

    # Act
    actual = convert_hs_to_var(c_sys_2q, hs, on_para_eq_constraint=True)
    # Assert
    expected = np.array(list(range(16, 16 * 16)), dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Act
    actual = convert_hs_to_var(c_sys_2q, hs, on_para_eq_constraint=False)
    # Assert
    expected = np.array(list(range(16 * 16)), dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)


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

    # on_para_eq_constraint=True
    hs = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    actual = calc_gradient_from_gate(c_sys, hs, 1, on_para_eq_constraint=True)
    expected = np.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
    )
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # on_para_eq_constraint=False
    hs = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    actual = calc_gradient_from_gate(c_sys, hs, 1, on_para_eq_constraint=False)
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

    # case: u is not unitary
    hs = np.array(
        [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    gate = Gate(c_sys, hs, is_physicality_required=False)
    with pytest.raises(ValueError):
        calc_agf(z.hs, gate)

    # case: g is not Gate
    hs = np.array(
        [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    lind = EffectiveLindbladian(c_sys, hs, is_physicality_required=False)
    with pytest.raises(ValueError):
        calc_agf(lind, i)

    # case: u is not Gate
    hs = np.array(
        [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    lind = EffectiveLindbladian(c_sys, hs, is_physicality_required=False)
    with pytest.raises(ValueError):
        calc_agf(i, lind)


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
    state = compose_qoperations(gate, z0_z0)
    npt.assert_almost_equal(
        state.to_density_matrix(), z0_z0.to_density_matrix(), decimal=15
    )

    # |01> -> |01>
    state = compose_qoperations(gate, z0_z1)
    npt.assert_almost_equal(
        state.to_density_matrix(), z0_z1.to_density_matrix(), decimal=15
    )

    # |10> -> |11>
    state = compose_qoperations(gate, z1_z0)
    npt.assert_almost_equal(
        state.to_density_matrix(), z1_z1.to_density_matrix(), decimal=15
    )

    # |11> -> |10>
    state = compose_qoperations(gate, z1_z1)
    npt.assert_almost_equal(
        state.to_density_matrix(), z1_z0.to_density_matrix(), decimal=15
    )

    ### gete: control bit is 2st qubit
    gate = get_cnot(c_sys01, e_sys1)

    # |00> -> |00>
    state = compose_qoperations(gate, z0_z0)
    npt.assert_almost_equal(
        state.to_density_matrix(), z0_z0.to_density_matrix(), decimal=15
    )

    # |01> -> |11>
    state = compose_qoperations(gate, z0_z1)
    npt.assert_almost_equal(
        state.to_density_matrix(), z1_z1.to_density_matrix(), decimal=15
    )

    # |10> -> |10>
    state = compose_qoperations(gate, z1_z0)
    npt.assert_almost_equal(
        state.to_density_matrix(), z1_z0.to_density_matrix(), decimal=15
    )

    # |11> -> |01>
    state = compose_qoperations(gate, z1_z1)
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
    state = compose_qoperations(gate, y0_y0)
    assert np.all(state.to_density_matrix() == y0_y0.to_density_matrix())

    # |i,-i> -> |i,-i>
    state = compose_qoperations(gate, y0_y1)
    assert np.all(state.to_density_matrix() == y0_y1.to_density_matrix())

    # |-i,i> -> |-i,-i>
    state = compose_qoperations(gate, y1_y0)
    assert np.all(state.to_density_matrix() == y1_y1.to_density_matrix())

    # |-i,-i> -> |-i,i>
    state = compose_qoperations(gate, y1_y1)
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
    state = compose_qoperations(gate, z0_z0)
    assert np.all(state.to_density_matrix() == z0_z0.to_density_matrix())

    # |01> -> |10>
    state = compose_qoperations(gate, z0_z1)
    assert np.all(state.to_density_matrix() == z1_z0.to_density_matrix())

    # |10> -> |01>
    state = compose_qoperations(gate, z1_z0)
    assert np.all(state.to_density_matrix() == z0_z1.to_density_matrix())

    # |11> -> |11>
    state = compose_qoperations(gate, z1_z1)
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


def test_get_depolarizing_channel():
    # Act
    actual = get_depolarizing_channel(p=0)
    # Assert
    expected = np.eye(4)
    npt.assert_almost_equal(actual.hs, expected, decimal=16)

    # Act
    actual = get_depolarizing_channel(p=0.05)
    # Assert
    expected = np.array(
        [[1, 0, 0, 0], [0, 0.95, 0, 0], [0, 0, 0.95, 0], [0, 0, 0, 0.95]]
    )
    npt.assert_almost_equal(actual.hs, expected, decimal=16)

    # Act
    actual = get_depolarizing_channel(p=1)
    # Assert
    expected = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    npt.assert_almost_equal(actual.hs, expected, decimal=16)

    # Array
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # Act
    actual = get_depolarizing_channel(p=0.05, c_sys=c_sys)
    # Assert
    expected = np.array(
        [[1, 0, 0, 0], [0, 0.95, 0, 0], [0, 0, 0.95, 0], [0, 0, 0, 0.95]]
    )
    npt.assert_almost_equal(actual.hs, expected, decimal=16)
    assert actual.composite_system is c_sys

    # 2qubit
    # Arange
    e_sys_1 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys_2 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys_2q = CompositeSystem([e_sys_1, e_sys_2])
    # Act
    actual = get_depolarizing_channel(0.1, c_sys_2q)
    # Assert
    expected = np.diag([0.9] * 16)
    expected[0][0] = 1
    npt.assert_almost_equal(actual.hs, expected, decimal=16)
    assert actual.composite_system is c_sys_2q


def test_get_depolarizing_channel_unexpected():
    # Array
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # Act & Assert
    with pytest.raises(ValueError):
        _ = get_depolarizing_channel(p=-0.1, c_sys=c_sys)

    # Act & Assert
    with pytest.raises(ValueError):
        _ = get_depolarizing_channel(p=1.1, c_sys=c_sys)


def test_get_x_rotation():
    # Act
    actual = get_x_rotation(theta=np.pi / 2)
    # Assert
    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
    npt.assert_almost_equal(actual.hs, expected, decimal=16)

    # Act
    actual = get_x_rotation(theta=np.pi)
    # Assert
    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
    npt.assert_almost_equal(actual.hs, expected, decimal=16)


def test_get_amplitutde_damping_channel():
    # Act
    actual = get_amplitutde_damping_channel(gamma=0.1)
    # Assert
    expected = np.array(
        [
            [1, 0, 0, 0],
            [0, np.sqrt(0.9), 0, 0],
            [0, 0, np.sqrt(0.9), 0],
            [0.1, 0, 0, 0.9],
        ]
    )
    npt.assert_almost_equal(actual.hs, expected, decimal=16)
