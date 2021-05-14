import sys

import numpy as np
import numpy.testing as npt
import pytest
from scipy.linalg import expm

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import Gate, convert_hs
from quara.objects.effective_lindbladian import EffectiveLindbladian
from quara.objects import effective_lindbladian as el
from quara.objects.operators import compose_qoperations, tensor_product
from quara.objects.state import get_y0_1q, get_y1_1q, get_z0_1q, get_z1_1q
from quara.settings import Settings


class TestEffectiveLindbladian:
    def test_init_error(self):
        # basis is not Hermitian
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])
        hs = np.zeros((4, 4))
        with pytest.raises(ValueError):
            EffectiveLindbladian(c_sys, hs)

        # 0th basis is not I/sqrt(dim)
        e_sys = ElementalSystem(0, matrix_basis.get_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hs = np.zeros((4, 4))
        with pytest.raises(ValueError):
            EffectiveLindbladian(c_sys, hs)

    def test_calc_h_mat(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hs = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        lindbladian = EffectiveLindbladian(c_sys, hs)
        actual = lindbladian.calc_h_mat()
        expected = np.array([[0, 0], [0, 0]], dtype=np.complex128)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_j_mat(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hs = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        lindbladian = EffectiveLindbladian(c_sys, hs)
        actual = lindbladian.calc_j_mat()
        expected = np.array([[0, 0], [0, 0]], dtype=np.complex128)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_k_mat(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hs = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        lindbladian = EffectiveLindbladian(c_sys, hs)
        actual = lindbladian.calc_k_mat()
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.complex128)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_h_part(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        # h=Z/sqrt(2)
        hs = (
            np.array(
                [[0, 0, 0, 0], [0, 0, -2, 0], [0, 2, 0, 0], [0, 0, 0, 0]],
                dtype=np.float64,
            )
            / np.sqrt(2)
        )
        lindbladian = EffectiveLindbladian(c_sys, hs)

        # mode_basis=default("hermitian_basis")
        actual = lindbladian.calc_h_part()
        expected = np.array(
            [[0, 0, 0, 0], [0, 0, -np.sqrt(2), 0], [0, np.sqrt(2), 0, 0], [0, 0, 0, 0]],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=14)

        # mode_basis="hermitian_basis"
        actual = lindbladian.calc_h_part(mode_basis="hermitian_basis")
        expected = np.array(
            [[0, 0, 0, 0], [0, 0, -np.sqrt(2), 0], [0, np.sqrt(2), 0, 0], [0, 0, 0, 0]],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=14)

        # mode_basis="comp_basis"
        actual = lindbladian.calc_h_part(mode_basis="comp_basis")
        expected = (
            np.array(
                [[0, 0, 0, 0], [0, -2j, 0, 0], [0, 0, 2j, 0], [0, 0, 0, 0]],
                dtype=np.complex128,
            )
            / np.sqrt(2)
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_j_part(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        # j=Z/sqrt(2)
        hs = (
            np.array(
                [[0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]],
                dtype=np.float64,
            )
            / np.sqrt(2)
        )
        lindbladian = EffectiveLindbladian(c_sys, hs, is_physicality_required=False)

        # mode_basis=default("hermitian_basis")
        actual = lindbladian.calc_j_part()
        expected = np.array(
            [[0, 0, 0, np.sqrt(2)], [0, 0, 0, 0], [0, 0, 0, 0], [np.sqrt(2), 0, 0, 0]],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=14)

        # mode_basis="hermitian_basis"
        actual = lindbladian.calc_j_part(mode_basis="hermitian_basis")
        expected = np.array(
            [[0, 0, 0, np.sqrt(2)], [0, 0, 0, 0], [0, 0, 0, 0], [np.sqrt(2), 0, 0, 0]],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=14)

        # mode_basis="comp_basis"
        actual = lindbladian.calc_j_part(mode_basis="comp_basis")
        expected = (
            np.array(
                [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -2]],
                dtype=np.complex128,
            )
            / np.sqrt(2)
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_k_part(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        # k=I
        k_mat = np.eye(3, dtype=np.complex128)
        lindbladian = el.generate_effective_lindbladian_from_k(c_sys, k_mat)

        # mode_basis=default("hermitian_basis")
        actual = lindbladian.calc_k_part()
        expected = np.array(
            [[3 / 2, 0, 0, 0], [0, -1 / 2, 0, 0], [0, 0, -1 / 2, 0], [0, 0, 0, -1 / 2]],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=14)

        # mode_basis="hermitian_basis"
        actual = lindbladian.calc_k_part(mode_basis="hermitian_basis")
        expected = np.array(
            [[3 / 2, 0, 0, 0], [0, -1 / 2, 0, 0], [0, 0, -1 / 2, 0], [0, 0, 0, -1 / 2]],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=14)

        # mode_basis="comp_basis"
        actual = lindbladian.calc_k_part(mode_basis="comp_basis")
        expected = np.array(
            [[1 / 2, 0, 0, 1], [0, -1 / 2, 0, 0], [0, 0, -1 / 2, 0], [1, 0, 0, 1 / 2]],
            dtype=np.complex128,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_to_var(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hs = np.zeros((4, 4), dtype=np.float64)

        # on_para_eq_constraint=True
        lindbladian = EffectiveLindbladian(c_sys, hs, on_para_eq_constraint=True)
        actual = lindbladian.to_var()
        assert len(actual) == 12

        # on_para_eq_constraint=False
        lindbladian = EffectiveLindbladian(c_sys, hs, on_para_eq_constraint=False)
        actual = lindbladian.to_var()
        assert len(actual) == 16

    def test_generate_origin_obj(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hs = np.zeros((4, 4), dtype=np.float64)

        lindbladian = EffectiveLindbladian(c_sys, hs, on_para_eq_constraint=True)

        # Act
        actual = lindbladian.generate_origin_obj()

        # Assert HS
        min = sys.float_info.min_exp
        expected = np.array(
            [[0, 0, 0, 0], [0, min, 0, 0], [0, 0, min, 0], [0, 0, 0, min]],
            dtype=np.float64,
        )
        assert type(actual) == EffectiveLindbladian
        npt.assert_almost_equal(actual.hs, expected, decimal=15)

        # Assert e^L = diag(1,0,...0)
        expected_exp = np.diag([1, 0, 0, 0])
        npt.assert_almost_equal(expm(actual.hs), expected_exp, decimal=13)

    def test_calc_gradient(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # Case 1: on_para_eq_constraint=True
        # Arrange
        hs = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
        )
        init_is_physicality_required = False
        init_is_estimation_object = False
        init_on_para_eq_constraint = True
        init_on_algo_eq_constraint = False
        init_on_algo_ineq_constraint = True
        init_eps_proj_physical = 10 ** (-5)

        lindbladian = EffectiveLindbladian(
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
        actual = lindbladian.calc_gradient(var_index=1)

        # Assert
        expected = np.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        assert type(actual) == EffectiveLindbladian
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
        lindbladian = EffectiveLindbladian(
            c_sys=c_sys,
            hs=hs,
            is_physicality_required=init_is_physicality_required,
            is_estimation_object=init_is_estimation_object,
            on_para_eq_constraint=init_on_para_eq_constraint,
            on_algo_eq_constraint=init_on_algo_eq_constraint,
            on_algo_ineq_constraint=init_on_algo_ineq_constraint,
            eps_proj_physical=init_eps_proj_physical,
        )

        actual = lindbladian.calc_gradient(var_index=1)
        expected = np.array(
            [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        assert type(actual) == EffectiveLindbladian
        npt.assert_almost_equal(actual.hs, expected, decimal=15)
        assert actual.composite_system is c_sys
        assert actual.is_physicality_required is False
        assert actual.is_estimation_object is init_is_estimation_object
        assert actual.on_para_eq_constraint is init_on_para_eq_constraint
        assert actual.on_algo_eq_constraint is init_on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is init_on_algo_ineq_constraint
        assert actual.eps_proj_physical is init_eps_proj_physical

    def test_calc_proj_eq_constraint(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        k_mat = -np.eye(3)
        lindbladian = el.generate_effective_lindbladian_from_k(
            c_sys, k_mat, is_physicality_required=False
        )

        actual = lindbladian.calc_proj_eq_constraint()
        expected = np.array(
            [[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]], dtype=np.float64
        )
        npt.assert_almost_equal(actual.hs, expected, decimal=15)

    def test_calc_proj_ineq_constraint(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # h=Z/sqrt(2), k=diag(1,1,-1)
        h_mat = np.array([[1, 0], [0, -1]], dtype=np.complex128) / np.sqrt(2)
        k_mat = np.diag([1, 1, -1])
        lindbladian = el.generate_effective_lindbladian_from_hk(
            c_sys, h_mat, k_mat, is_physicality_required=False
        )
        actual = lindbladian.calc_proj_ineq_constraint()
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, -np.sqrt(2), 0],
                [0, np.sqrt(2), 0, 0],
                [0, 0, 0, -1],
            ],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual.hs, expected, decimal=15)

    def test_add_vec(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        hs1 = np.diag([0, 1, 2, 3]).real.astype(np.float64)
        hs2 = np.diag([0, 10, 20, 30]).real.astype(np.float64)
        lindbladian1 = EffectiveLindbladian(c_sys, hs1, is_physicality_required=False)
        lindbladian2 = EffectiveLindbladian(c_sys, hs2, is_physicality_required=False)
        actual = lindbladian1 + lindbladian2
        assert type(actual) == EffectiveLindbladian
        expected = np.diag([0, 11, 22, 33]).real.astype(np.float64)
        npt.assert_almost_equal(actual.hs, expected, decimal=15)

    def test_sub_vec(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        hs1 = np.diag([0, 1, 2, 3]).real.astype(np.float64)
        hs2 = np.diag([0, -10, -20, -30]).real.astype(np.float64)
        lindbladian1 = EffectiveLindbladian(c_sys, hs1, is_physicality_required=False)
        lindbladian2 = EffectiveLindbladian(c_sys, hs2, is_physicality_required=False)
        actual = lindbladian1 - lindbladian2
        assert type(actual) == EffectiveLindbladian
        expected = np.diag([0, 11, 22, 33]).real.astype(np.float64)
        npt.assert_almost_equal(actual.hs, expected, decimal=15)

    def test_mul_vec(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        hs = np.diag([0, 1, 2, 3]).real.astype(np.float64)
        lindbladian = EffectiveLindbladian(c_sys, hs, is_physicality_required=False)
        actual = 3 * lindbladian
        assert type(actual) == EffectiveLindbladian
        expected = np.diag([0, 3, 6, 9]).real.astype(np.float64)
        npt.assert_almost_equal(actual.hs, expected, decimal=15)

    def test_truediv_vec(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        hs = np.diag([0, 1, 2, 3]).real.astype(np.float64)
        lindbladian = EffectiveLindbladian(c_sys, hs, is_physicality_required=False)
        actual = lindbladian / 3
        assert type(actual) == EffectiveLindbladian
        expected = np.diag([0, 1 / 3, 2 / 3, 3 / 3]).real.astype(np.float64)
        npt.assert_almost_equal(actual.hs, expected, decimal=15)

    def test_is_tp(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # hs=0
        hs = np.zeros((4, 4))
        actual = EffectiveLindbladian(c_sys, hs)
        assert actual.is_tp() == True

        # hs=I
        hs = np.eye(4)
        actual = EffectiveLindbladian(c_sys, hs, is_physicality_required=False)
        assert actual.is_tp() == False

    def test_is_cp(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # k=I
        k_mat = np.eye(3)
        actual = el.generate_effective_lindbladian_from_k(c_sys, k_mat)
        assert actual.is_cp() == True

        # k=-I
        k_mat = -np.eye(3)
        actual = el.generate_effective_lindbladian_from_k(
            c_sys, k_mat, is_physicality_required=False
        )
        assert actual.is_cp() == False

    def test_to_kraus_matrices(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        hs = np.array(
            [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        lindbladian = EffectiveLindbladian(c_sys, hs, is_physicality_required=False)
        actual = lindbladian.to_kraus_matrices()

        expected = np.array([[1 / 2, 1 / 2], [1 / 2, 1 / 2]], dtype=np.complex128)
        assert len(actual) == 1
        npt.assert_almost_equal(actual[0][0], np.sqrt(2), decimal=15)
        npt.assert_almost_equal(actual[0][1], expected, decimal=15)

    def test_to_gate(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hs = np.zeros((4, 4), dtype=np.float64)
        lindbladian = EffectiveLindbladian(c_sys, hs)

        # Act
        actual = lindbladian.to_gate()

        # Assert
        expected = np.eye(4)
        npt.assert_almost_equal(actual.hs, expected, decimal=15)
        assert actual.is_physicality_required == lindbladian.is_physicality_required
        assert actual.is_estimation_object == lindbladian.is_estimation_object
        assert actual.on_para_eq_constraint == lindbladian.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == lindbladian.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == lindbladian.on_algo_ineq_constraint
        assert actual.mode_proj_order == lindbladian.mode_proj_order
        assert actual.eps_proj_physical == lindbladian.eps_proj_physical


def test_calc_h_part_from_h_mat():
    # basis is normalized Pauli basis
    # h=0
    h_mat = np.zeros((2, 2), dtype=np.complex128)
    actual = el._calc_h_part_from_h_mat(h_mat)
    expected = np.zeros((4, 4), dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # h=I/sqrt(2)
    h_mat = np.eye(2, dtype=np.complex128) / np.sqrt(2)
    actual = el._calc_h_part_from_h_mat(h_mat)
    expected = np.zeros((4, 4), dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # h=X/sqrt(2)
    h_mat = np.array([[0, 1], [1, 0]], dtype=np.complex128) / np.sqrt(2)
    actual = el._calc_h_part_from_h_mat(h_mat)
    expected = (
        np.array(
            [[0, 1j, -1j, 0], [1j, 0, 0, -1j], [-1j, 0, 0, 1j], [0, -1j, 1j, 0]],
            dtype=np.complex128,
        )
        / np.sqrt(2)
    )
    npt.assert_almost_equal(actual, expected, decimal=15)

    # h=Y/sqrt(2)
    h_mat = np.array([[0, -1j], [1j, 0]], dtype=np.complex128) / np.sqrt(2)
    actual = el._calc_h_part_from_h_mat(h_mat)
    expected = (
        np.array(
            [[0, -1, -1, 0], [1, 0, 0, -1], [1, 0, 0, -1], [0, 1, 1, 0]],
            dtype=np.complex128,
        )
        / np.sqrt(2)
    )
    npt.assert_almost_equal(actual, expected, decimal=15)

    # h=Z/sqrt(2)
    h_mat = np.array([[1, 0], [0, -1]], dtype=np.complex128) / np.sqrt(2)
    actual = el._calc_h_part_from_h_mat(h_mat)
    expected = (
        np.array(
            [[0, 0, 0, 0], [0, -2j, 0, 0], [0, 0, 2j, 0], [0, 0, 0, 0]],
            dtype=np.complex128,
        )
        / np.sqrt(2)
    )
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_calc_j_part_from_j_mat():
    # basis is normalized Pauli basis
    # j=0
    j_mat = np.zeros((2, 2), dtype=np.complex128)
    actual = el._calc_j_part_from_j_mat(j_mat)
    expected = np.zeros((4, 4), dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # j=I/sqrt(2)
    j_mat = np.eye(2, dtype=np.complex128) / np.sqrt(2)
    actual = el._calc_j_part_from_j_mat(j_mat)
    expected = 2 * np.eye(4, dtype=np.float64) / np.sqrt(2)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # j=X/sqrt(2)
    j_mat = np.array([[0, 1], [1, 0]], dtype=np.complex128) / np.sqrt(2)
    actual = el._calc_j_part_from_j_mat(j_mat)
    expected = (
        np.array(
            [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]],
            dtype=np.complex128,
        )
        / np.sqrt(2)
    )
    npt.assert_almost_equal(actual, expected, decimal=15)

    # j=Y/sqrt(2)
    j_mat = np.array([[0, -1j], [1j, 0]], dtype=np.complex128) / np.sqrt(2)
    actual = el._calc_j_part_from_j_mat(j_mat)
    expected = (
        np.array(
            [[0, 1j, -1j, 0], [-1j, 0, 0, -1j], [1j, 0, 0, 1j], [0, 1j, -1j, 0]],
            dtype=np.complex128,
        )
        / np.sqrt(2)
    )
    npt.assert_almost_equal(actual, expected, decimal=15)

    # j=Z/sqrt(2)
    j_mat = np.array([[1, 0], [0, -1]], dtype=np.complex128) / np.sqrt(2)
    actual = el._calc_j_part_from_j_mat(j_mat)
    expected = (
        np.array(
            [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -2]],
            dtype=np.complex128,
        )
        / np.sqrt(2)
    )
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_calc_k_part_from_k_mat():
    # basis is normalized Pauli basis
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # k=0
    k_mat = np.zeros((3, 3), dtype=np.complex128)
    actual = el._calc_k_part_from_k_mat(k_mat, c_sys)
    expected = np.zeros((4, 4), dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # k=I
    k_mat = np.eye(3, dtype=np.complex128)
    actual = el._calc_k_part_from_k_mat(k_mat, c_sys)
    expected = np.array(
        [[1 / 2, 0, 0, 1], [0, -1 / 2, 0, 0], [0, 0, -1 / 2, 0], [1, 0, 0, 1 / 2]],
        dtype=np.complex128,
    )
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_generate_effective_lindbladian_from_hjk():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # h=0, j=0, k=0
    h_mat = np.zeros((2, 2), dtype=np.complex128)
    j_mat = np.zeros((2, 2), dtype=np.complex128)
    k_mat = np.zeros((3, 3), dtype=np.complex128)
    actual = el.generate_effective_lindbladian_from_hjk(c_sys, h_mat, j_mat, k_mat)
    expected = np.zeros((4, 4), dtype=np.float64)
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # h=I/sqrt(2), j=0, k=0
    h_mat = np.eye(2, dtype=np.complex128) / np.sqrt(2)
    j_mat = np.zeros((2, 2), dtype=np.complex128)
    k_mat = np.zeros((3, 3), dtype=np.complex128)
    actual = el.generate_effective_lindbladian_from_hjk(c_sys, h_mat, j_mat, k_mat)
    expected = np.zeros((4, 4), dtype=np.float64)
    npt.assert_almost_equal(actual.hs, expected, decimal=15)


def test_generate_effective_lindbladian_from_h():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # h=0
    h_mat = np.zeros((2, 2), dtype=np.complex128)
    actual = el.generate_effective_lindbladian_from_h(c_sys, h_mat)
    expected = np.zeros((4, 4), dtype=np.float64)
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # h=I/sqrt(2)
    h_mat = np.eye(2, dtype=np.complex128) / np.sqrt(2)
    actual = el.generate_effective_lindbladian_from_h(c_sys, h_mat)
    expected = np.zeros((4, 4), dtype=np.float64)
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # h=Z/sqrt(2)
    h_mat = np.array([[1, 0], [0, -1]], dtype=np.complex128) / np.sqrt(2)
    actual = el.generate_effective_lindbladian_from_h(c_sys, h_mat)
    expected = (
        np.array(
            [[0, 0, 0, 0], [0, 0, -2, 0], [0, 2, 0, 0], [0, 0, 0, 0]],
            dtype=np.float64,
        )
        / np.sqrt(2)
    )
    npt.assert_almost_equal(actual.hs, expected, decimal=15)


def test_generate_effective_lindbladian_from_hk():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # h=I/sqrt(2), k=0
    h_mat = np.eye(2, dtype=np.complex128) / np.sqrt(2)
    k_mat = np.zeros((3, 3), dtype=np.complex128)
    actual = el.generate_effective_lindbladian_from_hk(c_sys, h_mat, k_mat)
    expected = np.zeros((4, 4), dtype=np.float64)
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # h=I/sqrt(2), k=I
    h_mat = np.eye(2, dtype=np.complex128) / np.sqrt(2)
    k_mat = np.eye(3, dtype=np.complex128)
    actual = el.generate_effective_lindbladian_from_hk(c_sys, h_mat, k_mat)
    expected = np.array(
        [[0, 0, 0, 0], [0, -2, 0, 0], [0, 0, -2, 0], [0, 0, 0, -2]],
        dtype=np.float64,
    )
    npt.assert_almost_equal(actual.hs, expected, decimal=15)


def test_generate_effective_lindbladian_from_k():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # k=0
    k_mat = np.zeros((3, 3), dtype=np.complex128)
    actual = el.generate_effective_lindbladian_from_k(c_sys, k_mat)
    expected = np.zeros((4, 4), dtype=np.float64)
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # k=I
    k_mat = np.eye(3, dtype=np.complex128)
    actual = el.generate_effective_lindbladian_from_k(c_sys, k_mat)
    expected = np.array(
        [[0, 0, 0, 0], [0, -2, 0, 0], [0, 0, -2, 0], [0, 0, 0, -2]],
        dtype=np.float64,
    )
    npt.assert_almost_equal(actual.hs, expected, decimal=15)


def test_generate_j_part_cb_from_jump_operators():
    # Arrange
    c0 = np.array([[0, 1], [0, 0]])
    c1 = np.array([[0, 0], [1, 0]])
    c2 = np.array([[1, 0], [0, -1]])
    jump_operators = [c0, c1, c2]

    # Act
    actual = el.generate_j_part_cb_from_jump_operators(jump_operators)

    # Assert
    expected = (
        np.array(
            [[2, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, -2]],
            dtype=np.complex128,
        )
        * (-1 / 2)
    )
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_generate_j_part_gb_from_jump_operators():
    # Arrange
    basis = matrix_basis.get_normalized_pauli_basis()
    c0 = np.array([[0, 1], [0, 0]])
    c1 = np.array([[0, 0], [1, 0]])
    c2 = np.array([[1, 0], [0, -1]])
    jump_operators = [c0, c1, c2]

    # Act
    actual = el.generate_j_part_gb_from_jump_operators(jump_operators, basis)

    # Assert
    expected = np.array(
        [[0, -1, 0, -1], [-1, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 0]],
        dtype=np.float64,
    )
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_generate_k_part_cb_from_jump_operators():
    # Arrange
    c0 = np.array([[0, 1], [0, 0]])
    c1 = np.array([[0, 0], [1, 0]])
    c2 = np.array([[1, 0], [0, -1]])
    jump_operators = [c0, c1, c2]

    # Act
    actual = el.generate_k_part_cb_from_jump_operators(jump_operators)

    # Assert
    expected = np.array(
        [[1, 0, 0, 1], [0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 1]],
        dtype=np.complex128,
    )
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_generate_k_part_gb_from_jump_operators():
    # Arrange
    basis = matrix_basis.get_normalized_pauli_basis()
    c0 = np.array([[0, 1], [0, 0]])
    c1 = np.array([[0, 0], [1, 0]])
    c2 = np.array([[1, 0], [0, -1]])
    jump_operators = [c0, c1, c2]

    # Act
    actual = el.generate_k_part_gb_from_jump_operators(jump_operators, basis)

    # Assert
    expected = np.array(
        [[2, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 0]],
        dtype=np.float64,
    )
    npt.assert_almost_equal(actual, expected, decimal=15)
