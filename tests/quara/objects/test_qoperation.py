import copy

import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate_typical import generate_gate_from_gate_name
from quara.objects.matrix_basis import (
    get_normalized_pauli_basis,
    get_normalized_gell_mann_basis,
)
from quara.objects.mprocess_typical import generate_mprocess_from_name
from quara.objects.povm_typical import generate_povm_from_name
from quara.objects.qoperation import QOperation
from quara.objects.state_typical import generate_state_from_name
from quara.settings import Settings


class TestSetQOperation:
    def test_init(self):
        # Arrange
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # Act
        actual = QOperation(c_sys=c_sys)

        # Assert
        assert actual.composite_system == c_sys
        assert actual.is_physicality_required == True
        assert actual.is_estimation_object == True
        assert actual.on_para_eq_constraint == True
        assert actual.on_algo_eq_constraint == True
        assert actual.on_algo_ineq_constraint == True
        assert actual.eps_proj_physical == Settings.get_atol() / 10.0

    def test_init_error(self):
        # Arrange
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # eps_proj_physical is a negative number
        with pytest.raises(ValueError):
            QOperation(c_sys=c_sys, eps_proj_physical=-(10 ** (-4)))

    def test_access_is_physicality_required(self):
        # Arrange
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        is_physicality_required = False

        # Act
        actual = QOperation(
            c_sys=c_sys, is_physicality_required=is_physicality_required
        )

        # Assert
        assert actual.is_physicality_required == is_physicality_required

        # Test that the property cannot be updated
        with pytest.raises(AttributeError):
            actual.is_physicality_required = True

    def test_access_is_estimation_object(self):
        # Arrange
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        is_estimation_object = False

        # Act
        actual = QOperation(c_sys=c_sys, is_estimation_object=is_estimation_object)

        # Assert
        assert actual.is_estimation_object == is_estimation_object

        # Test that the property cannot be updated
        with pytest.raises(AttributeError):
            actual.is_estimation_object = True

    def test_access_on_para_eq_constraint(self):
        # Arrange
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        on_para_eq_constraint = False

        # Act
        actual = QOperation(c_sys=c_sys, on_para_eq_constraint=on_para_eq_constraint)

        # Assert
        assert actual.on_para_eq_constraint == on_para_eq_constraint

        # Test that the property cannot be updated
        with pytest.raises(AttributeError):
            actual.on_para_eq_constraint = True

    def test_access_on_algo_eq_constraint(self):
        # Arrange
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        on_algo_eq_constraint = False

        # Act
        actual = QOperation(c_sys=c_sys, on_algo_eq_constraint=on_algo_eq_constraint)

        # Assert
        assert actual.on_algo_eq_constraint == on_algo_eq_constraint

        # Test that the property cannot be updated
        with pytest.raises(AttributeError):
            actual.on_algo_eq_constraint = True

    def test_access_on_algo_ineq_constraint(self):
        # Arrange
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        on_algo_ineq_constraint = False

        # Act
        actual = QOperation(
            c_sys=c_sys, on_algo_ineq_constraint=on_algo_ineq_constraint
        )

        # Assert
        assert actual.on_algo_ineq_constraint == on_algo_ineq_constraint

        # Test that the property cannot be updated
        with pytest.raises(AttributeError):
            actual.on_algo_ineq_constraint = True

    def test_access_eps_proj_physical(self):
        # Arrange
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        eps_proj_physical = 10 ** (-3)

        # Act
        actual = QOperation(c_sys=c_sys, eps_proj_physical=eps_proj_physical)

        # Assert
        assert actual.eps_proj_physical == eps_proj_physical

        # Test that the property cannot be updated
        with pytest.raises(AttributeError):
            actual.eps_proj_physical = 10 ** (-2)

    def test_access_mode_proj_order(self):
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # default
        actual = QOperation(c_sys=c_sys)
        assert actual.mode_proj_order == "eq_ineq"

        # eq_ineq
        actual = QOperation(c_sys=c_sys, mode_proj_order="eq_ineq")
        assert actual.mode_proj_order == "eq_ineq"

        # ineq_eq
        actual = QOperation(c_sys=c_sys, mode_proj_order="ineq_eq")
        assert actual.mode_proj_order == "ineq_eq"

        # unsupported value
        with pytest.raises(ValueError):
            QOperation(c_sys=c_sys, mode_proj_order="unsupported")

        # updated mode_proj_order
        actual = QOperation(c_sys=c_sys)
        actual.set_mode_proj_order("ineq_eq")
        assert actual.mode_proj_order == "ineq_eq"

    def test_permutation_matrix_from_qutrits_to_qubits(self):
        # num_qutrits = 1
        num_qutrits = 1
        diag_mat = np.diag(list(range(4 ** num_qutrits)))
        actual = QOperation._permutation_matrix_from_qutrits_to_qubits(num_qutrits)
        actual_diag = np.diag(actual @ diag_mat @ actual.T)
        expected = [0, 1, 2, 3]
        npt.assert_almost_equal(actual_diag, expected, decimal=15)

        # num_qutrits = 2
        num_qutrits = 2
        actual = QOperation._permutation_matrix_from_qutrits_to_qubits(num_qutrits)
        diag_mat = np.diag(list(range(4 ** num_qutrits)))
        actual_diag = np.diag(actual @ diag_mat @ actual.T)
        expected = [0, 1, 2, 9, 3, 4, 5, 10, 6, 7, 8, 11, 12, 13, 14, 15]
        npt.assert_almost_equal(actual_diag, expected, decimal=15)

    def test_embed_qoperation_from_qutrits_to_qubits_error(self):
        # Arrange
        e_sys0 = ElementalSystem(0, get_normalized_gell_mann_basis())
        c_sys = CompositeSystem([e_sys0])
        e_sys0_qubits = ElementalSystem(0, get_normalized_pauli_basis())
        qope_qutrit = generate_state_from_name(c_sys, "01z0")

        # Act
        with pytest.raises(ValueError):
            QOperation.embed_qoperation_from_qutrits_to_qubits(
                qope_qutrit, [e_sys0_qubits]
            )

    def test_embed_qoperation_from_qutrits_to_qubits_state(self):
        ### case: num_qutrits = 1
        # Arrange
        e_sys0 = ElementalSystem(0, get_normalized_gell_mann_basis())
        c_sys = CompositeSystem([e_sys0])
        e_sys0_qubits = ElementalSystem(0, get_normalized_pauli_basis())
        e_sys1_qubits = ElementalSystem(1, get_normalized_pauli_basis())
        qope_qutrit = generate_state_from_name(c_sys, "01z0")

        # Act
        qope_qubit = QOperation.embed_qoperation_from_qutrits_to_qubits(
            qope_qutrit, [e_sys0_qubits, e_sys1_qubits]
        )

        # Assert
        actual = qope_qubit.to_density_matrix_with_sparsity()
        expected = np.diag([1, 0, 0, 0])
        npt.assert_almost_equal(actual, expected, decimal=15)

        ### case: num_qutrits = 2
        # Arrange
        e_sys0 = ElementalSystem(0, get_normalized_gell_mann_basis())
        e_sys1 = ElementalSystem(1, get_normalized_gell_mann_basis())
        c_sys = CompositeSystem([e_sys0, e_sys1])
        e_sys0_qubits = ElementalSystem(0, get_normalized_pauli_basis())
        e_sys1_qubits = ElementalSystem(1, get_normalized_pauli_basis())
        e_sys2_qubits = ElementalSystem(2, get_normalized_pauli_basis())
        e_sys3_qubits = ElementalSystem(3, get_normalized_pauli_basis())
        qope_qutrit = generate_state_from_name(c_sys, "01z0_01z0")

        # Act
        qope_qubit = QOperation.embed_qoperation_from_qutrits_to_qubits(
            qope_qutrit, [e_sys0_qubits, e_sys1_qubits, e_sys2_qubits, e_sys3_qubits]
        )

        # Assert
        actual = qope_qubit.to_density_matrix_with_sparsity()
        expected = np.diag([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_embed_qoperation_from_qutrits_to_qubits_povm(self):
        ### case: num_qutrits = 1
        # Arrange
        e_sys0 = ElementalSystem(0, get_normalized_gell_mann_basis())
        c_sys = CompositeSystem([e_sys0])
        e_sys0_qubits = ElementalSystem(0, get_normalized_pauli_basis())
        e_sys1_qubits = ElementalSystem(1, get_normalized_pauli_basis())
        qope_qutrit = generate_povm_from_name("z3", c_sys)

        # Act
        qope_qubit = QOperation.embed_qoperation_from_qutrits_to_qubits(
            qope_qutrit, [e_sys0_qubits, e_sys1_qubits]
        )

        # Assert
        actual = qope_qubit.matrices_with_sparsity()
        expected = [
            np.diag([1, 0, 0, 1 / 3]),
            np.diag([0, 1, 0, 1 / 3]),
            np.diag([0, 0, 1, 1 / 3]),
        ]
        for a, e in zip(actual, expected):
            npt.assert_almost_equal(a, e, decimal=15)

        ### case: num_qutrits = 2
        # Arrange
        e_sys0 = ElementalSystem(0, get_normalized_gell_mann_basis())
        e_sys1 = ElementalSystem(1, get_normalized_gell_mann_basis())
        c_sys = CompositeSystem([e_sys0, e_sys1])
        e_sys0_qubits = ElementalSystem(0, get_normalized_pauli_basis())
        e_sys1_qubits = ElementalSystem(1, get_normalized_pauli_basis())
        e_sys2_qubits = ElementalSystem(2, get_normalized_pauli_basis())
        e_sys3_qubits = ElementalSystem(3, get_normalized_pauli_basis())
        qope_qutrit = generate_povm_from_name("z3_z3", c_sys)

        # Act
        qope_qubit = QOperation.embed_qoperation_from_qutrits_to_qubits(
            qope_qutrit, [e_sys0_qubits, e_sys1_qubits, e_sys2_qubits, e_sys3_qubits]
        )

        # Assert
        actual = qope_qubit.matrices_with_sparsity()
        expected = [
            np.diag(
                [
                    1,
                    0,
                    0,
                    1 / 9,
                    0,
                    0,
                    0,
                    1 / 9,
                    0,
                    0,
                    0,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                ]
            ),
            np.diag(
                [
                    0,
                    1,
                    0,
                    1 / 9,
                    0,
                    0,
                    0,
                    1 / 9,
                    0,
                    0,
                    0,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                ]
            ),
            np.diag(
                [
                    0,
                    0,
                    1,
                    1 / 9,
                    0,
                    0,
                    0,
                    1 / 9,
                    0,
                    0,
                    0,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                ]
            ),
            np.diag(
                [
                    0,
                    0,
                    0,
                    1 / 9,
                    1,
                    0,
                    0,
                    1 / 9,
                    0,
                    0,
                    0,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                ]
            ),
            np.diag(
                [
                    0,
                    0,
                    0,
                    1 / 9,
                    0,
                    1,
                    0,
                    1 / 9,
                    0,
                    0,
                    0,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                ]
            ),
            np.diag(
                [
                    0,
                    0,
                    0,
                    1 / 9,
                    0,
                    0,
                    1,
                    1 / 9,
                    0,
                    0,
                    0,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                ]
            ),
            np.diag(
                [
                    0,
                    0,
                    0,
                    1 / 9,
                    0,
                    0,
                    0,
                    1 / 9,
                    1,
                    0,
                    0,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                ]
            ),
            np.diag(
                [
                    0,
                    0,
                    0,
                    1 / 9,
                    0,
                    0,
                    0,
                    1 / 9,
                    0,
                    1,
                    0,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                ]
            ),
            np.diag(
                [
                    0,
                    0,
                    0,
                    1 / 9,
                    0,
                    0,
                    0,
                    1 / 9,
                    0,
                    0,
                    1,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                    1 / 9,
                ]
            ),
        ]
        for a, e in zip(actual, expected):
            npt.assert_almost_equal(a, e, decimal=15)

    def test_embed_qoperation_from_qutrits_to_qubits_gate(self):
        ### case: num_qutrits = 1
        # Arrange
        e_sys0 = ElementalSystem(0, get_normalized_gell_mann_basis())
        c_sys = CompositeSystem([e_sys0])
        e_sys0_qubits = ElementalSystem(0, get_normalized_pauli_basis())
        e_sys1_qubits = ElementalSystem(1, get_normalized_pauli_basis())
        qope_qutrit = generate_gate_from_gate_name("01z180", c_sys)

        # Act
        qope_qubit = QOperation.embed_qoperation_from_qutrits_to_qubits(
            qope_qutrit, [e_sys0_qubits, e_sys1_qubits]
        )

        # Assert
        actual = qope_qubit.to_kraus_matrices()
        expected = np.diag([1, -1, 1j, 1])
        npt.assert_almost_equal(actual[0], expected, decimal=15)

    def test_embed_qoperation_from_qutrits_to_qubits_mprocess(self):
        ### case: num_qutrits = 1
        # Arrange
        e_sys0 = ElementalSystem(0, get_normalized_gell_mann_basis())
        c_sys = CompositeSystem([e_sys0])
        e_sys0_qubits = ElementalSystem(0, get_normalized_pauli_basis())
        e_sys1_qubits = ElementalSystem(1, get_normalized_pauli_basis())
        qope_qutrit = generate_mprocess_from_name(c_sys, "z3-type1")

        # Act
        qope_qubit = QOperation.embed_qoperation_from_qutrits_to_qubits(
            qope_qutrit, [e_sys0_qubits, e_sys1_qubits]
        )

        # Assert
        expected = [
            np.diag([1, 0, 0, 1 / np.sqrt(3)]),
            np.diag([0, 1, 0, 1 / np.sqrt(3)]),
            np.diag([0, 0, 1, 1 / np.sqrt(3)]),
        ]
        for index, e in enumerate(expected):
            npt.assert_almost_equal(
                qope_qubit.to_kraus_matrices(index)[0], e, decimal=15
            )
