from typing import List

import numpy as np
import numpy.testing as npt
import pytest

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import Gate, get_h, get_i, get_x, get_cnot, get_swap, get_cz
from quara.objects.povm import (
    Povm,
    get_x_measurement,
    get_y_measurement,
    get_z_measurement,
    get_xx_measurement,
    get_xy_measurement,
    get_yy_measurement,
    get_zz_measurement,
)
from quara.objects.qoperation import QOperation
from quara.objects.state import State, get_x0_1q, get_y0_1q, get_z0_1q, get_bell_2q
from quara.objects import qoperations as qope


class TestSetQOperation:
    def test_init(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
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
        assert actual.eps_proj_physical == 10 ** (-4)

    def test_init_error(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # eps_proj_physical is a negative number
        with pytest.raises(ValueError):
            QOperation(c_sys=c_sys, eps_proj_physical=-(10 ** (-4)))
            """
            QOperation(
                c_sys=c_sys,
                is_physicality_required=is_physicality_required,
                is_estimation_object=is_estimation_object,
                on_para_eq_constraint=on_para_eq_constraint,
                on_algo_eq_constraint=on_algo_eq_constraint,
                on_algo_ineq_constraint=on_algo_ineq_constraint,
                eps_proj_physical=10 ** (-4),
            )
            """

    def test_access_is_physicality_required(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
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
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
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
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
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
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
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
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
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
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])

        eps_proj_physical = 10 ** (-3)

        # Act
        actual = QOperation(c_sys=c_sys, eps_proj_physical=eps_proj_physical)

        # Assert
        assert actual.eps_proj_physical == eps_proj_physical

        # Test that the property cannot be updated
        with pytest.raises(AttributeError):
            actual.eps_proj_physical = 10 ** (-2)
