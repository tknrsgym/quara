import numpy as np
import numpy.testing as npt
import pytest

from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
from quara.protocol.qtomography.standard.projected_linear_estimator import (
    ProjectedLinearEstimator,
)
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects import povm
from quara.objects.state import State
from quara.simulation import consistency_check


def test_calc_mse_of_true_estimated():
    # Arrange
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    povm_x = povm.get_x_povm(c_sys)
    povm_y = povm.get_y_povm(c_sys)
    povm_z = povm.get_z_povm(c_sys)
    povms = [povm_x, povm_y, povm_z]

    vec = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
    true_object = State(c_sys, vec, is_physicality_required=False)

    # Case 1
    qst = StandardQst(povms, on_para_eq_constraint=True)
    estimator = LinearEstimator()

    # Act
    actual_mse, _ = consistency_check.calc_mse_of_true_estimated(
        true_object=true_object, qtomography=qst, estimator=estimator
    )
    # Assert
    npt.assert_almost_equal(actual_mse, 0, decimal=16)

    # Case 2
    # Array
    qst = StandardQst(povms, on_para_eq_constraint=False)
    estimator = LinearEstimator()

    # Act
    actual_mse, _ = consistency_check.calc_mse_of_true_estimated(
        true_object=true_object, qtomography=qst, estimator=estimator
    )
    # Assert
    npt.assert_almost_equal(actual_mse, 0, decimal=16)

    # Case 3
    # Array
    qst = StandardQst(povms, on_para_eq_constraint=True)
    estimator = ProjectedLinearEstimator()

    # Act
    actual_mse, _ = consistency_check.calc_mse_of_true_estimated(
        true_object=true_object, qtomography=qst, estimator=estimator
    )
    # Assert
    npt.assert_almost_equal(actual_mse, 0, decimal=16)

    # Case 4
    # Arrange
    qst = StandardQst(povms, on_para_eq_constraint=False)
    estimator = ProjectedLinearEstimator()

    # Act
    actual_mse, _ = consistency_check.calc_mse_of_true_estimated(
        true_object=true_object, qtomography=qst, estimator=estimator
    )
    # Assert
    npt.assert_almost_equal(actual_mse, 0, decimal=16)
