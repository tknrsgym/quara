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
from quara.data_analysis import physicality_violation_check as pvc


def test_get_sum_of_eigenvalues_violation():
    # Arrange
    source_vals = [[0.6, 0.5, 1, -0.5], [1, 0.5, 0.4, 0.1]]

    # Act
    actual_less_list, actual_greater_list = pvc.get_sum_of_eigenvalues_violation(
        source_vals, expected_values=(0, 2)
    )

    # Assert
    expected_less_list = [-0.5]
    expected_greater_list = [2.1]
    assert actual_less_list == expected_less_list
    assert actual_greater_list == expected_greater_list
