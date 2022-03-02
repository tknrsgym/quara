import numpy as np
import numpy.testing as npt
import pytest

from cvxpy.expressions.variable import Variable as CvxpyVariable
from cvxpy.expressions.expression import Expression as CvxpyExpression

from quara.interface.cvxpy.qtomography.standard.estimator import (
    CvxpyLossMinimizationEstimator,
    CvxpyLossMinimizationEstimationResult,
)

from quara.interface.cvxpy.qtomography.standard.minimization_algorithm import (
    CvxpyMinimizationAlgorithm,
    CvxpyMinimizationAlgorithmOption,
    CvxpyMinimizationResult,
)
from quara.interface.cvxpy.qtomography.standard.loss_function import (
    CvxpyRelativeEntropy,
    # CvxpyUniformSquaredError,
    # CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm,
    CvxpyLossFunctionOption,
)
from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.state_typical import generate_state_from_name
from quara.objects.povm_typical import generate_povm_from_name
from quara.protocol.qtomography.standard.standard_qst import StandardQst


def get_test_data_qst(on_para_eq_constraint=True):
    c_sys = generate_composite_system("qubit", 1)

    povm_x = generate_povm_from_name("x", c_sys)
    povm_y = generate_povm_from_name("y", c_sys)
    povm_z = generate_povm_from_name("z", c_sys)
    povms = [povm_x, povm_y, povm_z]

    qst = StandardQst(povms, on_para_eq_constraint=on_para_eq_constraint, seed_data=7)

    return qst, c_sys


class TestCvxpyLossMinimizationEstimationResult:
    def test_access_estimated_loss_sequence(self):
        estimated_var_sequence = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
        ]
        computation_times = [1.0, 2.0]
        c_sys = generate_composite_system("qubit", 1)
        template_qoperation = generate_state_from_name(c_sys, "z0")

        # estimated_loss_sequence = default(None)
        actual = CvxpyLossMinimizationEstimationResult(
            estimated_var_sequence, computation_times, template_qoperation
        )
        assert actual.estimated_loss_sequence == None

        # estimated_loss_sequence = [10.0, 20.0]
        estimated_loss_sequence = [10.0, 20.0]
        actual = CvxpyLossMinimizationEstimationResult(
            estimated_var_sequence,
            computation_times,
            template_qoperation,
            estimated_loss_sequence=estimated_loss_sequence,
        )
        assert actual.estimated_loss_sequence == estimated_loss_sequence

        # Test that "estimated_loss_sequence" cannot be updated
        with pytest.raises(AttributeError):
            actual.estimated_loss_sequence = estimated_loss_sequence


class TestCvxpyLossMinimizationEstimator:
    def test_calc_estimate(self):
        # Arrange
        sqt, c_sys = get_test_data_qst()
        true_object = generate_state_from_name(c_sys, "z0")
        prob_dists = sqt.calc_prob_dists(true_object)
        empi_dists = [(10, prob_dist) for prob_dist in prob_dists]
        loss = CvxpyRelativeEntropy()
        loss_option = CvxpyLossFunctionOption()
        algo = CvxpyMinimizationAlgorithm()
        algo_option = CvxpyMinimizationAlgorithmOption(
            name_solver="scs", mode_constraint="physical"
        )
        estimator = CvxpyLossMinimizationEstimator()

        # Act
        actual = estimator.calc_estimate(
            sqt, empi_dists, loss, loss_option, algo, algo_option
        )

        # Assert
        assert type(actual.estimated_var_sequence) == list
        assert len(actual.estimated_var_sequence) == 1
        assert type(actual.estimated_loss_sequence) == list
        assert len(actual.estimated_loss_sequence) == 1
        expected = np.array([0, 0, 1 / np.sqrt(2)])
        npt.assert_almost_equal(actual.estimated_var_sequence[0], expected, decimal=9)

    def test_calc_estimate_sequence(self):
        # Arrange
        sqt, c_sys = get_test_data_qst()
        true_object0 = generate_state_from_name(c_sys, "z0")
        prob_dists0 = sqt.calc_prob_dists(true_object0)
        empi_dists0 = [(10, prob_dist) for prob_dist in prob_dists0]
        true_object1 = generate_state_from_name(c_sys, "x0")
        prob_dists1 = sqt.calc_prob_dists(true_object1)
        empi_dists1 = [(10, prob_dist) for prob_dist in prob_dists1]
        loss = CvxpyRelativeEntropy()
        loss_option = CvxpyLossFunctionOption()
        algo = CvxpyMinimizationAlgorithm()
        algo_option = CvxpyMinimizationAlgorithmOption(
            name_solver="scs", mode_constraint="physical"
        )
        estimator = CvxpyLossMinimizationEstimator()

        # Act
        actual = estimator.calc_estimate_sequence(
            sqt, [empi_dists0, empi_dists1], loss, loss_option, algo, algo_option
        )

        # Assert
        assert type(actual.estimated_var_sequence) == list
        assert len(actual.estimated_var_sequence) == 2
        assert type(actual.estimated_loss_sequence) == list
        assert len(actual.estimated_loss_sequence) == 2
        expected = np.array([0, 0, 1 / np.sqrt(2)])
        npt.assert_almost_equal(actual.estimated_var_sequence[0], expected, decimal=9)
        expected = np.array([1 / np.sqrt(2), 0, 0])
        npt.assert_almost_equal(actual.estimated_var_sequence[1], expected, decimal=9)
