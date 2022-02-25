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
