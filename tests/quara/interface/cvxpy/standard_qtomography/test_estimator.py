import numpy as np
import numpy.testing as npt
import pytest

from itertools import product

# quara
from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.tester_typical import (
    generate_tester_states,
    generate_tester_povms,
)
from quara.objects.qoperation_typical import generate_qoperation
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.standard_povmt import StandardPovmt
from quara.protocol.qtomography.standard.standard_qpt import StandardQpt

from quara.interface.cvxpy.standard_qtomography.loss_function import (
    CvxpyLossFunction,
    CvxpyLossFunctionOption,
    CvxpyRelativeEntropy,
    CvxpyUniformSquaredError,
    CvxpyApproximateRelativeEntropyWithoutZeroProbabilityTerm,
    CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm,
    CvxpyApproximateRelativeEntropyWithZeroProbabilityTermSquared,
)
from quara.protocol.qtomography.standard.preprocessing import (
    combine_nums_prob_dists,
)
from quara.interface.cvxpy.standard_qtomography.minimization_algorithm import (
    CvxpyMinimizationAlgorithm,
    CvxpyMinimizationAlgorithmOption,
)
from quara.interface.cvxpy.standard_qtomography.estimator import (
    CvxpyLossMinimizationEstimator,
)


@pytest.mark.cvxpy
@pytest.mark.mosek
@pytest.mark.onequbit
def test_estimator_consistency_cvxpy_mosek_1qubit_qst_case01():
    mode = "qubit"
    num = 1
    c_sys = generate_composite_system(mode=mode, num=num)

    # Testers
    names = ["x", "y", "z"]
    testers = generate_tester_povms(c_sys=c_sys, names=names)

    seed = 7896
    sqt = StandardQst(testers, on_para_eq_constraint=True, schedules="all", seed=seed)

    mode = "state"
    name = "z0"
    true = generate_qoperation(mode=mode, name=name, c_sys=c_sys)

    num_data = 1000
    empi_dists = sqt.generate_empi_dists(state=true, num_sum=num_data)

    prob_dists_true = sqt.calc_prob_dists(true)
    nums = [num_data] * sqt.num_schedules
    empi_dists = combine_nums_prob_dists(nums, prob_dists_true)

    loss = CvxpyRelativeEntropy()
    # loss = CvxpyUniformSquaredError()
    # loss = CvxpyApproximateRelativeEntropyWithoutZeroProbabilityTerm()
    # loss = CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm()
    # loss = CvxpyApproximateRelativeEntropyWithZeroProbabilityTermSquared()
    mode_form = "sum"
    # modes_form = ['sum', 'quadratic']
    loss_option = CvxpyLossFunctionOption(mode_form=mode_form)
    algo = CvxpyMinimizationAlgorithm()
    estimator = CvxpyLossMinimizationEstimator()

    mode_constraint = (
        "physical"  #    "physical_and_zero_probability_equation_satisfied"]
    )
    name_solver = "mosek"
    algo_option = CvxpyMinimizationAlgorithmOption(
        name_solver=name_solver, mode_constraint=mode_constraint
    )

    result = estimator.calc_estimate(
        qtomography=sqt,
        empi_dists=empi_dists,
        loss=loss,
        loss_option=loss_option,
        algo=algo,
        algo_option=algo_option,
        is_computation_time_required=True,
    )
    var_estimate = result.estimated_var
    # estimate = sqt.convert_var_to_qoperation(var_estimate)

    actual = var_estimate
    expected = true.to_var()

    decimal = 1e-8
    npt.assert_almost_equal(actual, expected, decimal=decimal)
