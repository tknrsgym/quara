import itertools
from typing import List
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


def get_names_tester_states(mode_sys: str) -> List[str]:
    if mode_sys == "qubit":
        names = ["x0", "y0", "z0", "z1"]
    elif mode_sys == "qutrit":
        names = ["01z0", "12z0", "02z1", "01x0", "01y0", "12x0", "12y0", "02x0", "02y0"]
    else:
        raise ValueError(f"mode_sys is invalid!")
    return names


def get_names_tester_povms(mode_sys: str) -> List[str]:
    if mode_sys == "qubit":
        names = ["x", "y", "z"]
    elif mode_sys == "qutrit":
        names = ["01x3", "01y3", "z3", "12x3", "12y3", "02x3", "02y3"]
    else:
        raise ValueError(f"mode_sys is invalid!")
    return names


# State names
def get_names_1qubit_state_test() -> List[str]:
    l = ["z0", "a"]
    return l


def get_names_2qubit_state_test() -> List[str]:
    l = ["z0_z0"]
    return l


def get_names_3qubit_state_test() -> List[str]:
    l = ["z0_z0_z0"]
    return l


def get_names_1qutrit_state_test() -> List[str]:
    l = ["01z0"]
    return l


def get_names_2qutrit_state_test() -> List[str]:
    l = ["01z0_01z0"]
    return l


# POVM names
def get_names_1qubit_povm_test() -> List[str]:
    l = ["z"]
    return l


def get_names_2qubit_povm_test() -> List[str]:
    l = ["z_z"]
    return l


def get_names_3qubit_povm_test() -> List[str]:
    l = ["z_z_z"]
    return l


def get_names_1qutrit_povm_test() -> List[str]:
    l = ["z3"]
    return l


def get_names_2qutrit_povm_test() -> List[str]:
    l = ["z3_z3"]
    return l


# Gate names
def get_names_1qubit_gate_test() -> List[str]:
    l = ["identity", "x90", "hadamard"]
    return l


def get_names_2qubit_gate_test() -> List[str]:
    l = ["identity"]
    return l


def get_names_3qubit_gate_test() -> List[str]:
    l = ["identity"]
    return l


def get_names_1qutrit_gate_test() -> List[str]:
    l = ["identity"]
    return l


def get_names_2qutrit_gate_test() -> List[str]:
    l = ["identity"]
    return l


# Loss names


def get_names_cvxpy_loss_test() -> List[str]:
    l = ["uniform_squared_error"]
    l.append("relative_entropy")
    # l.append("approximate_relative_entropy_without_zero_probability_term")
    l.append("approximate_relative_entropy_with_zero_probability_term")
    l.append("approximate_relative_entropy_with_zero_probability_term_squared")
    return l


def get_names_cvxopt_loss_test() -> List[str]:
    l = ["uniform_squared_error"]
    # l.append("approximate_relative_entropy_without_zero_probability_term")
    l.append("approximate_relative_entropy_with_zero_probability_term")
    l.append("approximate_relative_entropy_with_zero_probability_term_squared")
    return l


def get_loss_from_name(name: str) -> CvxpyLossFunction:
    if name == "uniform_squared_error":
        loss = CvxpyUniformSquaredError()
    elif name == "relative_entropy":
        loss = CvxpyRelativeEntropy()
    elif name == "approximate_relative_entropy_without_zero_probability_term":
        loss = CvxpyApproximateRelativeEntropyWithoutZeroProbabilityTerm()
    elif name == "approximate_relative_entropy_with_zero_probability_term":
        loss = CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm()
    elif name == "approximate_relative_entropy_with_zero_probability_term_squared":
        loss = CvxpyApproximateRelativeEntropyWithZeroProbabilityTermSquared()

    return loss


def get_modes_loss_form() -> List[str]:
    l = ["sum", "quadratic"]
    return l


# mode_constraint
def get_modes_constraint() -> List[str]:
    l = ["physical", "physical_and_zero_probability_equation_satisfied"]
    return l


# test setting case class


def _test_estimator_consistency_cvxpy(
    mode_sys: str,
    num_sys: int,
    mode_tomo: str,
    name_true: str,
    name_loss: str,
    mode_loss_form: str,
    mode_constraint: str,
    name_solver: str,
    decimal: int,
):
    # CompositeSystem
    c_sys = generate_composite_system(mode=mode_sys, num=num_sys)

    # True object to be estimated
    ids = list(range(num_sys))
    true = generate_qoperation(mode=mode_tomo, name=name_true, c_sys=c_sys, ids=ids)

    # StandardQTomography
    if mode_tomo == "state":
        names_tester_povms = get_names_tester_povms(mode_sys)
        tester_povms = generate_tester_povms(c_sys=c_sys, names=names_tester_povms)
        sqt = StandardQst(
            povms=tester_povms, on_para_eq_constraint=True, schedules="all"
        )
    elif mode_tomo == "povm":
        names_tester_states = get_names_tester_states(mode_sys)
        tester_states = generate_tester_states(c_sys=c_sys, names=names_tester_states)
        sqt = StandardPovmt(
            states=tester_states,
            num_outcomes=true.num_outcomes,
            on_para_eq_constraint=True,
            schedules="all",
        )
    elif mode_tomo == "gate":
        names_tester_states = get_names_tester_states(mode_sys)
        tester_states = generate_tester_states(c_sys=c_sys, names=names_tester_states)
        names_tester_povms = get_names_tester_povms(mode_sys)
        tester_povms = generate_tester_povms(c_sys=c_sys, names=names_tester_povms)
        sqt = StandardQpt(
            states=tester_states,
            povms=tester_povms,
            on_para_eq_constraint=True,
            schedules="all",
        )
    # elif mode_tomo == "mprocess":
    else:
        raise ValueError(f"mode_tomo is invalid!")

    # Empirical Distributions calculated from true probability distributions
    num_data = 1
    prob_dists_true = sqt.calc_prob_dists(true)
    nums = [num_data] * sqt.num_schedules
    empi_dists = combine_nums_prob_dists(nums, prob_dists_true)

    if not (name_loss == "relative_entropy" and mode_loss_form != "sum"):
        loss = get_loss_from_name(name_loss)
        loss_option = CvxpyLossFunctionOption(mode_form=mode_loss_form)
        algo = CvxpyMinimizationAlgorithm()
        algo_option = CvxpyMinimizationAlgorithmOption(
            name_solver=name_solver, mode_constraint=mode_constraint
        )
        estimator = CvxpyLossMinimizationEstimator()

        # Estimation
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

        # Test
        actual = var_estimate
        expected = true.to_var()

        npt.assert_almost_equal(actual, expected, decimal=decimal)


# 1-qubit, MOSEK


@pytest.mark.cvxpy
@pytest.mark.mosek
@pytest.mark.onequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 1, "state", pro[0], pro[1], pro[2], pro[3], "mosek", 4)
        for pro in product(
            get_names_1qubit_state_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qubit_state_mosek(
    mode_sys: str,
    num_sys: int,
    mode_tomo: str,
    name_true: str,
    name_loss: str,
    mode_loss_form: str,
    mode_constraint: str,
    name_solver: str,
    decimal: int,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.mosek
@pytest.mark.onequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 1, "povm", pro[0], pro[1], pro[2], pro[3], "mosek", 4)
        for pro in product(
            get_names_1qubit_povm_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qubit_povm_mosek(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.mosek
@pytest.mark.onequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 1, "gate", pro[0], pro[1], pro[2], pro[3], "mosek", 4)
        for pro in product(
            get_names_1qubit_gate_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qubit_gate_mosek(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


# 2-qubit, MOSEK


@pytest.mark.cvxpy
@pytest.mark.mosek
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 2, "state", pro[0], pro[1], pro[2], pro[3], "mosek", 4)
        for pro in product(
            get_names_2qubit_state_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qubit_state_mosek(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.mosek
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 2, "povm", pro[0], pro[1], pro[2], pro[3], "mosek", 4)
        for pro in product(
            get_names_2qubit_povm_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qubit_povm_mosek(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.mosek
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 2, "gate", pro[0], pro[1], pro[2], pro[3], "mosek", 4)
        for pro in product(
            get_names_2qubit_gate_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qubit_gate_mosek(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


# 3-qubit, MOSEK


@pytest.mark.cvxpy
@pytest.mark.mosek
@pytest.mark.threequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 3, "state", pro[0], pro[1], pro[2], pro[3], "mosek", 4)
        for pro in product(
            get_names_3qubit_state_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_3qubit_state_mosek(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.mosek
@pytest.mark.threequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 3, "povm", pro[0], pro[1], pro[2], pro[3], "mosek", 4)
        for pro in product(
            get_names_3qubit_povm_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_3qubit_povm_mosek(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.mosek
@pytest.mark.threequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 3, "gate", pro[0], pro[1], pro[2], pro[3], "mosek", 4)
        for pro in product(
            get_names_3qubit_gate_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_3qubit_gate_mosek(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


# 1-qutrit, MOSEK


@pytest.mark.cvxpy
@pytest.mark.mosek
@pytest.mark.onequtrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 1, "state", pro[0], pro[1], pro[2], pro[3], "mosek", 4)
        for pro in product(
            get_names_1qutrit_state_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qutrit_state_mosek(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.mosek
@pytest.mark.onequtrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 1, "povm", pro[0], pro[1], pro[2], pro[3], "mosek", 4)
        for pro in product(
            get_names_1qutrit_povm_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qutrit_povm_mosek(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.mosek
@pytest.mark.onequtrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 1, "gate", pro[0], pro[1], pro[2], pro[3], "mosek", 4)
        for pro in product(
            get_names_1qutrit_gate_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qutrit_gate_mosek(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


# 2-qutrit, MOSEK


@pytest.mark.cvxpy
@pytest.mark.mosek
@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 2, "state", pro[0], pro[1], pro[2], pro[3], "mosek", 4)
        for pro in product(
            get_names_2qutrit_state_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qutrit_state_mosek(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.mosek
@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 2, "povm", pro[0], pro[1], pro[2], pro[3], "mosek", 4)
        for pro in product(
            get_names_2qutrit_povm_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qutrit_povm_mosek(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.mosek
@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 2, "gate", pro[0], pro[1], pro[2], pro[3], "mosek", 4)
        for pro in product(
            get_names_2qutrit_gate_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qutrit_gate_mosek(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


# 1-qubit, SCS


@pytest.mark.cvxpy
@pytest.mark.scs
@pytest.mark.onequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 1, "state", pro[0], pro[1], pro[2], pro[3], "scs", 4)
        for pro in product(
            get_names_1qubit_state_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qubit_state_scs(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.scs
@pytest.mark.onequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 1, "povm", pro[0], pro[1], pro[2], pro[3], "scs", 4)
        for pro in product(
            get_names_1qubit_povm_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qubit_povm_scs(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.scs
@pytest.mark.onequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 1, "gate", pro[0], pro[1], pro[2], pro[3], "scs", 4)
        for pro in product(
            get_names_1qubit_gate_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qubit_gate_scs(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


# 2-qubit, MOSEK


@pytest.mark.cvxpy
@pytest.mark.scs
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 2, "state", pro[0], pro[1], pro[2], pro[3], "scs", 4)
        for pro in product(
            get_names_2qubit_state_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qubit_state_scs(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.scs
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 2, "povm", pro[0], pro[1], pro[2], pro[3], "scs", 4)
        for pro in product(
            get_names_2qubit_povm_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qubit_povm_scs(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.scs
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 2, "gate", pro[0], pro[1], pro[2], pro[3], "scs", 4)
        for pro in product(
            get_names_2qubit_gate_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qubit_gate_scs(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


# 3-qubit, MOSEK


@pytest.mark.cvxpy
@pytest.mark.scs
@pytest.mark.threequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 3, "state", pro[0], pro[1], pro[2], pro[3], "scs", 4)
        for pro in product(
            get_names_3qubit_state_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_3qubit_state_scs(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.scs
@pytest.mark.threequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 3, "povm", pro[0], pro[1], pro[2], pro[3], "scs", 4)
        for pro in product(
            get_names_3qubit_povm_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_3qubit_povm_scs(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.scs
@pytest.mark.threequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 3, "gate", pro[0], pro[1], pro[2], pro[3], "scs", 4)
        for pro in product(
            get_names_3qubit_gate_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_3qubit_gate_scs(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


# 1-qutrit, MOSEK


@pytest.mark.cvxpy
@pytest.mark.scs
@pytest.mark.onequtrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 1, "state", pro[0], pro[1], pro[2], pro[3], "scs", 4)
        for pro in product(
            get_names_1qutrit_state_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qutrit_state_scs(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.scs
@pytest.mark.onequtrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 1, "povm", pro[0], pro[1], pro[2], pro[3], "scs", 4)
        for pro in product(
            get_names_1qutrit_povm_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qutrit_povm_scs(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.scs
@pytest.mark.onequtrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 1, "gate", pro[0], pro[1], pro[2], pro[3], "scs", 4)
        for pro in product(
            get_names_1qutrit_gate_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qutrit_gate_scs(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


# 2-qutrit, MOSEK


@pytest.mark.cvxpy
@pytest.mark.scs
@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 2, "state", pro[0], pro[1], pro[2], pro[3], "scs", 4)
        for pro in product(
            get_names_2qutrit_state_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qutrit_state_scs(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.scs
@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 2, "povm", pro[0], pro[1], pro[2], pro[3], "scs", 4)
        for pro in product(
            get_names_2qutrit_povm_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qutrit_povm_scs(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.scs
@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 2, "gate", pro[0], pro[1], pro[2], pro[3], "scs", 4)
        for pro in product(
            get_names_2qutrit_gate_test(),
            get_names_cvxpy_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qutrit_gate_scs(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


# 1-qubit, CVXOPT


@pytest.mark.cvxpy
@pytest.mark.cvxopt
@pytest.mark.onequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 1, "state", pro[0], pro[1], pro[2], pro[3], "cvxopt", 4)
        for pro in product(
            get_names_1qubit_state_test(),
            get_names_cvxopt_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qubit_state_cvxopt(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.cvxopt
@pytest.mark.onequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 1, "povm", pro[0], pro[1], pro[2], pro[3], "cvxopt", 4)
        for pro in product(
            get_names_1qubit_povm_test(),
            get_names_cvxopt_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qubit_povm_cvxopt(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.cvxopt
@pytest.mark.onequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 1, "gate", pro[0], pro[1], pro[2], pro[3], "cvxopt", 4)
        for pro in product(
            get_names_1qubit_gate_test(),
            get_names_cvxopt_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qubit_gate_cvxopt(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


# 2-qubit, MOSEK


@pytest.mark.cvxpy
@pytest.mark.cvxopt
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 2, "state", pro[0], pro[1], pro[2], pro[3], "cvxopt", 4)
        for pro in product(
            get_names_2qubit_state_test(),
            get_names_cvxopt_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qubit_state_cvxopt(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.cvxopt
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 2, "povm", pro[0], pro[1], pro[2], pro[3], "cvxopt", 4)
        for pro in product(
            get_names_2qubit_povm_test(),
            get_names_cvxopt_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qubit_povm_cvxopt(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.cvxopt
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 2, "gate", pro[0], pro[1], pro[2], pro[3], "cvxopt", 4)
        for pro in product(
            get_names_2qubit_gate_test(),
            get_names_cvxopt_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qubit_gate_cvxopt(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


# 3-qubit, MOSEK


@pytest.mark.cvxpy
@pytest.mark.cvxopt
@pytest.mark.threequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 3, "state", pro[0], pro[1], pro[2], pro[3], "cvxopt", 4)
        for pro in product(
            get_names_3qubit_state_test(),
            get_names_cvxopt_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_3qubit_state_cvxopt(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.cvxopt
@pytest.mark.threequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 3, "povm", pro[0], pro[1], pro[2], pro[3], "cvxopt", 4)
        for pro in product(
            get_names_3qubit_povm_test(),
            get_names_cvxopt_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_3qubit_povm_cvxopt(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.cvxopt
@pytest.mark.threequbit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qubit", 3, "gate", pro[0], pro[1], pro[2], pro[3], "cvxopt", 4)
        for pro in product(
            get_names_3qubit_gate_test(),
            get_names_cvxopt_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_3qubit_gate_cvxopt(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


# 1-qutrit, MOSEK


@pytest.mark.cvxpy
@pytest.mark.cvxopt
@pytest.mark.onequtrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 1, "state", pro[0], pro[1], pro[2], pro[3], "cvxopt", 4)
        for pro in product(
            get_names_1qutrit_state_test(),
            get_names_cvxopt_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qutrit_state_cvxopt(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.cvxopt
@pytest.mark.onequtrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 1, "povm", pro[0], pro[1], pro[2], pro[3], "cvxopt", 4)
        for pro in product(
            get_names_1qutrit_povm_test(),
            get_names_cvxopt_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qutrit_povm_cvxopt(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.cvxopt
@pytest.mark.onequtrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 1, "gate", pro[0], pro[1], pro[2], pro[3], "cvxopt", 4)
        for pro in product(
            get_names_1qutrit_gate_test(),
            get_names_cvxopt_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_1qutrit_gate_cvxopt(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


# 2-qutrit, MOSEK


@pytest.mark.cvxpy
@pytest.mark.cvxopt
@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 2, "state", pro[0], pro[1], pro[2], pro[3], "cvxopt", 4)
        for pro in product(
            get_names_2qutrit_state_test(),
            get_names_cvxopt_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qutrit_state_cvxopt(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.cvxopt
@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 2, "povm", pro[0], pro[1], pro[2], pro[3], "cvxopt", 4)
        for pro in product(
            get_names_2qutrit_povm_test(),
            get_names_cvxopt_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qutrit_povm_cvxopt(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )


@pytest.mark.cvxpy
@pytest.mark.cvxopt
@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    (
        "mode_sys",
        "num_sys",
        "mode_tomo",
        "name_true",
        "name_loss",
        "mode_loss_form",
        "mode_constraint",
        "name_solver",
        "decimal",
    ),
    [
        ("qutrit", 2, "gate", pro[0], pro[1], pro[2], pro[3], "cvxopt", 4)
        for pro in product(
            get_names_2qutrit_gate_test(),
            get_names_cvxopt_loss_test(),
            get_modes_loss_form(),
            get_modes_constraint(),
        )
    ],
)
def test_estimator_consistency_cvxpy_2qutrit_gate_cvxopt(
    mode_sys,
    num_sys,
    mode_tomo,
    name_true,
    name_loss,
    mode_loss_form,
    mode_constraint,
    name_solver,
    decimal,
):
    _test_estimator_consistency_cvxpy(
        mode_sys,
        num_sys,
        mode_tomo,
        name_true,
        name_loss,
        mode_loss_form,
        mode_constraint,
        name_solver,
        decimal,
    )
