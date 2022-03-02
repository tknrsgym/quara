import numpy as np
import numpy.testing as npt
import pytest

from quara.interface.cvxpy import api
from quara.objects.qoperation_typical import (
    generate_qoperation_object,
    generate_qoperation,
)
from quara.objects.composite_system_typical import generate_composite_system
from quara.interface.cvxpy.qtomography.standard.loss_function import (
    CvxpyRelativeEntropy,
)
from quara.interface.cvxpy.qtomography.standard.loss_function import (
    CvxpyRelativeEntropy,
    CvxpyUniformSquaredError,
    CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm,
)


def _get_tester_1qubit(mode: str):
    num_qubits = 1
    c_sys = generate_composite_system(mode="qubit", num=num_qubits)

    # State
    state_names = ["x0", "y0", "z0", "z1"]
    tester_states = [
        generate_qoperation_object(
            mode="state", object_name="state", name=name, c_sys=c_sys
        )
        for name in state_names
    ]
    # Povm
    povm_names = ["x", "y", "z"]
    tester_povms = [
        generate_qoperation_object(
            mode="povm", object_name="povm", name=name, c_sys=c_sys
        )
        for name in povm_names
    ]

    if mode == "state":
        return tester_states
    elif mode == "povm":
        return tester_povms


@pytest.mark.cvxpy
@pytest.mark.parametrize(
    ("estimator_name", "name_solver"),
    [
        ("maximum-likelihood", "mosek"),
        ("approximate-maximum-likelihood", "mosek"),
        ("least-squares", "mosek"),
    ],
)
def test_estimate_standard_qst_with_cvxpy(estimator_name: str, name_solver: str):
    # Arrange
    c_sys = generate_composite_system(mode="qubit", num=1)
    true_object_name = "a"
    true_object = generate_qoperation(mode="state", name=true_object_name, c_sys=c_sys)
    empi_dists = [
        (10, np.array([0.85355339, 0.14644661])),
        (10, np.array([0.85355339, 0.14644661])),
        (10, np.array([0.5, 0.5])),
    ]
    tester_povms = _get_tester_1qubit(mode="povm")

    # Act
    actual = api.estimate_standard_qst_with_cvxpy(
        tester_povms=tester_povms,
        empi_dists=empi_dists,
        estimator_name=estimator_name,
        schedules="all",
        name_solver=name_solver,
    )

    # Assert
    actual_var = actual.estimated_var
    expected_var = true_object.to_var()
    decimal = 1e-8
    npt.assert_almost_equal(actual_var, expected_var, decimal=decimal)

    # Act
    actual = api.estimate_standard_qtomography_with_cvxpy(
        type_qoperation="state",
        tester=tester_povms,
        empi_dists=empi_dists,
        estimator_name=estimator_name,
        schedules="all",
        name_solver=name_solver,
    )

    # Assert
    actual_var = actual.estimated_var
    expected_var = true_object.to_var()
    decimal = 1e-8
    npt.assert_almost_equal(actual_var, expected_var, decimal=decimal)


@pytest.mark.cvxpy
@pytest.mark.parametrize(
    ("estimator_name", "name_solver"),
    [
        ("maximum-likelihood", "mosek"),
        ("approximate-maximum-likelihood", "mosek"),
        ("least-squares", "mosek"),
    ],
)
def test_estimate_standard_povmt_with_cvxpy(estimator_name: str, name_solver: str):
    # Arrange
    c_sys = generate_composite_system(mode="qubit", num=1)
    true_object_name = "z"
    true_object = generate_qoperation(mode="povm", name=true_object_name, c_sys=c_sys)
    empi_dists = [
        (10, np.array([0.5, 0.5])),
        (10, np.array([0.5, 0.5])),
        (10, np.array([1.0, 0.0])),
        (10, np.array([0.0, 1.0])),
    ]
    tester_states = _get_tester_1qubit(mode="state")

    # Act
    actual = api.estimate_standard_povmt_with_cvxpy(
        tester_states=tester_states,
        empi_dists=empi_dists,
        estimator_name=estimator_name,
        schedules="all",
        name_solver=name_solver,
        num_outcomes=true_object.num_outcomes,
    )

    # Assert
    actual_var = actual.estimated_var
    expected_var = true_object.to_var()
    decimal = 1e-8
    npt.assert_almost_equal(actual_var, expected_var, decimal=decimal)

    # Act
    actual = api.estimate_standard_qtomography_with_cvxpy(
        type_qoperation="povm",
        tester=tester_states,
        empi_dists=empi_dists,
        estimator_name=estimator_name,
        schedules="all",
        name_solver=name_solver,
        num_outcomes=true_object.num_outcomes,
    )

    # Assert
    actual_var = actual.estimated_var
    expected_var = true_object.to_var()
    decimal = 1e-8
    npt.assert_almost_equal(actual_var, expected_var, decimal=decimal)


@pytest.mark.cvxpy
@pytest.mark.parametrize(
    ("estimator_name", "name_solver"),
    [
        ("maximum-likelihood", "mosek"),
        ("approximate-maximum-likelihood", "mosek"),
        ("least-squares", "mosek"),
    ],
)
def test_estimate_standard_qpt_with_cvxpy(estimator_name: str, name_solver: str):
    # Arrange
    c_sys = generate_composite_system(mode="qubit", num=1)
    true_object_name = "x90"
    true_object = generate_qoperation(mode="gate", name=true_object_name, c_sys=c_sys)
    empi_dists = [
        (10, np.array([1.0, 0.0])),
        (10, np.array([0.5, 0.5])),
        (10, np.array([0.5, 0.5])),
        (10, np.array([0.5, 0.5])),
        (10, np.array([0.5, 0.5])),
        (10, np.array([1.0, 0.0])),
        (10, np.array([0.5, 0.5])),
        (10, np.array([0.0, 1.0])),
        (10, np.array([0.5, 0.5])),
        (10, np.array([0.5, 0.5])),
        (10, np.array([1.0, 0.0])),
        (10, np.array([0.5, 0.5])),
    ]
    tester_states = _get_tester_1qubit(mode="state")
    tester_povms = _get_tester_1qubit(mode="povm")

    # Act
    actual = api.estimate_standard_qpt_with_cvxpy(
        tester_states=tester_states,
        tester_povms=tester_povms,
        empi_dists=empi_dists,
        estimator_name=estimator_name,
        schedules="all",
        name_solver=name_solver,
    )

    # Assert
    actual_var = actual.estimated_var
    expected_var = true_object.to_var()
    decimal = 1e-8
    npt.assert_almost_equal(actual_var, expected_var, decimal=decimal)

    # Act
    actual = api.estimate_standard_qtomography_with_cvxpy(
        type_qoperation="gate",
        tester=tester_povms + tester_states,
        empi_dists=empi_dists,
        estimator_name=estimator_name,
        schedules="all",
        name_solver=name_solver,
    )

    # Assert
    actual_var = actual.estimated_var
    expected_var = true_object.to_var()
    decimal = 1e-8
    npt.assert_almost_equal(actual_var, expected_var, decimal=decimal)


@pytest.mark.cvxpy
@pytest.mark.parametrize(
    ("estimator_name", "name_solver"),
    [
        ("maximum-likelihood", "mosek"),
        ("approximate-maximum-likelihood", "mosek"),
        ("least-squares", "mosek"),
    ],
)
def test_estimate_standard_qmpt_with_cvxpy(estimator_name: str, name_solver: str):
    # Arrange
    c_sys = generate_composite_system(mode="qubit", num=1)
    true_object_name = "z-type2"
    true_object = generate_qoperation(
        mode="mprocess", name=true_object_name, c_sys=c_sys
    )
    empi_dists = [
        (10, np.array([0.25, 0.25, 0.25, 0.25])),
        (10, np.array([0.25, 0.25, 0.25, 0.25])),
        (10, np.array([0.5, 0.0, 0.5, 0.0])),
        (10, np.array([0.25, 0.25, 0.25, 0.25])),
        (10, np.array([0.25, 0.25, 0.25, 0.25])),
        (10, np.array([0.5, 0.0, 0.5, 0.0])),
        (10, np.array([0.5, 0.5, 0.0, 0.0])),
        (10, np.array([0.5, 0.5, 0.0, 0.0])),
        (10, np.array([1.0, 0.0, 0.0, 0.0])),
        (10, np.array([0.0, 0.0, 0.5, 0.5])),
        (10, np.array([0.0, 0.0, 0.5, 0.5])),
        (10, np.array([0.0, 0.0, 1.0, 0.0])),
    ]
    tester_states = _get_tester_1qubit(mode="state")
    tester_povms = _get_tester_1qubit(mode="povm")

    # Act
    actual = api.estimate_standard_qmpt_with_cvxpy(
        tester_states=tester_states,
        tester_povms=tester_povms,
        empi_dists=empi_dists,
        estimator_name=estimator_name,
        schedules="all",
        name_solver=name_solver,
        num_outcomes=true_object.num_outcomes,
    )

    # Assert
    actual_var = actual.estimated_var
    expected_var = true_object.to_var()
    decimal = 1e-8
    npt.assert_almost_equal(actual_var, expected_var, decimal=decimal)

    # Act
    actual = api.estimate_standard_qtomography_with_cvxpy(
        type_qoperation="mprocess",
        tester=tester_povms + tester_states,
        empi_dists=empi_dists,
        estimator_name=estimator_name,
        schedules="all",
        name_solver=name_solver,
        num_outcomes=true_object.num_outcomes,
    )

    # Assert
    actual_var = actual.estimated_var
    expected_var = true_object.to_var()
    decimal = 1e-8
    npt.assert_almost_equal(actual_var, expected_var, decimal=decimal)


@pytest.mark.cvxpy
def test_estimate_standard_qtomography_with_cvxpy_invalid():
    # Arrange
    c_sys = generate_composite_system(mode="qubit", num=1)
    true_object_name = "z"
    true_object = generate_qoperation(mode="povm", name=true_object_name, c_sys=c_sys)
    empi_dists = [
        (10, np.array([0.5, 0.5])),
        (10, np.array([0.5, 0.5])),
        (10, np.array([1.0, 0.0])),
        (10, np.array([0.0, 1.0])),
    ]
    tester_states = _get_tester_1qubit(mode="state")
    ok_name_solver = "mosek"
    ok_estimator_name = "maximum-likelihood"
    ok_type_qoperation = "povm"
    ng_name_solver = "mose"
    ng_estimator_name = "maximum-likelihoodd"
    ng_type_qoperation = "states"  # Invalid

    # Invalid type_qoperation
    with pytest.raises(ValueError):
        _ = api.estimate_standard_qtomography_with_cvxpy(
            type_qoperation=ng_type_qoperation,
            tester=tester_states,
            empi_dists=empi_dists,
            estimator_name=ok_estimator_name,
            schedules="all",
            name_solver=ok_name_solver,
            num_outcomes=true_object.num_outcomes,
        )

    # Invalid name_solver
    with pytest.raises(ValueError):
        _ = api.estimate_standard_qtomography_with_cvxpy(
            type_qoperation=ok_type_qoperation,
            tester=tester_states,
            empi_dists=empi_dists,
            estimator_name=ok_estimator_name,
            schedules="all",
            name_solver=ng_name_solver,
            num_outcomes=true_object.num_outcomes,
        )

    # Invalid estimator_name
    with pytest.raises(ValueError):
        _ = api.estimate_standard_qtomography_with_cvxpy(
            type_qoperation=ok_type_qoperation,
            tester=tester_states,
            empi_dists=empi_dists,
            estimator_name=ng_estimator_name,
            schedules="all",
            name_solver=ok_name_solver,
            num_outcomes=true_object.num_outcomes,
        )

    # Invalid num_outcomes
    with pytest.raises(ValueError):
        _ = api.estimate_standard_qtomography_with_cvxpy(
            type_qoperation=ok_type_qoperation,
            tester=tester_states,
            empi_dists=empi_dists,
            estimator_name=ng_estimator_name,
            schedules="all",
            name_solver=ok_name_solver,
            num_outcomes=1,
        )

    # Invalid num_outcomes
    with pytest.raises(TypeError):
        _ = api.estimate_standard_qtomography_with_cvxpy(
            type_qoperation=ok_type_qoperation,
            tester=tester_states,
            empi_dists=empi_dists,
            estimator_name=ng_estimator_name,
            schedules="all",
            name_solver=ok_name_solver,
            num_outcomes=None,
        )


def test_get_estimator_and_options():
    actual = api._get_estimator_and_options("maximum-likelihood", "mosek")
    assert type(actual["loss"]) == CvxpyRelativeEntropy

    actual = api._get_estimator_and_options("approximate-maximum-likelihood", "mosek")
    assert (
        type(actual["loss"]) == CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm
    )

    actual = api._get_estimator_and_options("least-squares", "mosek")
    assert type(actual["loss"]) == CvxpyUniformSquaredError

    # Invalid
    with pytest.raises(ValueError):
        api._get_estimator_and_options("invalid", "mosek")

    with pytest.raises(ValueError):
        api._get_estimator_and_options("least-squares", "invalid")
