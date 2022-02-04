import numpy as np
import numpy.testing as npt
import pytest

from cvxpy.constraints.psd import PSD

from quara.interface.cvxpy import conversion
from quara.objects.composite_system_typical import generate_composite_system


def test_get_valid_qopeartion_type():
    actual = conversion.get_valid_qopeartion_type()
    assert actual == ["state", "povm", "gate", "mprocess"]


@pytest.mark.parametrize(
    ("t", "dim", "num_outcomes", "expected"),
    [
        ("state", 2, None, 3),
        ("state", 3, None, 8),
        ("state", 4, None, 15),
        ("povm", 2, 2, 4),
        ("povm", 3, 2, 9),
        ("povm", 3, 3, 18),
        ("povm", 4, 2, 16),
        ("gate", 2, None, 12),
        ("gate", 3, None, 72),
        ("gate", 4, None, 240),
        ("mprocess", 2, 2, 28),
        ("mprocess", 3, 2, 153),
        ("mprocess", 3, 3, 234),
        ("mprocess", 4, 2, 496),
    ],
)
def test_num_cvxpy_variable(t: str, dim: int, num_outcomes: int, expected: int):
    actual = conversion.num_cvxpy_variable(t, dim, num_outcomes=num_outcomes)
    assert actual == expected


def test_num_cvxpy_variable_error():
    # t is not supported.
    with pytest.raises(ValueError):
        conversion.num_cvxpy_variable("not_supported", 2, None)

    # dim is not a positive number.
    with pytest.raises(ValueError):
        conversion.num_cvxpy_variable("state", 0, None)

    # Type of QOperation is povm, but num_outcomes is not specified.
    with pytest.raises(ValueError):
        conversion.num_cvxpy_variable("povm", 2, None)

    # Type of QOperation is mprocess, but num_outcomes is not specified.
    with pytest.raises(ValueError):
        conversion.num_cvxpy_variable("mprocess", 2, None)


@pytest.mark.parametrize(
    ("t", "dim", "num_outcomes", "expected"),
    [
        ("state", 2, None, 3),
        ("povm", 2, 2, 4),
        ("gate", 2, None, 12),
        ("mprocess", 2, 2, 28),
    ],
)
def test_generate_cvxpy_variable(t: str, dim: int, num_outcomes: int, expected: int):
    actual = conversion.generate_cvxpy_variable(t, dim, num_outcomes=num_outcomes)
    assert actual.size == expected


def test_generate_cvxpy_constraints_from_cvxpy_variable():
    c_sys = generate_composite_system("qubit", 1)

    # state
    var = conversion.generate_cvxpy_variable("state", 2)
    actual = conversion.generate_cvxpy_constraints_from_cvxpy_variable(
        c_sys, "state", var
    )
    assert type(actual) == list
    assert len(actual) == 1
    assert type(actual[0]) == PSD
    expression = actual[0].args[0]
    assert expression.curvature == "AFFINE"
    assert expression.sign == "UNKNOWN"
    assert expression.shape == (2, 2)

    # povm
    var = conversion.generate_cvxpy_variable("povm", 2, num_outcomes=2)
    actual = conversion.generate_cvxpy_constraints_from_cvxpy_variable(
        c_sys, "povm", var, num_outcomes=2
    )
    assert type(actual) == list
    assert len(actual) == 2
    for a in actual:
        assert type(a) == PSD
        expression = a.args[0]
        assert expression.curvature == "AFFINE"
        assert expression.sign == "UNKNOWN"
        assert expression.shape == (2, 2)

    # gate
    var = conversion.generate_cvxpy_variable("gate", 2)
    actual = conversion.generate_cvxpy_constraints_from_cvxpy_variable(
        c_sys, "gate", var
    )
    assert type(actual) == list
    assert len(actual) == 1
    assert type(actual[0]) == PSD
    expression = actual[0].args[0]
    assert expression.curvature == "AFFINE"
    assert expression.sign == "UNKNOWN"
    assert expression.shape == (4, 4)

    # mprocess
    var = conversion.generate_cvxpy_variable("mprocess", 2, num_outcomes=2)
    actual = conversion.generate_cvxpy_constraints_from_cvxpy_variable(
        c_sys, "mprocess", var, num_outcomes=2
    )
    assert type(actual) == list
    assert len(actual) == 2
    for a in actual:
        assert type(a) == PSD
        expression = a.args[0]
        assert expression.curvature == "AFFINE"
        assert expression.sign == "UNKNOWN"
        assert expression.shape == (4, 4)


def test_dmat_from_var():
    # Arrange
    c_sys = generate_composite_system("qubit", 1)
    var = conversion.generate_cvxpy_variable("state", 2)

    # Act
    actual = conversion.dmat_from_var(c_sys, var)

    # Assert
    assert actual.curvature == "AFFINE"
    assert actual.sign == "UNKNOWN"
    assert actual.shape == (2, 2)


def test_povm_element_from_var():
    # Arrange
    c_sys = generate_composite_system("qubit", 1)
    var = conversion.generate_cvxpy_variable("povm", 2, num_outcomes=2)

    for index_outcomes in range(2):
        # Act
        actual = conversion.povm_element_from_var(c_sys, 2, index_outcomes, var)

        # Assert
        assert actual.curvature == "AFFINE"
        assert actual.sign == "UNKNOWN"
        assert actual.shape == (2, 2)


def test_choi_from_var():
    # Arrange
    c_sys = generate_composite_system("qubit", 1)
    var = conversion.generate_cvxpy_variable("gate", 2)

    # Act
    actual = conversion.choi_from_var(c_sys, var)

    # Assert
    assert actual.curvature == "AFFINE"
    assert actual.sign == "UNKNOWN"
    assert actual.shape == (4, 4)


def test_mprocess_element_choi_from_var():
    # Arrange
    c_sys = generate_composite_system("qubit", 1)
    var = conversion.generate_cvxpy_variable("mprocess", 2, num_outcomes=2)

    for index_outcomes in range(2):
        # Act
        actual = conversion.mprocess_element_choi_from_var(
            c_sys, 2, index_outcomes, var
        )

        # Assert
        assert actual.curvature == "AFFINE"
        assert actual.sign == "UNKNOWN"
        assert actual.shape == (4, 4)


def test_convert_quara_variable_to_state_vec():
    # Arrange
    var = np.array([0, 0, 1]) / np.sqrt(2)
    # Act
    actual = conversion.convert_quara_variable_to_state_vec(2, var)
    # Assert
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    npt.assert_almost_equal(actual, expected, decimal=15)
