import numpy as np
import numpy.testing as npt
import pytest

from cvxpy.expressions.variable import Variable as CvxpyVariable
from cvxpy.expressions.expression import Expression as CvxpyExpression


from quara.interface.cvxpy.qtomography.standard import loss_function
from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.state_typical import generate_state_from_name
from quara.objects.povm_typical import generate_povm_from_name
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.standard_povmt import StandardPovmt


def get_test_data_qst(on_para_eq_constraint=True):
    c_sys = generate_composite_system("qubit", 1)

    povm_x = generate_povm_from_name("x", c_sys)
    povm_y = generate_povm_from_name("y", c_sys)
    povm_z = generate_povm_from_name("z", c_sys)
    povms = [povm_x, povm_y, povm_z]

    qst = StandardQst(povms, on_para_eq_constraint=on_para_eq_constraint, seed_data=7)

    return qst, c_sys


def get_test_data_povmt(on_para_eq_constraint=True):
    c_sys = generate_composite_system("qubit", 1)

    # |+><+|
    state_x0 = generate_state_from_name(c_sys, "x0")
    # |+i><+i|
    state_y0 = generate_state_from_name(c_sys, "y0")
    # |0><0|
    state_z0 = generate_state_from_name(c_sys, "z0")
    # |1><1|
    state_z1 = generate_state_from_name(c_sys, "z1")
    tester_objects = [state_x0, state_y0, state_z0, state_z1]

    povmt = StandardPovmt(
        tester_objects,
        on_para_eq_constraint=on_para_eq_constraint,
        num_outcomes=2,
        seed_data=7,
    )

    return povmt, c_sys


class TestCvxpyLossFunctionOption:
    def test_access_eps_prob_zero(self):
        # eps_prob_zero = default
        actual = loss_function.CvxpyLossFunctionOption()
        assert actual.eps_prob_zero == 1e-12

        # eps_prob_zero = 1e-11
        actual = loss_function.CvxpyLossFunctionOption(eps_prob_zero=1e-11)
        assert actual.eps_prob_zero == 1e-11

        # Test that "eps_prob_zero" cannot be updated
        with pytest.raises(AttributeError):
            actual.eps_prob_zero = 1e-12


class TestCvxpyLossFunction:
    def test_access_on_value_cvxpy(self):
        actual = loss_function.CvxpyLossFunction()
        assert actual.on_value == False
        assert actual.on_value_cvxpy == False

    def test_access_eps_prob_zero(self):
        # eps_prob_zero = default
        actual = loss_function.CvxpyLossFunction()
        assert actual.eps_prob_zero == 1e-12

        # Test that "eps_prob_zero" cannot be updated
        with pytest.raises(AttributeError):
            actual.eps_prob_zero = 1e-12

    def test_set_from_option(self):
        # Arrange
        loss = loss_function.CvxpyLossFunction()
        option = loss_function.CvxpyLossFunctionOption(eps_prob_zero=1e-11)

        # Action
        loss.set_from_option(option)

        # Assert
        assert loss.option != None
        assert loss.eps_prob_zero == 1e-11

    def test_access_sqt(self):
        # sqt = default
        actual = loss_function.CvxpyLossFunction()
        assert actual.sqt == None

        # Test that "sqt" cannot be updated directly
        sqt, _ = get_test_data_qst()
        with pytest.raises(AttributeError):
            actual.sqt = sqt

    def test_set_standard_qtomography(self):
        ### on_para_eq_constraint = True
        # Arrange
        loss = loss_function.CvxpyLossFunction()
        sqt, c_sys = get_test_data_qst()

        # Action
        loss.set_standard_qtomography(sqt)

        # Assert
        assert loss.sqt != None
        assert loss.type_estimate == "state"
        assert loss.on_value == True
        assert loss.num_var == 3
        assert loss.composite_system == c_sys
        assert loss.dim_system() == 2
        with pytest.raises(ValueError):
            loss.num_outcomes_estimate()

        ### on_para_eq_constraint = False
        sqt, _ = get_test_data_qst(on_para_eq_constraint=False)
        with pytest.raises(ValueError):
            loss.set_standard_qtomography(sqt)

    def test_access_type_estimate(self):
        # type_estimate = default
        actual = loss_function.CvxpyLossFunction()
        assert actual.type_estimate == None

        # Test that "type_estimate" cannot be updated
        with pytest.raises(AttributeError):
            actual.type_estimate = "state"

    def test_access_nums_data(self):
        # nums_data = default
        actual = loss_function.CvxpyLossFunction()
        assert actual.nums_data == None

        # Test that "nums_data" cannot be updated
        with pytest.raises(AttributeError):
            actual.nums_data = [100, 100, 100]

    def test_access_num_data_total(self):
        # num_data_total = default
        actual = loss_function.CvxpyLossFunction()
        assert actual.num_data_total == None

        # Test that "num_data_total" cannot be updated
        with pytest.raises(AttributeError):
            actual.num_data_total = 300

    def test_access_num_data_ratios(self):
        # num_data_ratios = default
        actual = loss_function.CvxpyLossFunction()
        assert actual.num_data_ratios == None

        # Test that "num_data_ratios" cannot be updated
        with pytest.raises(AttributeError):
            actual.num_data_ratios = [1 / 3, 1 / 3, 1 / 3]

    def test_calc_average_num_data(self):
        # Arrange
        loss = loss_function.CvxpyLossFunction()
        empi_dists = [
            (100, np.array([0.5, 0.5])),
            (100, np.array([0.5, 0.5])),
            (100, np.array([1.0, 0.0])),
        ]

        # Act
        loss.set_prob_dists_data_from_empi_dists(empi_dists)

        # Assert
        assert loss.calc_average_num_data() == 100

    def test_access_prob_dists_data(self):
        # prob_dists_data = default
        actual = loss_function.CvxpyLossFunction()
        assert actual.prob_dists_data == None

        # Test that "prob_dists_data" cannot be updated
        prob_dists = [
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            np.array([1.0, 0.0]),
        ]
        with pytest.raises(AttributeError):
            actual.prob_dists_data = prob_dists

    def test_access_on_prob_dists_data(self):
        # prob_dists_data = default
        actual = loss_function.CvxpyLossFunction()
        assert actual.on_prob_dists_data == False

        # Test that "on_prob_dists_data" cannot be updated
        with pytest.raises(AttributeError):
            actual.on_prob_dists_data = True

    def test_set_prob_dists_data(self):
        # Arrange
        loss = loss_function.CvxpyLossFunction()
        prob_dists = [
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            np.array([1.0, 0.0]),
        ]

        # Action
        loss.set_prob_dists_data(prob_dists)

        # Assert
        assert loss.on_prob_dists_data == True
        for a, e in zip(loss.prob_dists_data, prob_dists):
            npt.assert_almost_equal(a, e, decimal=15)

    def test_erase_prob_dists_data(self):
        # Arrange
        loss = loss_function.CvxpyLossFunction()
        prob_dists = [
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            np.array([1.0, 0.0]),
        ]
        loss.set_prob_dists_data(prob_dists)

        # Action
        loss.erase_prob_dists_data()

        # Assert
        assert loss.on_prob_dists_data == False
        assert loss.prob_dists_data == None

    def test_calc_prob_model(self):
        # Arrange
        loss = loss_function.CvxpyLossFunction()
        sqt, _ = get_test_data_qst()
        loss.set_standard_qtomography(sqt)
        var = CvxpyVariable(3)
        var.value = np.array([0, 0, 1]) / np.sqrt(2)

        # Act & Assert
        for i in range(3):
            for x in range(2):
                actual = loss.calc_prob_model(i, x, var)
                assert isinstance(actual, CvxpyExpression) == True

    def test_set_prob_dists_data_from_empi_dists(self):
        # Arrange
        loss = loss_function.CvxpyLossFunction()
        empi_dists = [
            (100, np.array([0.5, 0.5])),
            (100, np.array([0.5, 0.5])),
            (100, np.array([1.0, 0.0])),
        ]

        # Act
        loss.set_prob_dists_data_from_empi_dists(empi_dists)

        # Assert
        assert loss.nums_data == [100, 100, 100]
        assert loss.num_data_total == 300
        assert loss.num_data_ratios == [1 / 3, 1 / 3, 1 / 3]
        expected_prob_dists_data = [
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            np.array([1.0, 0.0]),
        ]
        for a, e in zip(loss.prob_dists_data, expected_prob_dists_data):
            npt.assert_almost_equal(a, e, decimal=15)

    def test_access_composite_system(self):
        # composite_system = default
        actual = loss_function.CvxpyLossFunction()
        assert actual.composite_system == None

        # Test that "composite_system" cannot be updated
        _, c_sys = get_test_data_qst()
        with pytest.raises(AttributeError):
            actual.composite_system = c_sys

    def test_dim_system(self):
        # composite_system = default
        actual = loss_function.CvxpyLossFunction()
        assert actual.dim_system() == None

    def test_num_outcomes_estimate(self):
        # sqt = None
        actual = loss_function.CvxpyLossFunction()
        with pytest.raises(ValueError):
            actual.num_outcomes_estimate()

        # sqt = QST
        actual = loss_function.CvxpyLossFunction()
        sqt, _ = get_test_data_qst()
        actual.set_standard_qtomography(sqt)
        with pytest.raises(ValueError):
            actual.num_outcomes_estimate()

        # sqt = POVMT
        actual = loss_function.CvxpyLossFunction()
        sqt, _ = get_test_data_povmt()
        actual.set_standard_qtomography(sqt)
        assert actual.num_outcomes_estimate() == 2

    def test_value(self):
        # raise NotImplementedError
        loss = loss_function.CvxpyLossFunction()
        var = np.array([0, 0, 1]) / np.sqrt(2)
        with pytest.raises(NotImplementedError):
            loss.value(var)

    def test_value_cvxpy(self):
        # raise NotImplementedError
        actual = loss_function.CvxpyLossFunction()
        var = CvxpyVariable(3)
        with pytest.raises(NotImplementedError):
            actual.value_cvxpy(var)


class TestCvxpyRelativeEntropy:
    def test_access_on_value_cvxpy(self):
        actual = loss_function.CvxpyRelativeEntropy()
        assert actual.on_value == True
        assert actual.on_value_cvxpy == True

    def test_is_option_sufficient(self):
        # is_option_sufficient = True
        actual = loss_function.CvxpyRelativeEntropy()
        assert actual.is_option_sufficient() == True

    def test_value(self):
        # Arrange
        loss = loss_function.CvxpyRelativeEntropy()
        option = loss_function.CvxpyLossFunctionOption()
        loss.set_from_option(option)
        sqt, _ = get_test_data_qst()
        loss.set_standard_qtomography(sqt)
        empi_dists = [
            (100, np.array([0.5, 0.5])),
            (100, np.array([0.5, 0.5])),
            (100, np.array([1.0, 0.0])),
        ]
        loss.set_prob_dists_data_from_empi_dists(empi_dists)
        var = np.array([0, 0, 1]) / np.sqrt(2)

        # Act
        actual = loss.value(var)

        # Assert
        assert type(actual) == np.float64

    def test_value_cvxpy(self):
        ### on_prob_dists_data = True
        # Arrange
        loss = loss_function.CvxpyRelativeEntropy()
        option = loss_function.CvxpyLossFunctionOption()
        loss.set_from_option(option)
        sqt, _ = get_test_data_qst()
        loss.set_standard_qtomography(sqt)
        empi_dists = [
            (100, np.array([0.5, 0.5])),
            (100, np.array([0.5, 0.5])),
            (100, np.array([1.0, 0.0])),
        ]
        loss.set_prob_dists_data_from_empi_dists(empi_dists)
        var = CvxpyVariable(3)

        # Act
        actual = loss.value_cvxpy(var)
        assert isinstance(actual, CvxpyExpression) == True

        ### on_prob_dists_data = False
        # Arrange
        loss = loss_function.CvxpyRelativeEntropy()
        option = loss_function.CvxpyLossFunctionOption()
        loss.set_from_option(option)
        sqt, _ = get_test_data_qst()
        loss.set_standard_qtomography(sqt)
        var = CvxpyVariable(3)

        # Act & Assert
        with pytest.raises(ValueError):
            loss.value_cvxpy(var)


class TestCvxpyUniformSquaredError:
    def test_access_on_value_cvxpy(self):
        actual = loss_function.CvxpyUniformSquaredError()
        assert actual.on_value == True
        assert actual.on_value_cvxpy == True

    def test_value(self):
        # Arrange
        loss = loss_function.CvxpyUniformSquaredError()
        option = loss_function.CvxpyLossFunctionOption()
        loss.set_from_option(option)
        sqt, _ = get_test_data_qst()
        loss.set_standard_qtomography(sqt)
        empi_dists = [
            (100, np.array([0.5, 0.5])),
            (100, np.array([0.5, 0.5])),
            (100, np.array([1.0, 0.0])),
        ]
        loss.set_prob_dists_data_from_empi_dists(empi_dists)
        var = np.array([0, 0, 1]) / np.sqrt(2)

        # Act
        actual = loss.value(var)

        # Assert
        assert type(actual) == np.float64

    def test_value_cvxpy(self):
        ### on_prob_dists_data = True
        # Arrange
        loss = loss_function.CvxpyUniformSquaredError()
        option = loss_function.CvxpyLossFunctionOption()
        loss.set_from_option(option)
        sqt, _ = get_test_data_qst()
        loss.set_standard_qtomography(sqt)
        empi_dists = [
            (100, np.array([0.5, 0.5])),
            (100, np.array([0.5, 0.5])),
            (100, np.array([1.0, 0.0])),
        ]
        loss.set_prob_dists_data_from_empi_dists(empi_dists)
        var = CvxpyVariable(3)

        # Act
        actual = loss.value_cvxpy(var)
        assert isinstance(actual, CvxpyExpression) == True

        ### on_prob_dists_data = False
        # Arrange
        loss = loss_function.CvxpyUniformSquaredError()
        option = loss_function.CvxpyLossFunctionOption()
        loss.set_from_option(option)
        sqt, _ = get_test_data_qst()
        loss.set_standard_qtomography(sqt)
        var = CvxpyVariable(3)

        # Act & Assert
        with pytest.raises(ValueError):
            loss.value_cvxpy(var)


class TestCvxpyApproximateRelativeEntropyWithZeroProbabilityTerm:
    def test_access_on_value_cvxpy(self):
        actual = loss_function.CvxpyUniformSquaredError()
        assert actual.on_value == True
        assert actual.on_value_cvxpy == True

    def test_value(self):
        # Arrange
        loss = loss_function.CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm()
        option = loss_function.CvxpyLossFunctionOption()
        loss.set_from_option(option)
        sqt, _ = get_test_data_qst()
        loss.set_standard_qtomography(sqt)
        empi_dists = [
            (100, np.array([0.5, 0.5])),
            (100, np.array([0.5, 0.5])),
            (100, np.array([1.0, 0.0])),
        ]
        loss.set_prob_dists_data_from_empi_dists(empi_dists)
        var = np.array([0, 0, 1]) / np.sqrt(2)

        # Act
        actual = loss.value(var)

        # Assert
        assert type(actual) == np.float64

    def test_value_cvxpy(self):
        ### on_prob_dists_data = True
        # Arrange
        loss = loss_function.CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm()
        option = loss_function.CvxpyLossFunctionOption()
        loss.set_from_option(option)
        sqt, _ = get_test_data_qst()
        loss.set_standard_qtomography(sqt)
        empi_dists = [
            (100, np.array([0.5, 0.5])),
            (100, np.array([0.5, 0.5])),
            (100, np.array([1.0, 0.0])),
        ]
        loss.set_prob_dists_data_from_empi_dists(empi_dists)
        var = CvxpyVariable(3)

        # Act
        actual = loss.value_cvxpy(var)
        assert isinstance(actual, CvxpyExpression) == True

        ### on_prob_dists_data = False
        # Arrange
        loss = loss_function.CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm()
        option = loss_function.CvxpyLossFunctionOption()
        loss.set_from_option(option)
        sqt, _ = get_test_data_qst()
        loss.set_standard_qtomography(sqt)
        var = CvxpyVariable(3)

        # Act & Assert
        with pytest.raises(ValueError):
            loss.value_cvxpy(var)
