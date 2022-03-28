import numpy as np
import numpy.testing as npt
import pytest

from cvxpy.expressions.variable import Variable as CvxpyVariable
from cvxpy.expressions.expression import Expression as CvxpyExpression

from quara.interface.cvxpy.qtomography.standard.minimization_algorithm import (
    CvxpyMinimizationAlgorithm,
    CvxpyMinimizationAlgorithmOption,
    CvxpyMinimizationResult,
)
from quara.interface.cvxpy.qtomography.standard.loss_function import (
    CvxpyRelativeEntropy,
    CvxpyUniformSquaredError,
    CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm,
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


class TestCvxpyMinimizationResult:
    def test_access_variable_value(self):
        variable_value = np.array([1, 2])
        actual = CvxpyMinimizationResult(variable_value)
        npt.assert_almost_equal(actual.variable_value, variable_value, decimal=15)
        npt.assert_almost_equal(actual.value, variable_value, decimal=15)

        # Test that "variable_value" cannot be updated
        with pytest.raises(AttributeError):
            actual.variable_value = variable_value

    def test_access_loss_value(self):
        variable_value = np.array([1, 2])

        # loss_value = default(None)
        actual = CvxpyMinimizationResult(variable_value)
        assert actual.loss_value == None

        # loss_value = 1.0
        actual = CvxpyMinimizationResult(variable_value, loss_value=1.0)
        assert actual.loss_value == 1.0

        # Test that "loss_value" cannot be updated
        with pytest.raises(AttributeError):
            actual.loss_value = 2.0


class TestCvxpyMinimizationAlgorithmOption:
    def test_access_name_solver(self):
        # name_solver = scs
        name_solver = "scs"
        actual = CvxpyMinimizationAlgorithmOption(name_solver)
        assert actual.name_solver == "scs"

        # name_solver = cvxopt
        name_solver = "cvxopt"
        actual = CvxpyMinimizationAlgorithmOption(name_solver)
        assert actual.name_solver == "cvxopt"

        # name_solver = mosek
        name_solver = "mosek"
        actual = CvxpyMinimizationAlgorithmOption(name_solver)
        assert actual.name_solver == "mosek"

        # Unsupported name_solver
        name_solver = "Unsupported"
        with pytest.raises(ValueError):
            CvxpyMinimizationAlgorithmOption(name_solver)

        # Test that "name_solver" cannot be updated
        name_solver = "scs"
        actual = CvxpyMinimizationAlgorithmOption(name_solver)
        with pytest.raises(AttributeError):
            actual.name_solver = "cvxopt"

    def test_access_verbose(self):
        name_solver = "scs"

        # verbose = default(False)
        actual = CvxpyMinimizationAlgorithmOption(name_solver)
        assert actual.verbose == False

        # verbose = True
        actual = CvxpyMinimizationAlgorithmOption(name_solver, verbose=True)
        assert actual.verbose == True

        # verbose = False
        actual = CvxpyMinimizationAlgorithmOption(name_solver, verbose=False)
        assert actual.verbose == False

        # Test that "verbose" cannot be updated
        actual = CvxpyMinimizationAlgorithmOption(name_solver)
        with pytest.raises(AttributeError):
            actual.verbose = True

    def test_access_eps_tol(self):
        name_solver = "scs"

        # eps_tol = default(1e-8)
        actual = CvxpyMinimizationAlgorithmOption(name_solver)
        assert actual.eps_tol == 1e-8

        # eps_tol = 1e-10
        actual = CvxpyMinimizationAlgorithmOption(name_solver, eps_tol=1e-10)
        assert actual.eps_tol == 1e-10

        # eps_tol = negative
        with pytest.raises(ValueError):
            CvxpyMinimizationAlgorithmOption(name_solver, eps_tol=-1)

        # Test that "eps_tol" cannot be updated
        actual = CvxpyMinimizationAlgorithmOption(name_solver)
        with pytest.raises(AttributeError):
            actual.eps_tol = 1e-10

    def test_access_mode_constraint(self):
        name_solver = "scs"

        # mode_constraint = default("physical")
        actual = CvxpyMinimizationAlgorithmOption(name_solver)
        assert actual.mode_constraint == "physical"

        # mode_constraint = "physical"
        actual = CvxpyMinimizationAlgorithmOption(
            name_solver, mode_constraint="physical"
        )
        assert actual.mode_constraint == "physical"

        # mode_constraint = "unconstraint"
        actual = CvxpyMinimizationAlgorithmOption(
            name_solver, mode_constraint="unconstraint"
        )
        assert actual.mode_constraint == "unconstraint"

        # Unsupported mode_constraint
        with pytest.raises(ValueError):
            CvxpyMinimizationAlgorithmOption(name_solver, mode_constraint="Unsupported")

        # Test that "mode_constraint" cannot be updated
        actual = CvxpyMinimizationAlgorithmOption(name_solver)
        with pytest.raises(AttributeError):
            actual.mode_constraint = "physical"


class TestCvxpyMinimizationAlgorithm:
    def test_is_loss_sufficient(self):
        # is_loss_sufficient = default(False)
        actual = CvxpyMinimizationAlgorithm()
        assert actual.is_loss_sufficient() == False

        # is_loss_sufficient = True
        actual = CvxpyMinimizationAlgorithm()
        loss = CvxpyRelativeEntropy()
        actual.set_from_loss(loss)
        assert actual.is_loss_sufficient() == True

    def test_is_loss_and_option_sufficient(self):
        # case: set_from_option is not called
        actual = CvxpyMinimizationAlgorithm()
        assert actual.is_loss_and_option_sufficient() == True

        # case: loss = CvxpyRelativeEntropy, algo_option = None
        actual = CvxpyMinimizationAlgorithm()
        loss = CvxpyRelativeEntropy()
        actual.set_from_loss(loss)
        assert actual.is_loss_and_option_sufficient() == True

        # case: loss = CvxpyRelativeEntropy, loss = None
        actual = CvxpyMinimizationAlgorithm()
        algo_option = CvxpyMinimizationAlgorithmOption(
            name_solver="scs", mode_constraint="physical"
        )
        actual.set_from_option(algo_option)
        assert actual.is_loss_and_option_sufficient() == True

        # case: loss = CvxpyRelativeEntropy, algo_option = {name_solver="scs", mode_constraint="physical"}
        actual = CvxpyMinimizationAlgorithm()
        algo_option = CvxpyMinimizationAlgorithmOption(
            name_solver="scs", mode_constraint="physical"
        )
        actual.set_from_option(algo_option)
        loss = CvxpyRelativeEntropy()
        actual.set_from_loss(loss)
        assert actual.is_loss_and_option_sufficient() == True

        # case: loss = CvxpyRelativeEntropy, algo_option = {name_solver="cvxopt", mode_constraint="physical"}
        actual = CvxpyMinimizationAlgorithm()
        algo_option = CvxpyMinimizationAlgorithmOption(
            name_solver="cvxopt", mode_constraint="physical"
        )
        actual.set_from_option(algo_option)
        loss = CvxpyRelativeEntropy()
        actual.set_from_loss(loss)
        assert actual.is_loss_and_option_sufficient() == False

        # case: loss = CvxpyRelativeEntropy, algo_option = {name_solver="scs", mode_constraint="unconstraint"}
        actual = CvxpyMinimizationAlgorithm()
        algo_option = CvxpyMinimizationAlgorithmOption(
            name_solver="scs", mode_constraint="unconstraint"
        )
        actual.set_from_option(algo_option)
        loss = CvxpyRelativeEntropy()
        actual.set_from_loss(loss)
        assert actual.is_loss_and_option_sufficient() == False

    @pytest.mark.cvxpy
    def test_optimize(self):
        # case: loss = None
        algo = CvxpyMinimizationAlgorithm()
        algo_option = CvxpyMinimizationAlgorithmOption(
            name_solver="scs", mode_constraint="physical"
        )
        algo.set_from_option(algo_option)
        with pytest.raises(ValueError):
            algo.optimize()

        ### case: algo_option = None
        algo = CvxpyMinimizationAlgorithm()
        loss = CvxpyRelativeEntropy()
        algo.set_from_loss(loss)
        with pytest.raises(ValueError):
            algo.optimize()

        ### case: loss = CvxpyRelativeEntropy, algo_option = {name_solver="scs", mode_constraint="physical"}
        algo = CvxpyMinimizationAlgorithm()
        # algo_option
        algo_option = CvxpyMinimizationAlgorithmOption(
            name_solver="scs", mode_constraint="physical"
        )
        algo.set_from_option(algo_option)
        # loss
        loss = CvxpyRelativeEntropy()
        sqt, c_sys = get_test_data_qst()
        loss.set_standard_qtomography(sqt)
        true_object = generate_state_from_name(c_sys, "z0")
        prob_dists = sqt.calc_prob_dists(true_object)
        empi_dists = [(10, prob_dist) for prob_dist in prob_dists]
        loss.set_prob_dists_data_from_empi_dists(empi_dists)
        algo.set_from_loss(loss)
        # Act
        actual = algo.optimize()
        # Assert
        expected = np.array([0, 0, 1 / np.sqrt(2)])
        npt.assert_almost_equal(actual.value, expected, decimal=9)
        npt.assert_almost_equal(actual.variable_value, expected, decimal=9)
        assert type(actual.loss_value) == np.float64
        assert type(actual.computation_time) == float

        ### case: loss = CvxpyRelativeEntropy, algo_option = {name_solver="mosek", mode_constraint="physical"}
        algo = CvxpyMinimizationAlgorithm()
        # algo_option
        algo_option = CvxpyMinimizationAlgorithmOption(
            name_solver="mosek", mode_constraint="physical"
        )
        algo.set_from_option(algo_option)
        # loss
        loss = CvxpyRelativeEntropy()
        sqt, c_sys = get_test_data_qst()
        loss.set_standard_qtomography(sqt)
        true_object = generate_state_from_name(c_sys, "z0")
        prob_dists = sqt.calc_prob_dists(true_object)
        empi_dists = [(10, prob_dist) for prob_dist in prob_dists]
        loss.set_prob_dists_data_from_empi_dists(empi_dists)
        algo.set_from_loss(loss)
        # Act
        actual = algo.optimize()
        # Assert
        expected = np.array([0, 0, 1 / np.sqrt(2)])
        npt.assert_almost_equal(actual.value, expected, decimal=9)
        npt.assert_almost_equal(actual.variable_value, expected, decimal=9)
        assert type(actual.loss_value) == np.float64
        assert type(actual.computation_time) == float

        ### case: loss = CvxpyUniformSquaredError, algo_option = {name_solver="cvxopt", mode_constraint="physical"}
        algo = CvxpyMinimizationAlgorithm()
        # algo_option
        algo_option = CvxpyMinimizationAlgorithmOption(
            name_solver="cvxopt", mode_constraint="physical"
        )
        algo.set_from_option(algo_option)
        # loss
        loss = CvxpyUniformSquaredError()
        sqt, c_sys = get_test_data_qst()
        loss.set_standard_qtomography(sqt)
        true_object = generate_state_from_name(c_sys, "z0")
        prob_dists = sqt.calc_prob_dists(true_object)
        empi_dists = [(10, prob_dist) for prob_dist in prob_dists]
        loss.set_prob_dists_data_from_empi_dists(empi_dists)
        algo.set_from_loss(loss)
        # Act
        actual = algo.optimize()
        # Assert
        expected = np.array([0, 0, 1 / np.sqrt(2)])
        npt.assert_almost_equal(actual.value, expected, decimal=3)
        npt.assert_almost_equal(actual.variable_value, expected, decimal=3)
        assert type(actual.loss_value) == np.float64
        assert type(actual.computation_time) == float

        ### case: loss = CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm, algo_option = {name_solver="mosek", mode_constraint="unconstraint"}
        algo = CvxpyMinimizationAlgorithm()
        # algo_option
        algo_option = CvxpyMinimizationAlgorithmOption(
            name_solver="mosek", mode_constraint="unconstraint"
        )
        algo.set_from_option(algo_option)
        # loss
        loss = CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm()
        sqt, c_sys = get_test_data_qst()
        loss.set_standard_qtomography(sqt)
        true_object = generate_state_from_name(c_sys, "z0")
        prob_dists = sqt.calc_prob_dists(0.9 * true_object)
        empi_dists = [(10, prob_dist) for prob_dist in prob_dists]
        loss.set_prob_dists_data_from_empi_dists(empi_dists)
        algo.set_from_loss(loss)
        # Act
        actual = algo.optimize()
        # Assert
        assert type(actual.value) == np.ndarray
        assert type(actual.variable_value) == np.ndarray
        assert type(actual.loss_value) == np.float64
        assert type(actual.computation_time) == float
