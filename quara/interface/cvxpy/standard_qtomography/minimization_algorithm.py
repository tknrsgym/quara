from typing import List
import numpy as np
import time

# quara
from quara.minimization_algorithm.minimization_algorithm import (
    MinimizationAlgorithm,
    MinimizationAlgorithmOption,
    MinimizationResult,
)
from quara.protocol.qtomography.standard.preprocessing import (
    type_standard_qtomography,
)
from quara.interface.cvxpy.conversion import (
    generate_cvxpy_variable,
    generate_cvxpy_constraints_from_cvxpy_variable,
)
from quara.interface.cvxpy.standard_qtomography.loss_function import (
    CvxpyLossFunction,
    CvxpyLossFunctionOption,
    CvxpyRelativeEntropy,
)

# cvxpy
import cvxpy as cp


def get_valid_names_solver() -> List[str]:
    l = ["scs"]
    l.append("cvxopt")
    l.append("mosek")
    return l


def get_valid_modes_constraints() -> List[str]:
    l = ["physical"]
    l.append("physical_and_zero_probability_equation_satisfied")
    return l


class CvxpyMinimizationAlgorithmOption(MinimizationAlgorithmOption):
    def __init__(
        self,
        name_solver: str,
        verbose: bool = False,
        eps_tol: np.float64 = 1e-8,
        mode_constraint: str = "physical",
    ):
        assert name_solver in get_valid_names_solver()
        self._name_solver = name_solver
        self._verbose = verbose
        assert eps_tol >= 0
        self._eps_tol = eps_tol
        assert mode_constraint in get_valid_modes_constraints()
        self._mode_constraint = mode_constraint

    @property
    def name_solver(self):
        return self._name_solver

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def verbose(self):
        return self._verbose

    @property
    def eps_tol(self):
        return self._eps_tol

    @property
    def mode_constraint(self):
        return self._mode_constraint


class CvxpyMinimizationAlgorithm(MinimizationAlgorithm):
    def __init__(self):
        super().__init__()

    def is_loss_sufficient(self) -> bool:
        res = False
        if self.loss.on_value:
            res = True
        return res

    def is_loss_and_option_sufficient(self) -> bool:
        res = True
        if (
            type(self.loss) == CvxpyRelativeEntropy
            and self.option.name_solver == "cvxopt"
        ):
            res = False
        return res

    def optimize(
        self,
    ) -> MinimizationResult:
        assert self.loss != None

        time_start = time.time()

        # Generation of CVXPY objects
        t = self.loss.type_estimate
        dim = self.loss.dim_system()
        c_sys = self.loss.composite_system
        if t == "state" or t == "gate":
            var = generate_cvxpy_variable(t, dim)
            constraints = generate_cvxpy_constraints_from_cvxpy_variable(c_sys, t, var)
        elif t == "povm":
            num_outcomes = self.loss.num_outcomes_estimate()
            var = generate_cvxpy_variable(t, dim, num_outcomes)
            constraints = generate_cvxpy_constraints_from_cvxpy_variable(
                c_sys, t, var, num_outcomes
            )
        if (
            self.option.mode_constraint
            == "physical_and_zero_probability_equation_satisfied"
        ):
            for i, prob_dist in enumerate(self.loss.prob_dists_data):
                for x, q in enumerate(prob_dist):
                    if q < self.loss.eps_prob_zero:
                        pi_x = self.loss.calc_prob_model(i, x, var)
                        constraints.append(pi_x == 0)
        objective = cp.Minimize(self.loss.value_cvxpy(var))
        problem = cp.Problem(objective, constraints)

        # Execute Numerical Optimization
        name_solver = self.option.name_solver
        verbose = self.option.verbose
        eps_tol = self.option.eps_tol
        if name_solver == "scs":
            problem.solve(solver=cp.SCS, verbose=verbose, eps=eps_tol)
        elif name_solver == "mosek":
            params = {"MSK_DPAR_INTPNT_CO_TOL_DFEAS": eps_tol}
            problem.solve(
                solver=cp.MOSEK,
                verbose=verbose,
                mosek_params=params,
            )
        elif name_solver == "cvxopt":
            problem.solve(solver=cp.CVXOPT, verbose=verbose)
        else:
            raise ValueError(f"name_solver is invalid!")

        time_elapsed = time.time() - time_start

        result = CvxpyMinimizationResult(
            variable_value=var.value, loss_value=problem.value, comp_time=time_elapsed
        )

        return result


class CvxpyMinimizationResult(MinimizationResult):
    def __init__(
        self,
        variable_value: np.ndarray,
        loss_value: np.float64 = None,
        comp_time: float = None,
    ):
        super().__init__(value=variable_value, computation_time=comp_time)
        self._loss_value = loss_value

    @property
    def variable_value(self) -> np.ndarray:
        return super().value

    @property
    def loss_value(self) -> np.float64:
        return self._loss_value

    @property
    def computation_time(self) -> float:
        return super().computation_time
