from typing import List
import time

import numpy as np
import cvxpy as cp

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
    generate_cvxpy_constraints_from_cvxpy_variable_with_sparsity,
)
from quara.interface.cvxpy.qtomography.standard.loss_function import (
    CvxpyLossFunction,
    CvxpyLossFunctionOption,
    CvxpyRelativeEntropy,
)


def get_valid_names_solver() -> List[str]:
    """returns valid names of solver.

    Returns
    -------
    List[str]
        valid names of solver.
    """
    l = ["scs"]
    l.append("cvxopt")
    l.append("mosek")
    return l


def get_valid_modes_constraints() -> List[str]:
    """returns valid modes of constraints.

    Returns
    -------
    List[str]
        valid modes of constraints.
    """
    l = ["physical"]
    l.append("unconstraint")
    return l


class CvxpyMinimizationResult(MinimizationResult):
    def __init__(
        self,
        variable_value: np.ndarray,
        loss_value: np.float64 = None,
        comp_time: float = None,
    ):
        """Constructor

        Parameters
        ----------
        variable_value : np.ndarray
            the result of the minimization.
        loss_value : np.float64, optional
            loss for the minimization, by default None
        comp_time : float, optional
            computation time for the minimization, by default None
        """
        super().__init__(value=variable_value, computation_time=comp_time)
        self._loss_value = loss_value

    @property
    def variable_value(self) -> np.ndarray:
        """returns the result of the minimization.

        Returns
        -------
        np.ndarray
            the result of the minimization.
        """
        return self.value

    @property
    def loss_value(self) -> np.float64:
        """returns loss for the minimization.

        Returns
        -------
        np.float64
            loss for the minimization.
        """
        return self._loss_value


class CvxpyMinimizationAlgorithmOption(MinimizationAlgorithmOption):
    def __init__(
        self,
        name_solver: str,
        verbose: bool = False,
        eps_tol: np.float64 = 1e-8,
        mode_constraint: str = "physical",
    ):
        """Constructor

        Parameters
        ----------
        name_solver : str
            name of solver
        verbose : bool, optional
            verbose option of solver, by default False
        eps_tol : np.float64, optional
            eps option of solver, by default 1e-8
        mode_constraint : str, optional
            mode of constraint, by default "physical"

        Raises
        ------
        ValueError
            Unsupported name_solver is specified.
        ValueError
            eps_tol is negative.
        ValueError
            Unsupported mode_constraint is specified.
        """
        if name_solver not in get_valid_names_solver():
            raise ValueError(
                f"Unsupported name_solver is specified. name_solver={name_solver}"
            )
        self._name_solver: str = name_solver

        self._verbose: bool = verbose

        if eps_tol < 0:
            raise ValueError(f"eps_tol must be non-negative. eps_tol={eps_tol}")
        self._eps_tol: np.float64 = eps_tol

        if mode_constraint not in get_valid_modes_constraints():
            raise ValueError(
                f"Unsupported mode_constraint is specified. mode_constraint={mode_constraint}"
            )
        self._mode_constraint: str = mode_constraint

    @property
    def name_solver(self) -> str:
        """returns name of solver.

        Returns
        -------
        str
            name of solver.
        """
        return self._name_solver

    @property
    def verbose(self) -> bool:
        """returns verbose option of solver.

        Returns
        -------
        bool
            verbose option of solver.
        """
        return self._verbose

    @property
    def eps_tol(self) -> np.float64:
        """returns eps option of solver.

        Returns
        -------
        np.float64
            eps option of solver.
        """
        return self._eps_tol

    @property
    def mode_constraint(self) -> str:
        """returns mode of constraint.

        Returns
        -------
        str
            mode of constraint.
        """
        return self._mode_constraint


class CvxpyMinimizationAlgorithm(MinimizationAlgorithm):
    def __init__(self):
        super().__init__()

    def is_loss_sufficient(self) -> bool:
        res = False
        if self.loss != None and self.loss.on_value:
            res = True
        return res

    def is_loss_and_option_sufficient(self) -> bool:
        res = True
        if (
            type(self.loss) == CvxpyRelativeEntropy
            and self.option != None
            and self.option.name_solver == "cvxopt"
        ):
            res = False
        elif (
            type(self.loss) == CvxpyRelativeEntropy
            and self.option != None
            and self.option.mode_constraint == "unconstraint"
        ):
            res = False
        return res

    def optimize(
        self,
    ) -> MinimizationResult:
        if self.loss == None:
            raise ValueError(f"loss is not set.")
        if self.option == None:
            raise ValueError(f"algorithm option is not set.")

        time_start = time.time()

        # CVXPY variable
        t = self.loss.type_estimate
        dim = self.loss.dim_system()
        c_sys = self.loss.composite_system
        if t == "state" or t == "gate":
            num_outcomes = None
            var = generate_cvxpy_variable(t, dim)
        elif t == "povm" or t == "mprocess":
            num_outcomes = self.loss.num_outcomes_estimate()
            var = generate_cvxpy_variable(t, dim, num_outcomes)
        else:
            raise ValueError(f"type of estimate is invalid! type of estimate={t}")

        # CVXPY constraints
        if self.option.mode_constraint == "unconstraint":
            constraints = []
        elif self.option.mode_constraint == "physical":
            constraints = generate_cvxpy_constraints_from_cvxpy_variable_with_sparsity(
                c_sys, t, var, num_outcomes
            )
        else:
            raise ValueError(
                f"mode_constraint is invalid! mode_constraint={self.option.mode_constraint}"
            )

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
            raise ValueError(f"name_solver is invalid! name_solver={name_solver}")

        time_elapsed = time.time() - time_start

        result = CvxpyMinimizationResult(
            variable_value=var.value, loss_value=problem.value, comp_time=time_elapsed
        )

        return result
