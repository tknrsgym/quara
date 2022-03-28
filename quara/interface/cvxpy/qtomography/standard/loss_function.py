from abc import abstractmethod
from typing import List, Tuple, Union

import numpy as np
import cvxpy as cp
from cvxpy.expressions.variable import Variable as CvxpyVariable
from cvxpy.expressions.expression import Expression as CvxpyExpression

from quara.objects.composite_system import CompositeSystem
from quara.loss_function.loss_function import (
    LossFunction,
    LossFunctionOption,
)
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.protocol.qtomography.standard.preprocessing import (
    extract_nums_from_empi_dists,
    extract_prob_dists_from_empi_dists,
    calc_total_num,
    calc_num_ratios,
    type_standard_qtomography,
)
from quara.math.probability import validate_prob_dist


class CvxpyLossFunctionOption(LossFunctionOption):
    def __init__(self, eps_prob_zero: np.float64 = 1e-12):
        """Constructor

        Parameters
        ----------
        eps_prob_zero : np.float64, optional
            Threshold to truncate probability, by default 1e-12
        """
        super().__init__()
        self._eps_prob_zero = eps_prob_zero

    @property
    def eps_prob_zero(self) -> np.float64:
        """returns threshold to truncate probability

        Returns
        -------
        np.float64
            Threshold to truncate probability
        """
        return self._eps_prob_zero


class CvxpyLossFunction(LossFunction):
    def __init__(self, num_var: int = None):
        """Constructor

        Parameters
        ----------
        num_var : int, optional
            number of variables, by default None
        """
        super().__init__(num_var)
        self._on_value = False
        self._on_value_cvxpy: bool = False
        self._eps_prob_zero = 1e-12
        self._nums_data = None
        self._num_data_total = None
        self._num_data_ratios = None
        self._prob_dists_data = None
        self._on_prob_dists_data = False
        self._sqt = None
        self._type_estimate = None
        self._option = None

    @property
    def on_value_cvxpy(self) -> bool:
        """returns whether or not to support value_cvxpy.

        Returns
        -------
        bool
            whether or not to support value_cvxpy.
        """
        return self._on_value_cvxpy

    @property
    def eps_prob_zero(self) -> np.float64:
        """returns threshold to truncate probability

        Returns
        -------
        np.float64
            Threshold to truncate probability
        """
        return self._eps_prob_zero

    def set_from_option(self, option: CvxpyLossFunctionOption) -> None:
        """sets option from CvxpyLossFunctionOption.

        Parameters
        ----------
        option : CvxpyLossFunctionOption
            option to set.
        """
        self._eps_prob_zero = option.eps_prob_zero
        self._option = option

    @property
    def sqt(self) -> StandardQTomography:
        """returns StandardQTomography for settings of loss function.

        Returns
        -------
        StandardQTomography
            StandardQTomography for settings of loss function.
        """
        return self._sqt

    def set_standard_qtomography(self, sqt: StandardQTomography) -> None:
        """sets StandardQTomography for settings of loss function.

        Parameters
        ----------
        sqt : StandardQTomography
            StandardQTomography for settings of loss function.

        Raises
        ------
        ValueError
            on_para_eq_constraint of StandardQTomography is False.
        """
        if sqt.on_para_eq_constraint == False:
            raise ValueError(
                f"on_para_eq_constraint of StandardQTomography must be True."
            )
        self._sqt = sqt
        self._type_estimate = type_standard_qtomography(sqt)
        self._on_value = True
        self._num_var = sqt.num_variables

    @property
    def type_estimate(self) -> str:
        """returns Type of QOperation. "state", "povm", "gate", or "mprocess".

        type_estimate is set in `set_standard_qtomography` function.

        Returns
        -------
        str
            Type of QOperation.
        """
        return self._type_estimate

    @property
    def nums_data(self) -> List[int]:
        """returns numbers of data.

        nums_data is set in `set_prob_dists_data_from_empi_dists` function.

        Returns
        -------
        List[int]
            numbers of data.
        """
        return self._nums_data

    @property
    def num_data_total(self) -> int:
        """returns number of total of data(sum of data).

        num_data_total is set in `set_prob_dists_data_from_empi_dists` function.

        Returns
        -------
        int
            number of total of data.
        """
        return self._num_data_total

    @property
    def num_data_ratios(self) -> List[np.float64]:
        """returns ratios of data.

        num_data_ratios is set in `set_prob_dists_data_from_empi_dists` function.

        Returns
        -------
        List[np.float64]
            ratios of data.
        """
        return self._num_data_ratios

    def calc_average_num_data(self) -> np.float64:
        """calculates average of numbers of data.

        Returns
        -------
        np.float64
            average of numbers of data.
        """
        arr = np.array(self.nums_data)
        num_ave = np.mean(arr)
        return num_ave

    @property
    def prob_dists_data(self) -> List[np.ndarray]:
        """returns probability distributions of data.

        Returns
        -------
        List[np.ndarray]
            probability distributions of data.
        """
        return self._prob_dists_data

    @property
    def on_prob_dists_data(self) -> bool:
        """returns whether prob_dists_data is set.

        Returns
        -------
        bool
            whether prob_dists_data is set.
        """
        return self._on_prob_dists_data

    def set_prob_dists_data(self, ps: List[np.ndarray]) -> None:
        """sets probability distributions of data.

        Parameters
        ----------
        ps : List[np.ndarray]
            probability distributions of data.
        """
        q = ps
        for i, pi in enumerate(ps):
            validate_prob_dist(pi, eps=self.eps_prob_zero)
            for x, pi_x in enumerate(pi):
                if pi_x < self.eps_prob_zero:
                    q[i][x] = 0
                elif pi_x > 1.0:
                    q[i][x] = 1
        self._prob_dists_data = q
        self._on_prob_dists_data = True

    def erase_prob_dists_data(self) -> None:
        """erases probability distributions of data."""
        self._prob_dists_data = None
        self._on_prob_dists_data = False

    def calc_prob_model(
        self, i: int, x: int, var: Union[np.ndarray, CvxpyVariable]
    ) -> Union[np.float64, CvxpyExpression]:
        """calculates probability model.

        Parameters
        ----------
        i : int
            schedule index.
        x : int
            measurement outcome index.
        var : Union[np.ndarray, CvxpyVariable]
            Variable of CVXPY

        Returns
        -------
        Union[np.float64, CvxpyExpression]
            probability model.
        """
        pi_x = (
            self.sqt.get_coeffs_1st_mat(i)[x] @ var + self.sqt.get_coeffs_0th_vec(i)[x]
        )
        return pi_x

    def set_prob_dists_data_from_empi_dists(
        self, empi_dists: List[Tuple[int, np.ndarray]]
    ) -> None:
        """sets probability distributions of data from empirical distributions.

        Parameters
        ----------
        empi_dists : List[Tuple[int, np.ndarray]]
            empirical distributions.
        """
        self._nums_data = extract_nums_from_empi_dists(empi_dists)
        self._num_data_total = calc_total_num(self.nums_data)
        self._num_data_ratios = calc_num_ratios(self.nums_data)
        q = extract_prob_dists_from_empi_dists(empi_dists)
        self.set_prob_dists_data(q)

    @property
    def composite_system(self) -> CompositeSystem:
        """returns CompositeSystem of type of estimate.

        composite_system is set in `set_standard_qtomography` function.

        Returns
        -------
        CompositeSystem
            CompositeSystem of type of estimate
        """
        t = self.type_estimate
        if t == "state":
            c_sys = self.sqt._experiment.povms[0]._composite_system
        elif t == "povm":
            c_sys = self.sqt._experiment.states[0]._composite_system
        elif t == "gate":
            c_sys = self.sqt._experiment.povms[0]._composite_system
        elif t == "mprocess":
            c_sys = self.sqt._experiment.povms[0]._composite_system
        else:
            c_sys = None
        return c_sys

    def dim_system(self) -> int:
        """returns dimension of CompositeSystem.

        dim_system is set in `set_standard_qtomography` function.

        Returns
        -------
        int
            dimension of CompositeSystem.
        """
        if self.composite_system is None:
            return None
        else:
            return self.composite_system.dim

    def num_outcomes_estimate(self) -> int:
        """returns number of outcomes of estimate.

        num_outcomes_estimate is set in `set_standard_qtomography` function.

        Returns
        -------
        int
            number of outcomes of estimate.

        Raises
        ------
        ValueError
            num_outcomes_estimate of StandardQTomography is neither 'povm' nor 'mprocess'.
        """
        if self.type_estimate == "povm" or self.type_estimate == "mprocess":
            return self.sqt.num_outcomes_estimate
        else:
            raise ValueError(
                f"num_outcomes_estimate function is supported if type_estimate = 'povm' or 'mprocess'. type_estimate={self.type_estimate}"
            )

    def value(self, var: np.ndarray) -> np.float64:
        """returns the value of the loss function.

        Parameters
        ----------
        var : np.ndarray
            np.ndarray of variables.

        Returns
        -------
        np.float64
            the value of the loss function.
        """
        cvxvar = CvxpyVariable(len(var))
        cvxvar.value = var
        cvxval = self.value_cvxpy(cvxvar)
        return cvxval.value

    @abstractmethod
    def value_cvxpy(self, var: CvxpyVariable) -> CvxpyExpression:
        """returns CvxpyExpression of the loss function.

        Parameters
        ----------
        var : CvxpyVariable
            Variable of CVXPY

        Returns
        -------
        CvxpyExpression
            CvxpyExpression of the loss function.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError


class CvxpyRelativeEntropy(CvxpyLossFunction):
    def __init__(self):
        """Constructor"""
        super().__init__()
        self._on_value = True
        self._on_value_cvxpy = True

    def is_option_sufficient(self) -> bool:
        return True

    def value_cvxpy(self, var: CvxpyVariable) -> CvxpyExpression:
        """returns the value of the loss function in the form of Expression of CVXPY.

        Parameters
        ----------
        var : np.ndarray
            np.ndarray of variables.

        Returns
        -------
        CvxpyExpression
            the value of the loss function in the form of Expression of CVXPY.

        Raises
        ------
        ValueError
            prob_dists_data is not set.
        """
        if self.on_prob_dists_data is False:
            raise ValueError(f"prob_dists_data is not set.")

        const = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                if qi_j > self.eps_prob_zero:
                    t += qi_j * np.log(qi_j)
            ci = self.num_data_ratios[i]
            const += ci * t

        expr = const
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                if qi_j > self.eps_prob_zero:
                    pi_j = self.calc_prob_model(i, j, var)
                    t -= qi_j * cp.log(pi_j)
            ci = self.num_data_ratios[i]
            expr += ci * t

        return expr


class CvxpyUniformSquaredError(CvxpyLossFunction):
    def __init__(self):
        super().__init__()
        self._on_value = True
        self._on_value_cvxpy = True

    def value_cvxpy(self, var: CvxpyVariable) -> CvxpyExpression:
        """returns the value of the loss function in the form of Expression of CVXPY.

        Parameters
        ----------
        var : np.ndarray
            np.ndarray of variables.

        Returns
        -------
        CvxpyExpression
            the value of the loss function in the form of Expression of CVXPY.

        Raises
        ------
        ValueError
            prob_dists_data is not set.
        """
        if self.on_prob_dists_data is False:
            raise ValueError(f"prob_dists_data is not set.")

        expr = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                pi_j = self.calc_prob_model(i, j, var)
                t += cp.quad_over_lin(pi_j - qi_j, 1)
            ci = self.num_data_ratios[i]
            expr += ci * t
        return expr


class CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm(CvxpyLossFunction):
    def __init__(self):
        super().__init__()
        self._on_value = True
        self._on_value_cvxpy = True

    def value_cvxpy(self, var: CvxpyVariable) -> CvxpyExpression:
        """returns the value of the loss function in the form of Expression of CVXPY.

        Parameters
        ----------
        var : np.ndarray
            np.ndarray of variables.

        Returns
        -------
        CvxpyExpression
            the value of the loss function in the form of Expression of CVXPY.

        Raises
        ------
        ValueError
            prob_dists_data is not set.
        """
        if self.on_prob_dists_data is False:
            raise ValueError(f"prob_dists_data is not set.")

        expr = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                pi_j = self.calc_prob_model(i, j, var)
                if qi_j > self.eps_prob_zero:
                    t += 0.50 * cp.quad_over_lin(pi_j - qi_j, pi_j)
                else:
                    t += pi_j
            ci = self.num_data_ratios[i]
            expr += ci * t
        return expr
