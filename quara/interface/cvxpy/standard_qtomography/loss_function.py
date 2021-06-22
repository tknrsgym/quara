from abc import abstractmethod
from typing import List, Tuple, Union
import numpy as np
import cvxpy as cp

# quara
from quara.loss_function.loss_function import (
    LossFunction,
    LossFunctionOption,
)
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.protocol.qtomography.standard.preprocessing import (
    # StandardQTomographyPreprocessing,
    is_prob_dist,
    extract_nums_from_empi_dists,
    extract_prob_dists_from_empi_dists,
    calc_total_num,
    calc_num_ratios,
    type_standard_qtomography,
)

# cvxpy
import cvxpy as cp
from cvxpy.expressions.variable import Variable as CvxpyVariable
from cvxpy.expressions.expression import Expression as CvxpyExpression


def get_valid_mode_form() -> List[str]:
    l = ["sum"]
    l.append("quadratic")
    return l


class CvxpyLossFunctionOption(LossFunctionOption):
    def __init__(self, mode_form: str = "sum", eps_prob_zero: np.float64 = 1e-12):
        assert mode_form in get_valid_mode_form()
        self._mode_form = mode_form
        self._eps_prob_zero = eps_prob_zero

    @property
    def mode_form(self) -> str:
        return self._mode_form

    @property
    def eps_prob_zero(self) -> np.float64:
        return self._eps_prob_zero


class CvxpyLossFunction(LossFunction):
    def __init__(self, num_var: int = None):
        super().__init__(num_var)
        self._eps_prob_zero = 1e-12
        self._nums_data = None
        self._num_data_total = None
        self._num_data_ratios = None
        self._prob_dists_data = None
        self._sqt = None
        self._option = None
        self._mode_form = None

    @property
    def eps_prob_zero(self) -> np.float64:
        return self._eps_prob_zero

    def set_from_option(self, option: CvxpyLossFunctionOption) -> None:
        self._mode_form = option.mode_form
        self._eps_prob_zero = option.eps_prob_zero
        self._option = option

    @property
    def sqt(self) -> StandardQTomography:
        return self._sqt

    def set_standard_qtomography(self, sqt: StandardQTomography):
        assert sqt.on_para_eq_constraint
        self._sqt = sqt
        self._type_estimate = type_standard_qtomography(sqt)
        self._on_value = True

    @property
    def type_estimate(self) -> str:
        return self._type_estimate

    @property
    def mode_form(self) -> str:
        return self._mode_form

    @property
    def nums_data(self) -> List[int]:
        return self._nums_data

    @property
    def num_data_total(self) -> int:
        return self._num_data_total

    @property
    def num_data_ratios(self) -> List[np.float64]:
        return self._num_data_ratios

    def calc_average_num_data(self) -> np.float64:
        arr = np.array(self.nums_data)
        num_ave = np.mean(arr)
        return num_ave

    @property
    def prob_dists_data(self) -> List[np.ndarray]:
        return self._prob_dists_data

    def set_prob_dists_data(self, ps: List[np.ndarray]):
        q = ps
        for i, pi in enumerate(ps):
            assert is_prob_dist(pi, self.eps_prob_zero)
            for x, pi_x in enumerate(pi):
                if pi_x < self.eps_prob_zero:
                    q[i][x] = 0
                elif pi_x > 1.0:
                    q[i][x] = 1
        self._prob_dists_data = q

    def erase_prob_dists_data(self):
        self._prob_dists_data = None
        self._on_prob_dists_data = False

    def calc_prob_model(
        self, i: int, x: int, var: Union[np.ndarray, CvxpyVariable]
    ) -> Union[np.float64, CvxpyExpression]:
        pi_x = (
            self.sqt.get_coeffs_1st_mat(i)[x] @ var + self.sqt.get_coeffs_0th_vec(i)[x]
        )
        # if pi_x < self.eps_prob_zero:
        #    pi_x = 0.0
        return pi_x

    def set_prob_dists_data_from_empi_dists(
        self, empi_dists: List[Tuple[int, np.ndarray]]
    ):
        self._nums_data = extract_nums_from_empi_dists(empi_dists)
        self._num_data_total = calc_total_num(self.nums_data)
        self._num_data_ratios = calc_num_ratios(self.nums_data)
        q = extract_prob_dists_from_empi_dists(empi_dists)
        self.set_prob_dists_data(q)

    @property
    def composite_system(self):
        t = self.type_estimate
        if t == "state":
            c_sys = self.sqt._experiment.povms[0]._composite_system
        elif t == "povm":
            c_sys = self.sqt._experiment.states[0]._composite_system
        elif t == "gate":
            c_sys = self.sqt._experiment.povms[0]._composite_system
        return c_sys

    def dim_system(self):
        return self.composite_system.dim

    def num_outcomes_estimate(self):
        assert self.type_estimate == "povm"
        num = self.sqt.num_outcomes(schedule_index=0)
        return num

    def value(self, var: np.ndarray) -> np.float64:
        if self.option.mode_form == "sum":
            val = self.value_form_sum(var)
        elif self.option.mode_form == "quadratic":
            val = self.value_form_quadratic(var)
        return val

    @abstractmethod
    def value_form_sum(self, var: np.ndarray) -> np.float64:
        raise NotImplementedError

    def value_form_quadratic(self, var: np.ndarray) -> np.float64:
        # 1/2 var * matA * var - vecB * var + c
        matA = 0.50 * self.matQ()
        vecB = self.vecR()
        c = self.scalS()
        val = var @ matA @ var - vecB @ var + c
        return val

    def value_cvxpy(self, var: CvxpyVariable) -> CvxpyExpression:
        if self.option.mode_form == "sum":
            expr = self.value_cvxpy_form_sum(var)
        elif self.option.mode_form == "quadratic":
            expr = self.value_cvxpy_form_quadratic(var)
        return expr

    @abstractmethod
    def value_cvxpy_form_sum(self, var: CvxpyVariable) -> CvxpyExpression:
        raise NotImplementedError

    def value_cvxpy_form_quadratic(self, var: CvxpyVariable) -> CvxpyExpression:
        # 1/2 var * matA * var - vecB * var + c
        matA = 0.50 * self.matQ()
        vecB = self.vecR()
        c = self.scalS()
        expr = cp.quad_form(var, matA) - vecB @ var + c
        return expr

    @abstractmethod
    def matQ(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def vecR(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def scalS(self) -> np.float64:
        raise NotImplementedError


class CvxpyRelativeEntropy(CvxpyLossFunction):
    def __init__(self):
        """Constructor"""
        super().__init__()

    def is_option_sufficient(self) -> bool:
        res = False
        if self.option.mode_form == "sum":
            res = True
        return res

    def value_form_sum(self, var: np.ndarray) -> np.float64:
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
        assert self.prob_dists_data != None

        const = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                if qi_j > self.eps_prob_zero:
                    t += qi_j * np.log(qi_j)
            ci = self.num_data_ratios[i]
            const += ci * t

        value = const
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                if qi_j > self.eps_prob_zero:
                    pi_j = self.calc_prob_model(i, j, var)
                    print(i, j, pi_j)
                    t -= qi_j * np.log(pi_j)
            ci = self.num_data_ratios[i]
        value += ci * t

        return value

    def value_cvxpy_form_sum(self, var: CvxpyVariable) -> CvxpyExpression:
        """returns the value of the loss function in the form of Expression of CVXPY.

        Parameters
        ----------
        var : np.ndarray
            np.ndarray of variables.

        Returns
        -------
        CvxpyExpression
            the value of the loss function in the form of Expression of CVXPY.
        """
        assert self.prob_dists_data != None

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

    def value_form_sum(self, var: np.ndarray) -> np.float64:
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
        assert self.prob_dists_data != None

        const = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                if qi_j > self.eps_prob_zero:
                    t += qi_j * np.log(qi_j)
            ci = self.num_data_ratios[i]
            const += ci * t

        value = const
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                if qi_j > self.eps_prob_zero:
                    pi_j = self.calc_prob_model(i, j, var)
                    print(i, j, pi_j)
                    t -= qi_j * np.log(pi_j)
            ci = self.num_data_ratios[i]
        value += ci * t

        return value

    def value_cvxpy_form_sum(self, var: CvxpyVariable) -> CvxpyExpression:
        """returns the value of the loss function in the form of Expression of CVXPY.

        Parameters
        ----------
        var : np.ndarray
            np.ndarray of variables.

        Returns
        -------
        CvxpyExpression
            the value of the loss function in the form of Expression of CVXPY.
        """
        assert self.prob_dists_data != None
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

    def matQ(self) -> np.ndarray:
        mat = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                a = self.sqt.get_coeffs_1st_mat(i)[j]
                t += np.outer(a, a)
            ci = self.num_data_ratios[i]
            mat += ci * t
        return mat

    def vecR(self) -> np.ndarray:
        vec = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                a = self.sqt.get_coeffs_1st_mat(i)[j]
                t += (
                    self.prob_dists_data[i][j] - self.sqt.get_coeffs_0th_vec(i)[j]
                ) * a
            ci = self.num_data_ratios[i]
            vec += ci * t
        return vec

    def scalS(self) -> np.float64:
        s = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                t += 0.50 * (
                    self.prob_dists_data[i][j] - self.sqt.get_coeffs_0th_vec(i)[j]
                )
            ci = self.num_data_ratios[i]
            s += ci * t
        return s


class CvxpyApproximateRelativeEntropyWithoutZeroProbabilityTerm(CvxpyLossFunction):
    def __init__(self):
        super().__init__()

    def value_form_sum(self, var: np.ndarray) -> np.float64:
        assert self.prob_dists_data != None
        value = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                pi_j = self.calc_prob_model(i, j, var)
                if qi_j > self.eps_prob_zero:
                    t += 0.50 * ((pi_j - qi_j) ** 2) / pi_j
                # else:
                #    t += pi_j
            ci = self.num_data_ratios[i]
            value += ci * t
        return value

    def value_cvxpy_form_sum(self, var: CvxpyVariable) -> CvxpyExpression:
        assert self.prob_dists_data != None
        expr = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                pi_j = self.calc_prob_model(i, j, var)
                if qi_j > self.eps_prob_zero:
                    t += 0.50 * cp.quad_over_lin(pi_j - qi_j, pi_j)
                # else:
                #    t += pi_j
            ci = self.num_data_ratios[i]
            expr += ci * t
        return expr

    def matQ(self) -> np.ndarray:
        mat = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                if qi_j > self.eps_prob_zero:
                    a = self.sqt.get_coeffs_1st_mat(i)[j]
                    t += np.outer(a, a) / qi_j
            ci = self.num_data_ratios[i]
            mat += ci * t
        return mat

    def vecR(self) -> np.ndarray:
        vec = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                if qi_j > self.eps_prob_zero:
                    a = self.sqt.get_coeffs_1st_mat(i)[j]
                    t += (1 - self.sqt.get_coeffs_0th_vec(i)[j] / qi_j) * a
            ci = self.num_data_ratios[i]
            vec += ci * t
        return vec

    def scalS(self) -> np.float64:
        s = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                if qi_j > self.eps_prob_zero:
                    t += 0.50 * ((qi_j - self.sqt.get_coeffs_0th_vec(i)[j]) ** 2) / qi_j
            ci = self.num_data_ratios[i]
            s += ci * t
        return s


class CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm(CvxpyLossFunction):
    def __init__(self):
        super().__init__()

    def value_form_sum(self, var: np.ndarray) -> np.float64:
        assert self.prob_dists_data != None
        value = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                pi_j = self.calc_prob_model(i, j, var)
                if qi_j > self.eps_prob_zero:
                    t += 0.50 * ((pi_j - qi_j) ** 2) / pi_j
                else:
                    t += pi_j
            ci = self.num_data_ratios[i]
            value += ci * t
        return value

    def value_cvxpy_form_sum(self, var: CvxpyVariable) -> CvxpyExpression:
        assert self.prob_dists_data != None
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

    def matQ(self) -> np.ndarray:
        mat = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                if qi_j > self.eps_prob_zero:
                    a = self.sqt.get_coeffs_1st_mat(i)[j]
                    t += np.outer(a, a) / qi_j
            ci = self.num_data_ratios[i]
            mat += ci * t
        return mat

    def vecR(self) -> np.ndarray:
        vec = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                a = self.sqt.get_coeffs_1st_mat(i)[j]
                if qi_j > self.eps_prob_zero:
                    t += (1 - self.sqt.get_coeffs_0th_vec(i)[j] / qi_j) * a
                else:
                    t -= a
            ci = self.num_data_ratios[i]
            vec += ci * t
        return vec

    def scalS(self) -> np.float64:
        s = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                bi_j = self.sqt.get_coeffs_0th_vec(i)[j]
                if qi_j > self.eps_prob_zero:
                    t += 0.50 * ((qi_j - bi_j) ** 2) / qi_j
                else:
                    t += bi_j
            ci = self.num_data_ratios[i]
            s += ci * t
        return s


class CvxpyApproximateRelativeEntropyWithZeroProbabilityTermSquared(CvxpyLossFunction):
    def __init__(self):
        super().__init__()

    def value_form_sum(self, var: np.ndarray) -> np.float64:
        assert self.prob_dists_data != None
        value = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                pi_j = self.calc_prob_model(i, j, var)
                if qi_j > self.eps_prob_zero:
                    t += 0.50 * ((pi_j - qi_j) ** 2) / pi_j
                else:
                    t += pi_j ** 2
            ci = self.num_data_ratios[i]
            value += ci * t
        return value

    def value_cvxpy_form_sum(self, var: CvxpyVariable) -> CvxpyExpression:
        assert self.prob_dists_data != None
        expr = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                pi_j = self.calc_prob_model(i, j, var)
                if qi_j > self.eps_prob_zero:
                    t += 0.50 * cp.quad_over_lin(pi_j - qi_j, pi_j)
                else:
                    t += cp.quad_over_lin(pi_j, 1)
            ci = self.num_data_ratios[i]
            expr += ci * t
        return expr

    def matQ(self) -> np.ndarray:
        mat = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                a = self.sqt.get_coeffs_1st_mat(i)[j]
                if qi_j > self.eps_prob_zero:
                    t += np.outer(a, a) / qi_j
                else:
                    t += 2.0 * np.outer(a, a)
            ci = self.num_data_ratios[i]
            mat += ci * t
        return mat

    def vecR(self) -> np.ndarray:
        vec = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                a = self.sqt.get_coeffs_1st_mat(i)[j]
                b = self.sqt.get_coeffs_0th_vec(i)[j]
                if qi_j > self.eps_prob_zero:
                    t += (1 - b / qi_j) * a
                else:
                    t += -2.0 * b * a
            ci = self.num_data_ratios[i]
            vec += ci * t
        return vec

    def scalS(self) -> np.float64:
        s = 0.0
        for i in range(self.sqt.num_schedules):
            t = 0.0
            for j in range(self.sqt.num_outcomes(i)):
                qi_j = self.prob_dists_data[i][j]
                bi_j = self.sqt.get_coeffs_0th_vec(i)[j]
                if qi_j > self.eps_prob_zero:
                    t += 0.50 * ((qi_j - bi_j) ** 2) / qi_j
                else:
                    t += bi_j ** 2
            ci = self.num_data_ratios[i]
            s += ci * t
        return s
