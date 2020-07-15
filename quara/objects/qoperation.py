from abc import abstractmethod
import logging
from typing import Dict, List, Tuple, Union

import numpy as np

from quara.objects.composite_system import CompositeSystem

logger = logging.getLogger(__name__)


class QOperation:
    def __init__(
        self,
        c_sys: CompositeSystem,
        is_physicality_required: bool = True,
        is_estimation_object: bool = True,
        on_para_eq_constraint: bool = True,
        on_algo_eq_constraint: bool = True,
        on_algo_ineq_constraint: bool = True,
        eps_proj_physical: float = 10 ** (-4),
    ):
        """Constructor

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem of this QOperation.
        is_physicality_required : bool, optional
            whether this QOperation is physicality required, by default True
        is_estimation_object : bool, optional
            whether this QOperation is estimation object, by default True
        on_para_eq_constraint : bool, optional
            whether this QOperation is on parameter equality constraint, by default True
        on_algo_eq_constraint : bool, optional
            whether this QOperation is on algorithm equality constraint, by default True
        on_algo_ineq_constraint : bool, optional
            whether this QOperation is on algorithm inequality constraint, by default True
        eps_proj_physical : float, optional
            epsiron that is projection algorithm error threshold for being physical, by default 10**(-4)

        Raises
        ------
        ValueError
            ``eps_proj_physical`` is negative.
        """
        # Validation
        if eps_proj_physical < 0:
            raise ValueError("'eps_proj_physical' must be non-negative.")

        # Set
        self._composite_system: CompositeSystem = c_sys
        self._is_physicality_required = is_physicality_required
        self._is_estimation_object = is_estimation_object
        self._on_para_eq_constraint: bool = on_para_eq_constraint
        self._on_algo_eq_constraint: bool = on_algo_eq_constraint
        self._on_algo_ineq_constraint: bool = on_algo_ineq_constraint
        self._eps_proj_physical = eps_proj_physical

    @property
    def composite_system(self) -> CompositeSystem:  # read only
        """Property to get composite system.

        Returns
        -------
        CompositeSystem
            composite system.
        """
        return self._composite_system

    @property
    def is_physicality_required(self) -> bool:  # read only
        """returns whether this QOperation is physicality required.

        Returns
        -------
        bool
            whether this QOperation is physicality required.
        """
        return self._is_physicality_required

    @property
    def is_estimation_object(self) -> bool:  # read only
        """returns whether this QOperation is estimation object.

        Returns
        -------
        bool
            whether this QOperation is estimation object.
        """
        return self._is_estimation_object

    @property
    def on_para_eq_constraint(self) -> bool:  # read only
        """returns whether this QOperation is on parameter equality constraint.

        Returns
        -------
        bool
            whether this QOperation is on parameter equality constraint.
        """
        return self._on_para_eq_constraint

    @property
    def on_algo_eq_constraint(self) -> bool:  # read only
        """returns whether this QOperation is on algorithm equality constraint.

        Returns
        -------
        bool
            whether this QOperation is on algorithm equality constraint.
        """
        return self._on_algo_eq_constraint

    @property
    def on_algo_ineq_constraint(self) -> bool:  # read only
        """returns whether this QOperation is on algorithm inequality constraint.

        Returns
        -------
        bool
            whether this QOperation is on algorithm inequality constraint.
        """
        return self._on_algo_ineq_constraint

    @property
    def eps_proj_physical(self) -> float:  # read only
        """returns epsiron that is projection algorithm error threshold for being physical.

        Returns
        -------
        float
            epsiron that is projection algorithm error threshold for being physical.
        """
        return self._eps_proj_physical

    @abstractmethod
    def is_physical(self, atol: float = None) -> bool:
        """returns whether the state is physically correct.

        Parameters
        ----------
        atol : float, optional
            the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.

        Returns
        -------
        bool
            whether the state is physically correct.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_zero(self):
        """sets parameters to zero.

        this function must be implemented in the subclass.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    def generate_zero_obj(self) -> "QOperation":
        """returns zero object of QOperation.

        Returns
        -------
        QOperation
            zero object of QOperation.
        """
        new_value = self._generate_zero_obj()
        new_qoperation = self.__class__(
            self.composite_system,
            new_value,
            is_physicality_required=False,
            is_estimation_object=False,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            eps_proj_physical=self.eps_proj_physical,
        )
        return new_qoperation

    @abstractmethod
    def _generate_zero_obj(self):
        """returns ``np.array`` which zero object of QOperation has.

        this function is called from :func:`~quara.objects.qoperation.QOperation.generate_zero_obj`.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    def generate_origin_obj(self) -> "QOperation":
        """returns origin object of QOperation.

        Returns
        -------
        QOperation
            origin object of QOperation.
        """
        new_value = self._generate_origin_obj()
        new_qoperation = self.__class__(
            self.composite_system,
            new_value,
            is_physicality_required=False,
            is_estimation_object=False,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            eps_proj_physical=self.eps_proj_physical,
        )
        return new_qoperation

    @abstractmethod
    def _generate_origin_obj(self):
        """returns ``np.array`` which origin object of QOperation has.

        this function is called from :func:`~quara.objects.qoperation.QOperation.generate_origin_obj`.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    def copy(self) -> "QOperation":
        """returns copy of QOperation.

        Returns
        -------
        QOperation
            copy of QOperation.
        """
        new_value = self._copy()
        new_qoperation = self.__class__(
            self.composite_system,
            new_value,
            is_physicality_required=self.is_physicality_required,
            is_estimation_object=self.is_estimation_object,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            eps_proj_physical=self.eps_proj_physical,
        )
        return new_qoperation

    @abstractmethod
    def _copy(self):
        """returns ``np.array`` which copy of QOperation has.

        this function is called from :func:`~quara.objects.qoperation.QOperation.copy`.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_var(self) -> np.array:
        """converts QOperation to variables.

        this function must be implemented in the subclass.

        Returns
        -------
        np.array
            variable representation of QOperation.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_stacked_vector(self) -> np.array:
        """converts QOperation to stacked vector.

        this function must be implemented in the subclass.

        Returns
        -------
        np.array
            stacked vector representation of QOperation.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def calc_gradient(self, var_index: int) -> "QOperation":
        """calculates gradient of QOperation.

        this function must be implemented in the subclass.

        Parameters
        ----------
        var_index : int
            index of variables to calculate gradient.

        Returns
        -------
        QOperation
            gradient of QOperation.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def calc_proj_eq_constraint(self) -> "QOperation":
        """calculates the projection of QOperation on equal constraint.

        Returns
        -------
        QOperation
            the projection of QOperation on equal constraint.
        """
        raise NotImplementedError()

    @abstractmethod
    def calc_proj_ineq_constraint(self) -> "QOperation":
        """calculates the projection of QOperation on inequal constraint.

        Returns
        -------
        QOperation
            the projection of QOperation on inequal constraint.
        """
        raise NotImplementedError()

    @abstractmethod
    def _generate_from_var_func(self):
        raise NotImplementedError()

    def generate_from_var(
        self,
        var: np.array,
        is_physicality_required: bool = None,
        is_estimation_object: bool = None,
        on_para_eq_constraint: bool = None,
        on_algo_eq_constraint: bool = None,
        on_algo_ineq_constraint: bool = None,
        eps_proj_physical: float = None,
    ) -> "QOperation":
        """generates QOperation from variables.

        Parameters
        ----------
        var : np.array
        is_physicality_required : bool, optional
            whether this QOperation is physicality required, by default None.
            if this parameter is None, the value of this instance is set.
        is_estimation_object : bool, optional
            whether this QOperation is estimation object, by default None.
            if this parameter is None, the value of this instance is set.
        on_para_eq_constraint : bool, optional
            whether this QOperation is on parameter equality constraint, by default None.
            if this parameter is None, the value of this instance is set.
        on_algo_eq_constraint : bool, optional
            whether this QOperation is on algorithm equality constraint, by default None.
            if this parameter is None, the value of this instance is set.
        on_algo_ineq_constraint : bool, optional
            whether this QOperation is on algorithm inequality constraint, by default None.
            if this parameter is None, the value of this instance is set.
        eps_proj_physical : float, optional
            epsiron that is projection algorithm error threshold for being physical, by default None.
            if this parameter is None, the value of this instance is set.

        Returns
        -------
        QOperation
            generated QOperation.
        """
        # generate_from_var_func()
        is_physicality_required = (
            self.is_physicality_required
            if is_physicality_required is None
            else is_physicality_required
        )
        is_estimation_object = (
            self.is_estimation_object
            if is_estimation_object is None
            else is_estimation_object
        )
        on_para_eq_constraint = (
            self.on_para_eq_constraint
            if on_para_eq_constraint is None
            else on_para_eq_constraint
        )
        on_algo_eq_constraint = (
            self.on_algo_eq_constraint
            if on_algo_eq_constraint is None
            else on_algo_eq_constraint
        )
        on_algo_ineq_constraint = (
            self.on_algo_ineq_constraint
            if on_algo_ineq_constraint is None
            else on_algo_ineq_constraint
        )
        eps_proj_physical = (
            self.eps_proj_physical if eps_proj_physical is None else eps_proj_physical
        )

        generate_from_var_func = self._generate_from_var_func()
        c_sys = self.composite_system
        new_qoperation = generate_from_var_func(
            c_sys=c_sys,
            var=var,
            is_physicality_required=is_physicality_required,
            is_estimation_object=is_estimation_object,
            on_para_eq_constraint=on_para_eq_constraint,
            on_algo_eq_constraint=on_algo_eq_constraint,
            on_algo_ineq_constraint=on_algo_ineq_constraint,
            eps_proj_physical=eps_proj_physical,
        )
        return new_qoperation

    def calc_proj_physical(
        self, is_iteration_history: bool = False
    ) -> Union["QOperation", Tuple["QOperation", Dict]]:
        """calculates the projection of QOperation with physically correctness.
 
        Parameters
        ----------
        is_iteration_history : bool, optional
            whether this funstion returns iteration history, by default False.

        Returns
        -------
        Union["QOperation", Tuple["QOperation", Dict]]
            if ``is_iteration_history`` is True, returns the projection of QOperation with physically correctness and iteration history.
            otherwise, returns only the projection of QOperation with physically correctness.

            iteration history forms the following dict:
            {
                "p": list of opject ``p``,
                "q": list of opject ``q``,
                "x": list of opject ``x``,
                "y": list of opject ``y``,
                "error_value": list of opject ``error_value``,
            }

        """
        p_prev = self.generate_zero_obj()
        q_prev = self.generate_zero_obj()
        x_prev = self.copy()
        x_prev._is_physicality_required = False
        x_prev._is_estimation_object = False
        y_prev = x_prev.calc_proj_ineq_constraint()
        p_next = x_next = q_next = y_next = None

        # variables for debug
        if is_iteration_history:
            ps = [p_prev]
            qs = [q_prev]
            xs = [x_prev]
            ys = [y_prev]
            error_values = []

        k = 0
        is_stopping = False
        while not is_stopping:
            # shift variables
            if (
                p_next is not None
                and q_next is not None
                and x_next is not None
                and y_next is not None
            ):
                p_prev = p_next
                q_prev = q_next
                x_prev = x_next
                y_prev = y_next

            p_next = x_prev + p_prev - y_prev
            x_next = (y_prev + q_prev).calc_proj_eq_constraint()
            q_next = y_prev + q_prev - x_next
            y_next = (x_next + p_next).calc_proj_ineq_constraint()

            # logging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"calc_proj_physical iteration={k}")
                logger.debug(
                    f"p_prev={p_prev.to_stacked_vector()}, p_next={p_next.to_stacked_vector()}"
                )
                logger.debug(
                    f"q_prev={q_prev.to_stacked_vector()}, q_next={q_next.to_stacked_vector()}"
                )
                logger.debug(
                    f"x_prev={x_prev.to_stacked_vector()}, x_next={x_next.to_stacked_vector()}"
                )
                logger.debug(
                    f"y_prev={y_prev.to_stacked_vector()}, y_next={y_next.to_stacked_vector()}"
                )
            k += 1

            (
                is_stopping,
                error_value,
            ) = self._is_satisfied_stopping_criterion_birgin_raydan_qoperations(
                p_prev,
                p_next,
                q_prev,
                q_next,
                x_prev,
                x_next,
                y_prev,
                y_next,
                self.eps_proj_physical,
            )
            if is_iteration_history:
                ps.append(p_next)
                qs.append(q_next)
                xs.append(x_next)
                ys.append(y_next)
                error_values.append(error_value)

        if is_iteration_history:
            history = {
                "p": ps,
                "q": qs,
                "x": xs,
                "y": ys,
                "error_value": error_values,
            }
            return x_next, history
        else:
            return x_next

    def _calc_stopping_criterion_birgin_raydan_vectors(
        self,
        p_prev: np.array,
        p_next: np.array,
        q_prev: np.array,
        q_next: np.array,
        x_prev: np.array,
        x_next: np.array,
        y_prev: np.array,
        y_next: np.array,
    ) -> float:
        val = (
            np.sum((p_prev - p_next) ** 2 + (q_prev - q_next) ** 2)
            + 2 * np.abs(np.dot(p_prev, x_prev - x_next))
            + 2 * np.abs(np.dot(q_prev, y_prev - y_next))
        )

        logger.debug(f"result of _calc_stopping_criterion_birgin_raydan_vectors={val}")
        return val

    def _is_satisfied_stopping_criterion_birgin_raydan_vectors(
        self,
        p_prev: np.array,
        p_next: np.array,
        q_prev: np.array,
        q_next: np.array,
        x_prev: np.array,
        x_next: np.array,
        y_prev: np.array,
        y_next: np.array,
        eps_proj_physical: float,
    ):
        error_value = self._calc_stopping_criterion_birgin_raydan_vectors(
            p_prev, p_next, q_prev, q_next, x_prev, x_next, y_prev, y_next
        )
        if error_value < eps_proj_physical:
            return True, error_value
        else:
            return False, error_value

    def _is_satisfied_stopping_criterion_birgin_raydan_qoperations(
        self,
        p_prev: "QOperation",
        p_next: "QOperation",
        q_prev: "QOperation",
        q_next: "QOperation",
        x_prev: "QOperation",
        x_next: "QOperation",
        y_prev: "QOperation",
        y_next: "QOperation",
        eps_proj_physical: float,
    ) -> bool:
        if (
            p_prev is None
            or p_next is None
            or q_prev is None
            or q_next is None
            or x_prev is None
            or x_next is None
            or y_prev is None
            or y_next is None
        ):
            return False

        (
            result,
            error_value,
        ) = self._is_satisfied_stopping_criterion_birgin_raydan_vectors(
            p_prev.to_stacked_vector(),
            p_next.to_stacked_vector(),
            q_prev.to_stacked_vector(),
            q_next.to_stacked_vector(),
            x_prev.to_stacked_vector(),
            x_next.to_stacked_vector(),
            y_prev.to_stacked_vector(),
            y_next.to_stacked_vector(),
            eps_proj_physical,
        )
        return result, error_value

    def __add__(self, other):
        # Validation
        if type(other) != type(self):
            error_message = (
                f"'+' not supported between instances of {type(self)} and {type(other)}"
            )
            raise TypeError(error_message)

        if other.composite_system is not self.composite_system:
            # TODO: error message
            raise ValueError()

        if (
            (self.is_physicality_required != other.is_physicality_required)
            or (self.is_estimation_object != other.is_estimation_object)
            or (self.on_para_eq_constraint != other.on_para_eq_constraint)
            or (self.on_algo_eq_constraint != other.on_algo_eq_constraint)
            or (self.on_algo_ineq_constraint != other.on_algo_ineq_constraint)
            or (self.eps_proj_physical != other.eps_proj_physical)
        ):
            # TODO: error message
            raise ValueError()

        # Calculation
        new_values = self._add_vec(other)

        # Ganerate new QObject
        new_qobject = self.__class__(
            self.composite_system,
            new_values,
            is_physicality_required=False,
            is_estimation_object=False,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            eps_proj_physical=self.eps_proj_physical,
        )
        return new_qobject

    def __sub__(self, other):
        # Validation
        if type(other) != type(self):
            error_message = (
                f"'-' not supported between instances of {type(self)} and {type(other)}"
            )
            raise TypeError(error_message)

        if other.composite_system is not self.composite_system:
            # TODO: error message
            raise ValueError()

        if (
            (self.is_physicality_required != other.is_physicality_required)
            or (self.is_estimation_object != other.is_estimation_object)
            or (self.on_para_eq_constraint != other.on_para_eq_constraint)
            or (self.on_algo_eq_constraint != other.on_algo_eq_constraint)
            or (self.on_algo_ineq_constraint != other.on_algo_ineq_constraint)
            or (self.eps_proj_physical != other.eps_proj_physical)
        ):
            # TODO: error message
            raise ValueError()

        # Calculation
        new_values = self._sub_vec(other)

        # Ganerate new QObject
        new_qobject = self.__class__(
            self.composite_system,
            new_values,
            is_physicality_required=False,
            is_estimation_object=False,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            eps_proj_physical=self.eps_proj_physical,
        )
        return new_qobject

    def __mul__(self, other):
        # Validation
        if type(other) not in [int, float, np.int64, np.float64]:
            error_message = (
                f"'*' not supported between instances of {type(self)} and {type(other)}"
            )
            raise TypeError(error_message)

        # Calculation
        new_values = self._mul_vec(other)

        # Ganerate new QObject
        new_qobject = self.__class__(
            self.composite_system,
            new_values,
            is_physicality_required=False,
            is_estimation_object=False,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            eps_proj_physical=self.eps_proj_physical,
        )
        return new_qobject

    def __rmul__(self, other):
        # other * self
        new_qobject = self.__mul__(other)
        return new_qobject

    def __truediv__(self, other):
        # Validation
        if type(other) not in [int, float, np.int64, np.float64]:
            error_message = (
                f"'/' not supported between instances of {type(self)} and {type(other)}"
            )
            raise TypeError(error_message)

        # Calculation
        new_values = self._truediv_vec(other)

        # Ganerate new QObject
        new_qobject = self.__class__(
            self.composite_system,
            new_values,
            is_physicality_required=False,
            is_estimation_object=False,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            eps_proj_physical=self.eps_proj_physical,
        )
        return new_qobject
