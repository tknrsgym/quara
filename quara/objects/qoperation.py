from abc import abstractmethod
import logging

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

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    def generate_zero_obj(self) -> "QOperation":
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
        raise NotImplementedError()

    def generate_origin_obj(self) -> "QOperation":
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
        raise NotImplementedError()

    def copy(self) -> "QOperation":
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
        raise NotImplementedError()

    @abstractmethod
    def to_var(self) -> np.array:
        raise NotImplementedError()

    @abstractmethod
    def to_stacked_vector(self) -> np.array:
        raise NotImplementedError()

    @abstractmethod
    def calc_gradient(self):
        raise NotImplementedError()

    @abstractmethod
    def calc_proj_eq_constraint(self):
        raise NotImplementedError()

    @abstractmethod
    def calc_proj_ineq_constraint(self):
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

    def calc_proj_physical(self) -> "QOperation":
        p_prev = self.generate_zero_obj()
        q_prev = self.generate_zero_obj()
        x_prev = self.copy()
        x_prev._is_physicality_required = False
        x_prev._is_estimation_object = False
        y_prev = x_prev.calc_proj_ineq_constraint()
        p_next = x_next = q_next = y_next = None

        k = 0
        while not self._is_satisfied_stopping_criterion_birgin_raydan_qoperations(
            p_prev,
            p_next,
            q_prev,
            q_next,
            x_prev,
            x_next,
            y_prev,
            y_next,
            self.eps_proj_physical,
        ):
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

    def is_satisfied_stopping_criterion_birgin_raydan_vectors(
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
        val = self._calc_stopping_criterion_birgin_raydan_vectors(
            p_prev, p_next, q_prev, q_next, x_prev, x_next, y_prev, y_next
        )
        if val < eps_proj_physical:
            return True
        else:
            return False

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

        result = self.is_satisfied_stopping_criterion_birgin_raydan_vectors(
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
        return result

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

