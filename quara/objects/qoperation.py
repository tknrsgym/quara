from abc import abstractmethod
import logging
from typing import Callable, Dict, List, Tuple, Union

import numpy as np

from quara.objects.composite_system import CompositeSystem
from quara.settings import Settings

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
        mode_proj_order: str = "eq_ineq",
        eps_proj_physical: float = None,
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
        mode_proj_order : str, optional
            the order in which the projections are performed, by default "eq_ineq"
        eps_proj_physical : float, optional
            epsiron that is projection algorithm error threshold for being physical, by default :func:`~quara.settings.Settings.get_atol` / 10.0

        Raises
        ------
        ValueError
            ``eps_proj_physical`` is negative.
        """
        eps_proj_physical = (
            eps_proj_physical if eps_proj_physical else Settings.get_atol() / 10.0
        )
        # Validation
        if eps_proj_physical < 0:
            raise ValueError("'eps_proj_physical' must be non-negative.")

        # if not c_sys.is_basis_hermitian:
        #     message = "`c_sys.is_basis_hermitian` is False. Basis must be Hermitian."
        #     raise ValueError(message)

        self._validate_mode_proj_order(mode_proj_order)

        # Set
        self._composite_system: CompositeSystem = c_sys
        self._is_physicality_required = is_physicality_required
        self._is_estimation_object = is_estimation_object
        self._on_para_eq_constraint: bool = on_para_eq_constraint
        self._on_algo_eq_constraint: bool = on_algo_eq_constraint
        self._on_algo_ineq_constraint: bool = on_algo_ineq_constraint
        self._mode_proj_order: str = mode_proj_order
        self._eps_proj_physical = eps_proj_physical

    def _validate_mode_proj_order(self, mode_proj_order):
        if not mode_proj_order in ["eq_ineq", "ineq_eq"]:
            raise ValueError(f"unsupported mode_proj_order={mode_proj_order}")

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
    def mode_proj_order(self) -> str:  # read only
        """returns the order in which the projections are performed.

        Returns
        -------
        str
            the order in which the projections are performed.
        """
        return self._mode_proj_order

    def set_mode_proj_order(self, mode_proj_order: str) -> None:
        """sets the order in which the projections are performed.

        Parameters
        ----------
        str
            the order in which the projections are performed.
        """
        self._validate_mode_proj_order(mode_proj_order)
        self._mode_proj_order = mode_proj_order

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
    def is_eq_constraint_satisfied(self, atol: float = None):
        raise NotImplementedError()

    @abstractmethod
    def is_ineq_constraint_satisfied(self, atol: float = None):
        raise NotImplementedError()

    @abstractmethod
    def estimation_object_type(self) -> type:
        """returns type of estimation object.

        Returns
        -------
        type
            type of estimation object.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def is_physical(
        self, atol_eq_const: float = None, atol_ineq_const: float = None
    ) -> bool:
        """returns whether the qoperation is physically correct.

        Parameters
        ----------
        atol_eq_const : float, optional
            Error tolerance used to determine if the equality constraint is satisfied. The absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.
        atol_ineq_const : float, optional
            Error tolerance used to determine if the inequality constraint is satisfied. The absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.

        Returns
        -------
        bool
            whether the qoperation is physically correct.
        """
        return self.is_eq_constraint_satisfied(
            atol_eq_const
        ) and self.is_ineq_constraint_satisfied(atol_ineq_const)

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
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
        )
        return new_qoperation

    @abstractmethod
    def _generate_zero_obj(self):
        """returns ``np.ndarray`` which zero object of QOperation has.

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
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
        )
        return new_qoperation

    @abstractmethod
    def _generate_origin_obj(self):
        """returns ``np.ndarray`` which origin object of QOperation has.

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
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
        )
        return new_qoperation

    @abstractmethod
    def _copy(self):
        """returns ``np.ndarray`` which copy of QOperation has.

        this function is called from :func:`~quara.objects.qoperation.QOperation.copy`.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_var(self) -> np.ndarray:
        """converts QOperation to variables.

        this function must be implemented in the subclass.

        Returns
        -------
        np.ndarray
            variable representation of QOperation.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_stacked_vector(self) -> np.ndarray:
        """converts QOperation to stacked vector.

        this function must be implemented in the subclass.

        Returns
        -------
        np.ndarray
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

    def func_calc_proj_eq_constraint(
        self, on_para_eq_constraint: bool = None
    ) -> Callable[[np.ndarray], np.ndarray]:
        if on_para_eq_constraint is None:
            on_para_eq_constraint = self._on_para_eq_constraint

        qobj_empty = self.generate_zero_obj()

        def _func_proj(var: np.ndarray) -> np.ndarray:
            qobj_tmp = qobj_empty.generate_from_var(
                var, on_para_eq_constraint=on_para_eq_constraint
            )
            qobj_result = qobj_tmp.calc_proj_eq_constraint()
            return qobj_result.to_var()

        return _func_proj

    def func_calc_proj_eq_constraint_with_var(
        self, on_para_eq_constraint: bool = None
    ) -> Callable[[np.ndarray], np.ndarray]:
        if on_para_eq_constraint is None:
            on_para_eq_constraint = self._on_para_eq_constraint

        qobj_empty = self.generate_zero_obj()

        def _func_proj(var: np.ndarray) -> np.ndarray:
            new_var = self.calc_proj_eq_constraint_with_var(
                self.composite_system, var, on_para_eq_constraint=on_para_eq_constraint
            )
            return new_var

        return _func_proj

    @abstractmethod
    def calc_proj_ineq_constraint(self) -> "QOperation":
        """calculates the projection of QOperation on inequal constraint.

        Returns
        -------
        QOperation
            the projection of QOperation on inequal constraint.
        """
        raise NotImplementedError()

    def func_calc_proj_ineq_constraint(
        self, on_para_eq_constraint: bool = None
    ) -> Callable[[np.ndarray], np.ndarray]:
        if on_para_eq_constraint is None:
            on_para_eq_constraint = self._on_para_eq_constraint

        qobj_empty = self.generate_zero_obj()

        def _func_proj(var: np.ndarray) -> np.ndarray:
            qobj_tmp = qobj_empty.generate_from_var(
                var, on_para_eq_constraint=on_para_eq_constraint
            )
            qobj_result = qobj_tmp.calc_proj_ineq_constraint()
            return qobj_result.to_var()

        return _func_proj

    def func_calc_proj_ineq_constraint_with_var(
        self, on_para_eq_constraint: bool = None
    ) -> Callable[[np.ndarray], np.ndarray]:
        if on_para_eq_constraint is None:
            on_para_eq_constraint = self._on_para_eq_constraint

        qobj_empty = self.generate_zero_obj()

        def _func_proj(var: np.ndarray) -> np.ndarray:
            new_var = self.calc_proj_ineq_constraint_with_var(
                self.composite_system, var, on_para_eq_constraint=on_para_eq_constraint
            )
            return new_var

        return _func_proj

    @abstractmethod
    def _generate_from_var_func(self):
        raise NotImplementedError()

    def generate_from_var(
        self,
        var: np.ndarray,
        is_physicality_required: bool = None,
        is_estimation_object: bool = None,
        on_para_eq_constraint: bool = None,
        on_algo_eq_constraint: bool = None,
        on_algo_ineq_constraint: bool = None,
        mode_proj_order: str = "eq_ineq",
        eps_proj_physical: float = None,
    ) -> "QOperation":
        """generates QOperation from variables.

        Parameters
        ----------
        var : np.ndarray
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
        mode_proj_order : str, optional
            the order in which the projections are performed, by default "eq_ineq".
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
            mode_proj_order=mode_proj_order,
            eps_proj_physical=eps_proj_physical,
        )
        return new_qoperation

    def calc_proj_physical(
        self, max_iteration: int = 1000, is_iteration_history: bool = False
    ) -> Union["QOperation", Tuple["QOperation", Dict]]:
        """calculates the projection of QOperation with physically correctness.

        Parameters
        ----------
        max_iteration: int, optional
            maximun number of iterations, by default 1000.
        is_iteration_history : bool, optional
            whether this function returns iteration history, by default False.

        Returns
        -------
        Union["QOperation", Tuple["QOperation", Dict]]
            if ``is_iteration_history`` is True, returns the projection of QOperation with physically correctness and iteration history.
            otherwise, returns only the projection of QOperation with physically correctness.

            iteration history forms the following dict:

            .. line-block::
                {
                    "p": list of opject ``p``,
                    "q": list of opject ``q``,
                    "x": list of opject ``x``,
                    "y": list of opject ``y``,
                    "error_value": list of opject ``error_value``,
                }

            When step=0, "y" and "error_value" are not calculated, so None is set.

        """
        p_prev = self.generate_zero_obj()
        q_prev = self.generate_zero_obj()
        x_prev = self.copy()
        x_prev._is_physicality_required = False
        x_prev._is_estimation_object = False
        y_prev = None

        p_next = x_next = q_next = y_next = None

        # variables for debug
        if is_iteration_history:
            ps = [p_prev]
            qs = [q_prev]
            xs = [x_prev]
            ys = [y_prev]
            error_values = []

        is_stopping = False
        for k in range(max_iteration):
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

            if self.mode_proj_order == "eq_ineq":
                y_next = (x_prev + p_prev).calc_proj_eq_constraint()
                p_next = x_prev + p_prev - y_next
                x_next = (y_next + q_prev).calc_proj_ineq_constraint()
                q_next = y_next + q_prev - x_next
            else:
                y_next = (x_prev + p_prev).calc_proj_ineq_constraint()
                p_next = x_prev + p_prev - y_next
                x_next = (y_next + q_prev).calc_proj_eq_constraint()
                q_next = y_next + q_prev - x_next

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

            # check satisfied stopping criterion
            if k >= 1:
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
            else:
                error_value = None

            if is_iteration_history:
                ps.append(p_next)
                qs.append(q_next)
                xs.append(x_next)
                ys.append(y_next)
                error_values.append(error_value)

            if is_stopping:
                break

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
        p_prev: np.ndarray,
        p_next: np.ndarray,
        q_prev: np.ndarray,
        q_next: np.ndarray,
        x_prev: np.ndarray,
        x_next: np.ndarray,
        y_prev: np.ndarray,
        y_next: np.ndarray,
    ) -> float:
        val = (
            np.sum((p_prev - p_next) ** 2 + (q_prev - q_next) ** 2)
            - 2 * np.dot(p_prev, y_next - y_prev)
            - 2 * np.dot(q_prev, x_next - x_prev)
        )

        logger.debug(f"result of _calc_stopping_criterion_birgin_raydan_vectors={val}")
        return val

    def _is_satisfied_stopping_criterion_birgin_raydan_vectors(
        self,
        p_prev: np.ndarray,
        p_next: np.ndarray,
        q_prev: np.ndarray,
        q_next: np.ndarray,
        x_prev: np.ndarray,
        x_next: np.ndarray,
        y_prev: np.ndarray,
        y_next: np.ndarray,
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

    def func_calc_proj_physical(
        self,
        on_para_eq_constraint: bool = None,
        mode_proj_order: str = "eq_ineq",
    ) -> Callable[[np.ndarray], np.ndarray]:
        if on_para_eq_constraint is None:
            on_para_eq_constraint = self._on_para_eq_constraint

        qobj_empty = self.generate_zero_obj()

        def _func_proj(var: np.ndarray) -> np.ndarray:
            qobj_tmp = qobj_empty.generate_from_var(
                var,
                on_para_eq_constraint=on_para_eq_constraint,
                mode_proj_order=mode_proj_order,
            )
            qobj_result = qobj_tmp.calc_proj_physical()
            return qobj_result.to_var()

        return _func_proj

    def calc_proj_physical_with_var(
        self,
        var: np.ndarray,
        on_para_eq_constraint: bool = True,
        max_iteration: int = 1000,
        is_iteration_history: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """calculates the projection of variables with physically correctness.

        Parameters
        ----------
        var : np.ndarray
            variables.
        on_para_eq_constraint : bool, optional
            whether this variables is on parameter equality constraint, by default True.
        max_iteration: int, optional
            maximun number of iterations, by default 1000.
        is_iteration_history : bool, optional
            whether this function returns iteration history, by default False.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, Dict]]
            if ``is_iteration_history`` is True, returns the projection of variables with physically correctness and iteration history.
            otherwise, returns only the projection of variables with physically correctness.

            iteration history forms the following dict:

            .. line-block::
                {
                    "p": list of opject ``p``,
                    "q": list of opject ``q``,
                    "x": list of opject ``x``,
                    "y": list of opject ``y``,
                    "error_value": list of opject ``error_value``,
                }

            When step=0, "y" and "error_value" are not calculated, so None is set.
        """
        p_prev = self.generate_zero_obj().to_stacked_vector()
        q_prev = self.generate_zero_obj().to_stacked_vector()
        x_prev = self.convert_var_to_stacked_vector(
            self.composite_system, var, on_para_eq_constraint=on_para_eq_constraint
        )
        y_prev = None

        p_next = x_next = q_next = y_next = None

        # variables for debug
        if is_iteration_history:
            ps = [p_prev]
            qs = [q_prev]
            xs = [x_prev]
            ys = [y_prev]
            error_values = []

        is_stopping = False
        for k in range(max_iteration):
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

            if self.mode_proj_order == "eq_ineq":
                y_next = self.calc_proj_eq_constraint_with_var(
                    self.composite_system, x_prev + p_prev, on_para_eq_constraint=False
                )
                p_next = x_prev + p_prev - y_next
                x_next = self.calc_proj_ineq_constraint_with_var(
                    self.composite_system, y_next + q_prev, on_para_eq_constraint=False
                )
                q_next = y_next + q_prev - x_next
            else:
                y_next = self.calc_proj_ineq_constraint_with_var(
                    self.composite_system, x_prev + p_prev, on_para_eq_constraint=False
                )
                p_next = x_prev + p_prev - y_next
                x_next = self.calc_proj_eq_constraint_with_var(
                    self.composite_system, y_next + q_prev, on_para_eq_constraint=False
                )
                q_next = y_next + q_prev - x_next

            # logging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"calc_proj_physical iteration={k}")
                logger.debug(f"p_prev={p_prev}, p_next={p_next}")
                logger.debug(f"q_prev={q_prev}, q_next={q_next}")
                logger.debug(f"x_prev={x_prev}, x_next={x_next}")
                logger.debug(f"y_prev={y_prev}, y_next={y_next}")

            # check satisfied stopping criterion
            if k >= 1:
                (
                    is_stopping,
                    error_value,
                ) = self._is_satisfied_stopping_criterion_birgin_raydan_vectors(
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
            else:
                error_value = None

            if is_iteration_history:
                ps.append(p_next)
                qs.append(q_next)
                xs.append(x_next)
                ys.append(y_next)
                error_values.append(error_value)

            if is_stopping:
                break

        x_next = self.convert_stacked_vector_to_var(
            self.composite_system,
            x_next,
            on_para_eq_constraint=on_para_eq_constraint,
        )
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

    def func_calc_proj_physical_with_var(
        self,
        on_para_eq_constraint: bool = None,
        mode_proj_order: str = "eq_ineq",
    ) -> Callable[[np.ndarray], np.ndarray]:
        if on_para_eq_constraint is None:
            on_para_eq_constraint = self._on_para_eq_constraint

        def _func_proj(var: np.ndarray) -> np.ndarray:
            new_var = self.calc_proj_physical_with_var(
                var, on_para_eq_constraint=on_para_eq_constraint
            )
            return new_var

        return _func_proj

    def __add__(self, other):
        # Validation
        if type(other) != type(self):
            error_message = f"'+' not supported between instances of {type(self)} and {type(other)}."
            raise TypeError(error_message)

        if other.composite_system is not self.composite_system:
            message = "'+' not supported between instances with different composite_system.elemental_systems."
            raise ValueError(message)

        if (
            (self.is_physicality_required != other.is_physicality_required)
            or (self.is_estimation_object != other.is_estimation_object)
            or (self.on_para_eq_constraint != other.on_para_eq_constraint)
            or (self.on_algo_eq_constraint != other.on_algo_eq_constraint)
            or (self.on_algo_ineq_constraint != other.on_algo_ineq_constraint)
            or (self.mode_proj_order != other.mode_proj_order)
            or (self.eps_proj_physical != other.eps_proj_physical)
        ):
            message = "'-' not supported between instances with different configration."
            config_dict = dict(
                is_physicality_required=(
                    self.is_physicality_required,
                    other.is_physicality_required,
                ),
                is_estimation_object=(
                    self.is_estimation_object,
                    other.is_estimation_object,
                ),
                on_para_eq_constraint=(
                    self.on_para_eq_constraint,
                    other.on_para_eq_constraint,
                ),
                on_algo_eq_constraint=(
                    self.on_algo_eq_constraint,
                    other.on_algo_eq_constraint,
                ),
                on_algo_ineq_constraint=(
                    self.on_algo_ineq_constraint,
                    other.on_algo_ineq_constraint,
                ),
                mode_proj_order=(
                    self.mode_proj_order,
                    other.mode_proj_order,
                ),
                eps_proj_physical=(
                    self.eps_proj_physical,
                    other.eps_proj_physical,
                ),
            )
            for k, v in config_dict.items():
                if v[0] != v[1]:
                    message += f"\nself.{k}=v[0], other.{k}=v[1]"
            raise ValueError(message)

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
            mode_proj_order=self.mode_proj_order,
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
            message = "'-' not supported between instances with different composite_system.elemental_systems."
            raise ValueError(message)

        if (
            (self.is_physicality_required != other.is_physicality_required)
            or (self.is_estimation_object != other.is_estimation_object)
            or (self.on_para_eq_constraint != other.on_para_eq_constraint)
            or (self.on_algo_eq_constraint != other.on_algo_eq_constraint)
            or (self.on_algo_ineq_constraint != other.on_algo_ineq_constraint)
            or (self.mode_proj_order != other.mode_proj_order)
            or (self.eps_proj_physical != other.eps_proj_physical)
        ):
            message = "'-' not supported between instances with different configration."
            config_dict = dict(
                is_physicality_required=(
                    self.is_physicality_required,
                    other.is_physicality_required,
                ),
                is_estimation_object=(
                    self.is_estimation_object,
                    other.is_estimation_object,
                ),
                on_para_eq_constraint=(
                    self.on_para_eq_constraint,
                    other.on_para_eq_constraint,
                ),
                on_algo_eq_constraint=(
                    self.on_algo_eq_constraint,
                    other.on_algo_eq_constraint,
                ),
                on_algo_ineq_constraint=(
                    self.on_algo_ineq_constraint,
                    other.on_algo_ineq_constraint,
                ),
                mode_proj_order=(
                    self.mode_proj_order,
                    other.mode_proj_order,
                ),
                eps_proj_physical=(
                    self.eps_proj_physical,
                    other.eps_proj_physical,
                ),
            )
            for k, v in config_dict.items():
                if v[0] != v[1]:
                    message += f"\nself.{k}=v[0], other.{k}=v[1]"
            raise ValueError(message)

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
            mode_proj_order=self.mode_proj_order,
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
            mode_proj_order=self.mode_proj_order,
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
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
        )
        return new_qobject

    def __str__(self):
        desc = ""
        for k, v in self._info().items():
            desc += f"{k}:\n{v.__str__()}\n\n"
        desc = desc.rstrip("\n")
        return desc
