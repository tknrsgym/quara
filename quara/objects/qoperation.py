from abc import abstractmethod

import numpy as np

from quara.objects.composite_system import CompositeSystem


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
    def is_physicality_required(self) -> bool:  # read only
        return self._is_physicality_required

    @property
    def is_estimation_object(self) -> bool:  # read only
        return self._is_estimation_object

    @property
    def on_para_eq_constraint(self) -> bool:  # read only
        return self._on_para_eq_constraint

    @property
    def on_algo_eq_constraint(self) -> bool:  # read only
        return self._on_algo_eq_constraint

    @property
    def on_algo_ineq_constraint(self) -> bool:  # read only
        return self._on_algo_ineq_constraint

    @property
    def eps_proj_physical(self) -> float:  # read only
        return self._eps_proj_physical

    @abstractmethod
    def is_physical(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def set_zero(self):
        raise NotImplementedError()

    @abstractmethod
    def zero_obj(self):
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
            is_physicality_required
            if is_physicality_required
            else self.is_physicality_required
        )
        is_estimation_object = (
            is_estimation_object if is_estimation_object else self.is_estimation_object
        )
        on_para_eq_constraint = (
            on_para_eq_constraint
            if on_para_eq_constraint
            else self.on_para_eq_constraint
        )
        on_algo_eq_constraint = (
            on_algo_eq_constraint
            if on_algo_eq_constraint
            else self.on_algo_eq_constraint
        )
        on_algo_ineq_constraint = (
            on_algo_ineq_constraint
            if on_algo_ineq_constraint
            else self.on_algo_ineq_constraint
        )
        eps_proj_physical = (
            eps_proj_physical if eps_proj_physical else self.eps_proj_physical
        )

        generate_from_var_func = self._generate_from_var_func()
        c_sys = self._composite_system
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

    def calc_proj_physical(self):
        raise NotImplementedError()

    def calc_stopping_criterion_birgin_raydan_vectors(self):
        raise NotImplementedError()

    def is_satisfied_stopping_criterion_birgin_raydan_vectors(self):
        raise NotImplementedError()

    def is_satisfied_stopping_criterion_birgin_raydan_qoperations(self):
        raise NotImplementedError()
