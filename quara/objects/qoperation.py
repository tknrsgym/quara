from abc import abstractmethod

import numpy as np

from quara.objects.composite_system import CompositeSystem


class QOperation:
    def __init__(
        self,
        c_sys: CompositeSystem,
        is_physicality_required: bool = True,
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
        self._on_para_eq_constraint: bool = on_para_eq_constraint
        self._on_algo_eq_constraint: bool = on_algo_eq_constraint
        self._on_algo_ineq_constraint: bool = on_algo_ineq_constraint
        self._eps_proj_physical = eps_proj_physical

    @property
    def is_physicality_required(self) -> bool:  # read only
        return self._is_physicality_required

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

    def calc_proj_physical(self):
        raise NotImplementedError()

    def calc_stopping_criterion_birgin_raydan_vectors(self):
        raise NotImplementedError()

    def is_satisfied_stopping_criterion_birgin_raydan_vectors(self):
        raise NotImplementedError()

    def is_satisfied_stopping_criterion_birgin_raydan_qoperations(self):
        raise NotImplementedError()
