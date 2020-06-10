from abc import abstractmethod


class QOperation:
    def __init__(
        self,
        is_physical: bool = True,
        on_para_eq_constraint: bool = True,
        on_algo_eq_constraint: bool = True,
        on_algo_ineq_constraint: bool = True,
        eps_proj_physical: float = 10 ** (-4),
    ):
        # Validation
        if eps_proj_physical < 0:
            raise ValueError("'eps_proj_physical' must be non-negative.")

        # Set
        self._is_physical = is_physical
        self._on_para_eq_constraint: bool = on_para_eq_constraint
        self._on_algo_eq_constraint: bool = on_algo_eq_constraint
        self._on_algo_ineq_constraint: bool = on_algo_ineq_constraint
        self._eps_proj_physical = eps_proj_physical

    @property
    def is_physical(self):
        return self._is_physical

    @property
    def on_para_eq_constraint(self) -> bool:  # read only
        return self._on_para_eq_constraint

    @property
    def on_algo_eq_constraint(self) -> bool:  # read only
        return self._on_algo_eq_constraint

    @property
    def on_algo_eq_constraint(self) -> bool:  # read only
        return self._on_algo_ineq_constraint

    @property
    def eps_proj_physical(self) -> float:  # read only
        return self._eps_proj_physical

    @abstractmethod
    def to_var(self):
        raise NotImplementedError()
