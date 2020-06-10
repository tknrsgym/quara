from abc import abstractmethod


class QOperation:
    def __init__(self, on_para_eq_constraint: bool = True):
        self._on_para_eq_constraint: bool = on_para_eq_constraint

    @property
    def on_para_eq_constraint(self) -> bool:  # read only
        return self._on_para_eq_constraint

    @abstractmethod
    def to_var(self):
        raise NotImplementedError()
