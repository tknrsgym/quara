from abc import abstractmethod

import numpy as np


class MinimizationAlgorithmOption:
    def __init__(
        self,
        var_start: np.array,
        is_gradient_required: bool,
        is_hessian_required: bool,
        on_iteration_history: bool = False,
    ):
        self._var_start: np.array = var_start
        self._is_gradient_required: bool = is_gradient_required
        self._is_hessian_required: bool = is_hessian_required
        self._on_iteration_history: bool = on_iteration_history

    @property
    def var_start(self) -> np.array:
        return self._var_start

    @property
    def is_gradient_required(self) -> bool:
        return self._is_gradient_required

    @property
    def is_hessian_required(self) -> bool:
        return self._is_hessian_required

    @property
    def on_iteration_history(self) -> bool:
        return self._on_iteration_history


class MinimizationAlgorithm:
    def __init__(self):
        pass

    @abstractmethod
    def optimize(self):
        raise NotImplementedError()

