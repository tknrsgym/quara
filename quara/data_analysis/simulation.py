from typing import List
import copy


class SimulationSetting:
    def __init__(
        self,
        name: str,
        estimator: "Estimator",
        loss=None,
        loss_option=None,
        algo=None,
        algo_option=None,
    ) -> None:
        self.name = name
        self.estimator = copy.copy(estimator)
        self.loss = copy.copy(loss)
        self.loss_option = loss_option
        self.algo = copy.copy(algo)
        self.algo_option = algo_option

    def __str__(self):
        desc = f"Name: {self.name}"
        desc += f"Estimator: {self.estimator.__class__.__name__}"
        loss = None if self.loss is None else self.loss.__class__.__name__
        desc += f"Loss: {loss}"
        algo = None if self.algo is None else self.algo.__class__.__name__
        desc += f"Algo: {algo}"
        return desc