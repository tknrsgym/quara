from abc import abstractmethod
from typing import List

import numpy as np

from quara.objects.qoperations import SetQOperations
from quara.qcircuit.experiment import Experiment


class QTomography:
    def __init__(
        self, experiment: Experiment, set_qoperations: SetQOperations,
    ):
        self._experiment = experiment
        self._num_schedules = self._experiment.schedules
        self._set_qoperations = set_qoperations
        # TODO num_variables

    @abstractmethod
    def calc_prob_dist(self) -> List[np.array]:
        raise NotImplementedError()

    @abstractmethod
    def calc_prob_dists(self) -> List[List[np.array]]:
        raise NotImplementedError()

    @abstractmethod
    def generate_dataset(self):
        raise NotImplementedError()

    @abstractmethod
    def generate_datasets(self):
        raise NotImplementedError()

    @abstractmethod
    def func_prob_dist(self):
        raise NotImplementedError()

    @abstractmethod
    def func_prob_dists(self):
        raise NotImplementedError()

    @abstractmethod
    def is_valid_experiment(self) -> bool:
        raise NotImplementedError()
