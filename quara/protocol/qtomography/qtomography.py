from abc import abstractmethod
from typing import List

import numpy as np

from quara.objects.qoperations import SetQOperations
from quara.qcircuit.experiment import Experiment


class QTomography:
    def __init__(
        self, experiment: Experiment, set_qoperations: SetQOperations,
    ):
        """initialize quantum tomography class.

        Parameters
        ----------
        experiment : Experiment
            Experiment class used in quantum tomography.
        set_qoperations : SetQOperations
            SetQOperations class used in quantum tomography.
        """
        self._experiment = experiment
        self._num_schedules = len(self._experiment.schedules)
        self._set_qoperations = set_qoperations
        # TODO num_variables

    @property
    def num_schedules(self) -> int:
        return self._num_schedules

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
