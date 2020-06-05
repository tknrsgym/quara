from typing import List

import numpy as np

from quara.qcircuit.experiment import Experiment


class QTomography:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def calc_prob_dist(self) -> List[np.array]:
        pass

    def calc_prob_dists(self) -> List[List[np.array]]:
        pass

    def generate_dataset(self):
        pass

    def generate_datasets(self):
        pass

    def func_prob_dist(self):
        pass

    def func_prob_dists(self):
        pass

    def is_valid_experiment(self) -> bool:
        pass
