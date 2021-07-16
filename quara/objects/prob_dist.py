from typing import Tuple
import numpy as np


from typing import Tuple


class ProbDist:
    def __init__(self, ps: np.ndarray, shape: Tuple[int] = None):
        self._ps = ps
        self._shape = shape

    @property  # read only
    def ps(self):
        return self._ps

    @property  # read only
    def shape(self):
        return self._shape

    def __getitem__(self, idx):
        # Working in progress
        if type(idx) == int:
            # One-dimensional access
            # ex) prob_dist[0]
            return self._ps[idx]
        elif type(idx) == tuple:
            # Multidimensional access
            # ex) prob_dist[0][1]
            if self._shape is None:
                # TODO: message
                raise ValueError
            target = self._ps.reshape(*self._shape)
            for i in idx:
                target = target[i]
            return target
        else:
            raise TypeError

    def __str__(self):
        desc = f"shape = {self.shape}\n"
        desc += f"ps = {self._ps}"
        return desc
