from typing import List, Tuple, Union
import numpy as np
from quara.objects.qoperation import QOperation
from quara.objects.state import State
from quara.objects.multinomial_distribution import MultinomialDistribution
from quara.utils.index_util import index_serial_from_index_multi_dimensional


class StateEnsemble(QOperation):
    def __init__(
        self,
        states: List[State],
        prob_dist: MultinomialDistribution,
        eps_zero: Union[float, np.float64] = 10 ** -8,
    ):
        # Validation
        if eps_zero < 0:
            raise ValueError("eps_zero must be a non-negative value.")

        # Set
        self._states: List[State] = states
        self._prob_dist: MultinomialDistribution = prob_dist
        self._eps_zero: Union[float, np.float64] = eps_zero

    @property  # read only
    def states(self):
        return self._states

    @property  # read only
    def prob_dist(self):
        return self._prob_dist

    @property  # read only
    def eps_zero(self):
        return self._eps_zero

    def _info(self):
        info = {}
        info["Type"] = self.__class__.__name__
        info["States"] = "\n".join(
            [f"states[{i}]: {s._info()['Vec']}" for i, s in enumerate(self.states)]
        )
        info["MultinomialDistribution"] = self.prob_dist.__str__()

        return info

    def state(self, outcome: Union[int, Tuple[int]]):
        shape = self.prob_dist.shape
        serial_index = index_serial_from_index_multi_dimensional(nums_length=list(shape), index_multi_dimensional=outcome)
        return self._states[serial_index]
