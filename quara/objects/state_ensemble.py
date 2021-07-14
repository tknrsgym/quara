from typing import List

from quara.objects.qoperation import QOperation
from quara.objects.state import State
from quara.objects.prob_dist import ProbDist


class StateEnsemble(QOperation):
    def __init__(self, states: List[State], prob_dist: ProbDist):
        self._states = states
        self._prob_dist = prob_dist

    @property  # read only
    def states(self):
        return self._states

    @property  # read only
    def prob_dist(self):
        return self._prob_dist

    def _info(self):
        info = {}
        info["Type"] = self.__class__.__name__
        info["States"] = "\n".join(
            [f"states[{i}]: {s._info()['Vec']}" for i, s in enumerate(self.states)]
        )
        info["ProbDist"] = self.prob_dist.__str__()

        return info
