from typing import List

from quara.objects.qoperation import QOperation, State
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
