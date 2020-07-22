from typing import List


class EstimationResult:
    def __init__(self, computation_times: List[float]):
        self._computation_times: List[float] = computation_times

    @property
    def computation_time(self):
        return self._computation_times[0]

    @property
    def computation_times(self):
        return self._computation_times


class Estimator:
    pass
