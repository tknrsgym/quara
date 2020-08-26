from typing import List


class EstimationResult:
    def __init__(self, computation_times: List[float]):
        """Constructor

        Parameters
        ----------
        computation_times : List[float]
            computation times for each estimate.
        """
        self._computation_times: List[float] = computation_times

    @property
    def computation_time(self) -> float:
        """returns computation time for the estimate.

        Returns
        -------
        float
            computation time for the estimate.
        """
        if self._computation_times is None:
            return None
        else:
            return self._computation_times[0]

    @property
    def computation_times(self) -> List[float]:
        """returns computation times for each estimate.

        Returns
        -------
        List[float]
            computation times for each estimate.
        """
        return self._computation_times


class Estimator:
    def __init__(self):
        pass
