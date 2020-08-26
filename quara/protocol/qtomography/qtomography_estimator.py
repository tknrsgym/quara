from typing import List

from quara.protocol.qtomography.estimator import Estimator, EstimationResult


class QTomographyEstimationResult(EstimationResult):
    def __init__(self, computation_times: List[float]):
        super().__init__(computation_times)


class QTomographyEstimator(Estimator):
    def __init__(self):
        super().__init__()
