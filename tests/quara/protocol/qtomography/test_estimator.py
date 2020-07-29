import numpy as np
import numpy.testing as npt

from quara.protocol.qtomography.estimator import EstimationResult


class TestEstimationResult:
    def get_estimation_result(self):
        computation_times = [0.1, 0.2]
        result = EstimationResult(computation_times)
        return result

    def test_access_computation_time(self):
        result = self.get_estimation_result()
        actual = result.computation_time
        expected = 0.1
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_access_computation_times(self):
        result = self.get_estimation_result()
        actual = result.computation_times
        expected = [0.1, 0.2]
        for a, e in zip(actual, expected):
            npt.assert_almost_equal(a, e, decimal=15)
