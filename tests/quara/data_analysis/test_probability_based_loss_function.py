from abc import abstractmethod
from typing import List

import numpy as np
import numpy.testing as npt
import pytest

from quara.data_analysis.probability_based_loss_function import (
    ProbabilityBasedLossFunction,
)


# parameters for test
mat_p = np.array(
    [
        [1, 1, 0, 0],
        [1, -1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, -1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, -1],
    ],
    dtype=np.float64,
) / np.sqrt(2)


def _func_prob_dist(index: int):
    def _process(var: np.array) -> np.float64:
        return np.dot(mat_p[index], var)

    return _process


def func_prob_dists(x: int = None):
    funcs = []
    for index in range(len(mat_p)):
        funcs.append(_func_prob_dist(index))
    return funcs


prob_dists_q = [
    np.array([0.5, 0.5], dtype=np.float64),
    np.array([0.5, 0.5], dtype=np.float64),
    np.array([1.0, 0.0], dtype=np.float64),
]


class TestProbabilityBasedLossFunction:
    def test_access_prob_dists_q(self):
        loss_func = ProbabilityBasedLossFunction(4)
        assert loss_func.prob_dists_q is None

        loss_func = ProbabilityBasedLossFunction(4, prob_dists_q=prob_dists_q)
        assert len(loss_func.prob_dists_q) == 3

        # Test that "prob_dists_q" cannot be updated
        with pytest.raises(AttributeError):
            loss_func.prob_dists_q = prob_dists_q

    def test_access_on_prob_dists_q(self):
        loss_func = ProbabilityBasedLossFunction(4)
        assert loss_func.on_prob_dists_q == False

        loss_func = ProbabilityBasedLossFunction(4, prob_dists_q=prob_dists_q)
        assert loss_func.on_prob_dists_q == True

        # Test that "on_prob_dists_q" cannot be updated
        with pytest.raises(AttributeError):
            loss_func.on_prob_dists_q = False

    def test_set_prob_dists_q(self):
        loss_func = ProbabilityBasedLossFunction(4)
        assert loss_func.prob_dists_q is None

        loss_func.set_prob_dists_q(prob_dists_q)
        assert len(loss_func.prob_dists_q) == 3
