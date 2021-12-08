import numpy as np
import pytest

from quara.math import probability


def test_validate_prob_dist():
    # case 1: success
    prob_dist = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    probability.validate_prob_dist(prob_dist)

    # case 2: sum is not 1
    prob_dist = np.array([0.1, 0.2, 0.3, 0.5], dtype=np.float64)
    with pytest.raises(ValueError):
        probability.validate_prob_dist(prob_dist)

    # case 3: negative number exists
    prob_dist = np.array([-0.1, 0.2, 0.3, 0.5], dtype=np.float64)
    with pytest.raises(ValueError):
        probability.validate_prob_dist(prob_dist)

    # case 4: negative number exists and not raise error
    prob_dist = np.array([-0.1, 0.2, 0.3, 0.5], dtype=np.float64)
    probability.validate_prob_dist(prob_dist, raise_error=False)
