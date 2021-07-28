import numpy as np


def validate_prob_dist(prob_dist: np.ndarray, eps: float = None) -> None:
    """validate the probability distribution.

    Parameters
    ----------
    prob_dist : np.ndarray
        the probability distribution.
    eps : float, optional
        the absolute tolerance parameter, by default 1e-8.
        checks ``absolute(the sum of probabilities - 1) <= atol`` in this function.

    Raises
    ------
    ValueError
        some elements of prob_dist are negative numbers.
    ValueError
        the sum of prob_dist is not 1.
    """
    if eps == None:
        eps = 1e-8

    # whether each probability is a positive number.
    for prob in prob_dist:
        if prob < 0:
            raise ValueError(
                f"each probability must be a non-negative number. there is {prob} in a probability distribution"
            )

    # whether the sum of probabilities equals 1.
    sum = np.sum(prob_dist)
    if not np.isclose(sum, 1.0, atol=eps, rtol=0.0):
        raise ValueError(f"the sum of prob_dist must be 1. the sum of prob_dist={sum}")
