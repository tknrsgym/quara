from typing import List

import numpy as np


def round_varz(z: np.float64, eps: np.float64) -> np.float64:
    # TODO validation, float, non-negative
    val = max(z, eps)
    return val


def relative_entropy(
    prob_dist_q: np.array,
    prob_dist_p: np.array,
    eps_q: float = None,
    eps_p: float = None,
) -> float:
    if eps_q == None:
        eps_q = 1e-10
    if eps_p == None:
        eps_p = 1e-10

    # TODO validate each prob_dist_q > 0

    relative_entropy = 0
    for p, q in zip(prob_dist_p, prob_dist_q):
        if q >= eps_q:
            q_round = round_varz(q, eps_q)
            p_round = round_varz(p, eps_p)
            q_div_p_round = round_varz(q / p_round, eps_p)
            relative_entropy += q_round * np.log(q_div_p_round)

    return relative_entropy


"""
def gradient_relativeEntropy_2nd(
    prob_dist_q: np.array,
    prob_dist_p: np.array,
    gradient_prob_dist_p: np.array,
) -> float:
"""
