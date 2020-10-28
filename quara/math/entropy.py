from typing import List, Union

import numpy as np


def round_varz(
    z: Union[float, np.float64], eps: Union[float, np.float64]
) -> np.float64:
    # validation
    if type(z) != float and type(z) != np.float64:
        raise ValueError(
            f"z must be real numbers(float or np.float64). dtype of z is {type(z)}"
        )
    if z < 0:
        raise ValueError(f"z must be non-negative numbers. z is {z}")
    if type(eps) != float and type(z) != np.float64:
        raise ValueError(
            f"eps must be real numbers(float or np.float64). dtype of eps is {type(eps)}"
        )
    if eps < 0:
        raise ValueError(f"eps must be non-negative numbers. eps is {eps}")

    # round
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

    val = 0
    for q, p in zip(prob_dist_q, prob_dist_p):
        if q >= eps_q:
            q_round = round_varz(q, eps_q)
            p_round = round_varz(p, eps_p)
            q_div_p_round = round_varz(q_round / p_round, eps_p)
            val += q_round * np.log(q_div_p_round)

    return val


def gradient_relative_entropy_2nd(
    prob_dist_q: np.array,
    prob_dist_p: np.array,
    gradient_prob_dist_ps: np.array,  # TODO 2D
    eps_q: float = None,
    eps_p: float = None,
) -> np.array:
    if eps_q == None:
        eps_q = 1e-10
    if eps_p == None:
        eps_p = 1e-10

    val = np.zeros(gradient_prob_dist_ps.shape[1], dtype=np.float64)
    for q, p, grad_p in zip(prob_dist_q, prob_dist_p, gradient_prob_dist_ps):
        if q >= eps_q:
            p_round = round_varz(p, eps_p)
            val += -q * grad_p / p_round

    return val


def hessian_relative_entropy_2nd(
    prob_dist_q: np.array,
    prob_dist_p: np.array,
    gradient_prob_dist_ps: np.array,  # TODO 2D
    hessian_prob_dist_ps: np.array,  # TODO 3D
    eps_q: float = None,
    eps_p: float = None,
) -> float:
    if eps_q == None:
        eps_q = 1e-10
    if eps_p == None:
        eps_p = 1e-10

    val = 0
    for q, p, grad_p, hess_p in zip(
        prob_dist_q, prob_dist_p, gradient_prob_dist_ps, hessian_prob_dist_ps
    ):
        if q >= eps_q:
            p_round = round_varz(p, eps_p)
            mat_grad_p = np.array([grad_p], dtype=np.float64)
            val += (
                -q * hess_p / p_round + (q / p_round ** 2) * mat_grad_p.T @ mat_grad_p
            )

    return val
