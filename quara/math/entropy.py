from typing import List, Union

import numpy as np

from quara.settings import Settings
from quara.utils.matrix_util import truncate_computational_fluctuation


def round_varz(
    z: Union[float, np.float64],
    eps: Union[float, np.float64],
    is_valid_required: bool = True,
    atol: float = None,
) -> np.float64:
    """returns max{z , eps}.

    This function to be used to avoid divergence.
    Both the arguments z and eps must be negative real numbers.

    Parameters
    ----------
    z : Union[float, np.float64]
        variable z.
    eps : Union[float, np.float64]
        variable eps.
    is_valid_required : bool, optional
        if is_valid_required is True, then check whetever z is a negative number, uses True by default.
    atol : float, optional
        the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.

    Returns
    -------
    np.float64
        max{z , eps}.

    Raises
    ------
    ValueError
        z is not a real number(float or np.float64).
    ValueError
        z is a negative number.
    ValueError
        eps is not a real number(float or np.float64).
    ValueError
        eps is a negative number.
    """
    atol = atol if atol else Settings.get_atol()

    # validation
    if type(z) != float and type(z) != np.float64:
        raise ValueError(
            f"z must be a real number(float or np.float64). dtype of z is {type(z)}"
        )
    if is_valid_required and z < 0 and not np.isclose(z, 0, atol=atol, rtol=0.0):
        raise ValueError(f"z must be a non-negative number. z is {z}")
    if type(eps) != float and type(z) != np.float64:
        raise ValueError(
            f"eps must be a real number(float or np.float64). dtype of eps is {type(eps)}"
        )
    if eps < 0:
        raise ValueError(f"eps must be a non-negative number. eps is {eps}")

    # round
    val = max(z, eps)
    return val


def round_varz_vector(
    z: np.ndarray,
    eps: Union[float, np.float64],
    is_valid_required: bool = True,
    atol: float = None,
) -> np.ndarray:
    """returns pointwise max{z , eps}.

    This function to be used to avoid divergence.
    Both the arguments z and eps must be negative real numbers.

    Parameters
    ----------
    z : Union[float, np.float64]
        variable z.
    eps : Union[float, np.float64]
        variable eps.
    is_valid_required : bool, optional
        if is_valid_required is True, then check whetever z is a negative number, uses True by default.
    atol : float, optional
        the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.

    Returns
    -------
    np.float64
        max{z , eps}.

    Raises
    ------
    ValueError
        z is not a real number(float or np.float64).
    ValueError
        z is a negative number.
    ValueError
        eps is not a real number(float or np.float64).
    ValueError
        eps is a negative number.
    """
    atol = atol if atol else Settings.get_atol()

    # validation
    if (
        is_valid_required
        and np.any(z < 0)
        and not np.all(np.isclose(z, 0, atol=atol, rtol=0.0))
    ):
        raise ValueError(f"z must consist of a non-negative number. z is {z}")
    if eps < 0:
        raise ValueError(f"eps must be a non-negative number. eps is {eps}")

    # round
    vector = np.where(z > eps, z, eps)
    return vector


def relative_entropy(
    prob_dist_q: np.ndarray,
    prob_dist_p: np.ndarray,
    eps_q: float = None,
    eps_p: float = None,
    is_valid_required: bool = True,
    atol: float = None,
) -> float:
    """returns relative entropy of probability distributions q and p.

    Parameters
    ----------
    prob_dist_q : np.ndarray
        a probability distribution q.
    prob_dist_p : np.ndarray
        a probability distribution p.
    eps_q : float, optional
        a parameter to avoid divergence about q, by default 1e-10
    eps_p : float, optional
        a parameter to avoid divergence about p, by default 1e-10
    is_valid_required : bool, optional
        if is_valid_required is True, then check whetever the entries of prob_dist_p is a negative number, uses True by default.
    atol : float, optional
        the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.

    Returns
    -------
    float
        relative entropy of probability distributions q and p.
    """
    if eps_q == None:
        eps_q = 1e-10
    if eps_p == None:
        eps_p = 1e-10

    val = 0
    for q, p in zip(prob_dist_q, prob_dist_p):
        if q >= eps_q:
            q_round = round_varz(q, eps_q, atol=atol)
            p_round = round_varz(
                p, eps_p, is_valid_required=is_valid_required, atol=atol
            )
            q_div_p_round = round_varz(q_round / p_round, eps_p, atol=atol)
            val += q_round * np.log(q_div_p_round)

    return val


def relative_entropy_vector(
    prob_dist_q: np.ndarray,
    prob_dist_p: np.ndarray,
    eps_q: float = None,
    eps_p: float = None,
    is_valid_required: bool = True,
    atol: float = None,
) -> np.ndarray:
    """returns pointwise relative entropy of probability distributions q and p.

    Parameters
    ----------
    prob_dist_q : np.ndarray
        a probability distribution q.
    prob_dist_p : np.ndarray
        a probability distribution p.
    eps_q : float, optional
        a parameter to avoid divergence about q, by default 1e-10
    eps_p : float, optional
        a parameter to avoid divergence about p, by default 1e-10
    is_valid_required : bool, optional
        if is_valid_required is True, then check whetever the entries of prob_dist_p is a negative number, uses True by default.
    atol : float, optional
        the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.

    Returns
    -------
    float
        relative entropy of probability distributions q and p.
    """
    if eps_q == None:
        eps_q = 1e-10
    if eps_p == None:
        eps_p = 1e-10

    q_truncated = truncate_computational_fluctuation(prob_dist_q, eps=eps_q)

    q_round = round_varz_vector(prob_dist_q, eps_q, atol=atol)
    p_round = round_varz_vector(
        prob_dist_p, eps_p, is_valid_required=is_valid_required, atol=atol
    )
    q_div_p_round = round_varz_vector(q_round / p_round, eps_p, atol=atol)
    vector = q_truncated * np.log(q_div_p_round)

    return vector


def gradient_relative_entropy_2nd(
    prob_dist_q: np.ndarray,
    prob_dist_p: np.ndarray,
    gradient_prob_dist_ps: np.ndarray,
    eps_q: float = None,
    eps_p: float = None,
    is_valid_required: bool = True,
    atol: float = None,
) -> np.ndarray:
    """returns gradient of relative entropy of probability distributions q and p.

    Parameters
    ----------
    prob_dist_q : np.ndarray
        a probability distribution q.
    prob_dist_p : np.ndarray
        a probability distribution p.
    gradient_prob_dist_ps : np.ndarray
        gradients of probability distribution p. ``ndim`` of this parameter must be 2 (list of gradients).
    eps_q : float, optional
        a parameter to avoid divergence about q, by default 1e-10
    eps_p : float, optional
        a parameter to avoid divergence about p, by default 1e-10
    is_valid_required : bool, optional
        if is_valid_required is True, then check whetever the entries of prob_dist_p is a negative number, uses True by default.
    atol : float, optional
        the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.

    Returns
    -------
    np.ndarray
        gradient of relative entropy of probability distributions q and p.
    """
    if eps_q == None:
        eps_q = 1e-10
    if eps_p == None:
        eps_p = 1e-10

    val = np.zeros(gradient_prob_dist_ps.shape[1], dtype=np.float64)
    for q, p, grad_p in zip(prob_dist_q, prob_dist_p, gradient_prob_dist_ps):
        if q >= eps_q:
            p_round = round_varz(
                p, eps_p, is_valid_required=is_valid_required, atol=atol
            )
            val += -q * grad_p / p_round

    return val


def gradient_relative_entropy_2nd_vector(
    prob_dist_q: np.ndarray,
    prob_dist_p: np.ndarray,
    gradient_prob_dist_ps: np.ndarray,
    eps_q: float = None,
    eps_p: float = None,
    is_valid_required: bool = True,
    atol: float = None,
) -> np.ndarray:
    """returns pointwise gradient of relative entropy of probability distributions q and p.

    Parameters
    ----------
    prob_dist_q : np.ndarray
        a probability distribution q.
    prob_dist_p : np.ndarray
        a probability distribution p.
    gradient_prob_dist_ps : np.ndarray
        gradients of probability distribution p. ``ndim`` of this parameter must be 2 (list of gradients).
    eps_q : float, optional
        a parameter to avoid divergence about q, by default 1e-10
    eps_p : float, optional
        a parameter to avoid divergence about p, by default 1e-10
    is_valid_required : bool, optional
        if is_valid_required is True, then check whetever the entries of prob_dist_p is a negative number, uses True by default.
    atol : float, optional
        the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.

    Returns
    -------
    np.ndarray
        gradient of relative entropy of probability distributions q and p.
    """
    if eps_q == None:
        eps_q = 1e-10
    if eps_p == None:
        eps_p = 1e-10

    q_truncated = truncate_computational_fluctuation(prob_dist_q, eps=eps_q)

    p_round = round_varz_vector(
        prob_dist_p, eps_p, is_valid_required=is_valid_required, atol=atol
    )
    coefficients = -q_truncated / p_round
    vectors = []
    for coefficient, gradient_prob_dist_p in zip(coefficients, gradient_prob_dist_ps):
        vectors.append(coefficient * gradient_prob_dist_p)

    vector = np.array(vectors, dtype=np.float64)
    return vector


def hessian_relative_entropy_2nd(
    prob_dist_q: np.ndarray,
    prob_dist_p: np.ndarray,
    gradient_prob_dist_ps: np.ndarray,
    hessian_prob_dist_ps: np.ndarray,
    eps_q: float = None,
    eps_p: float = None,
    is_valid_required: bool = True,
    atol: float = None,
) -> float:
    """returns Hessian of relative entropy of probability distributions q and p.

    Parameters
    ----------
    prob_dist_q : np.ndarray
        [description]
    prob_dist_p : np.ndarray
        [description]
    gradient_prob_dist_ps : np.ndarray
        gradients of probability distribution p. ``ndim`` of this parameter must be 2 (list of gradients).
    hessian_prob_dist_ps : np.ndarray
        Hessians of probability distribution p. ``ndim`` of this parameter must be 3 (list of Hessians).
    eps_q : float, optional
        a parameter to avoid divergence about q, by default 1e-10
    eps_p : float, optional
        a parameter to avoid divergence about p, by default 1e-10
    is_valid_required : bool, optional
        if is_valid_required is True, then check whetever the entries of prob_dist_p is a negative number, uses True by default.
    atol : float, optional
        the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.

    Returns
    -------
    float
        Hessian of relative entropy of probability distributions q and p.
    """
    if eps_q == None:
        eps_q = 1e-10
    if eps_p == None:
        eps_p = 1e-10

    val = 0
    for q, p, grad_p, hess_p in zip(
        prob_dist_q, prob_dist_p, gradient_prob_dist_ps, hessian_prob_dist_ps
    ):
        if q >= eps_q:
            p_round = round_varz(
                p, eps_p, is_valid_required=is_valid_required, atol=atol
            )
            mat_grad_p = np.array([grad_p], dtype=np.float64)
            val += (
                -q * hess_p / p_round + (q / p_round ** 2) * mat_grad_p.T @ mat_grad_p
            )

    return val
