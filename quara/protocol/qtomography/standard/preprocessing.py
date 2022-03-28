from multiprocessing.sharedctypes import Value
from typing import List, Tuple
import numpy as np

from quara.objects.qoperation import QOperation
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate
from quara.objects.mprocess import MProcess
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.standard_povmt import StandardPovmt
from quara.protocol.qtomography.standard.standard_qpt import StandardQpt
from quara.protocol.qtomography.standard.standard_qmpt import StandardQmpt


def type_standard_qtomography(sqt: StandardQTomography) -> str:
    """Returns string of Type of StandardQTomography.

    Parameters
    ----------
    sqt : StandardQTomography
        StandardQTomography object to get string of Type.

    Returns
    -------
    str
        string of Type of StandardQTomography.

    Raises
    ------
    TypeError
        Type of StandardQTomography is invalid.
    """
    if type(sqt) == StandardQst:
        t = "state"
    elif type(sqt) == StandardPovmt:
        t = "povm"
    elif type(sqt) == StandardQpt:
        t = "gate"
    elif type(sqt) == StandardQmpt:
        t = "mprocess"
    else:
        raise TypeError(
            f"Type of StandardQTomography is invalid. Type of sqt={type(sqt)}"
        )
    return t


def extract_nums_from_empi_dists(empi_dists: List[Tuple[int, np.ndarray]]) -> List[int]:
    """Returns a list of numbers of data extracted from empirical distributions.

    Parameters
    ----------
    empi_dists : List[Tuple[int, np.array]]
        A empirical distributions

    Returns
    -------
    List[int]
        A list of numbers of data
    """
    nums = [empi_dist[0] for empi_dist in empi_dists]
    return nums


def extract_prob_dists_from_empi_dists(
    empi_dists: List[Tuple[int, np.ndarray]]
) -> List[np.array]:
    """Returns a list of probbility distributions extracted from empirical distributions.

    Parameters
    ----------
    empi_dists : List[Tuple[int, np.array]]

    Returns
    -------
    List[np.array]
        A list of probability distributions
    """
    prob_dists = [empi_dist[1] for empi_dist in empi_dists]
    return prob_dists


def calc_total_num(nums: List[int]) -> int:
    """Returns the total number in the list.

    Parameters
    ----------
    nums : List[int]
        a list of non-negative integers, Ni.

    Returns
    -------
    int
        sum_i Ni
    """
    n = sum(nums)
    return n


def calc_num_ratios(nums: List[int]) -> List[float]:
    """Returns number ratios.

    Parameters
    ----------
    nums : List[int]
        a list of non-negative integers

    Returns
    -------
    List[float]
        a list of racios, ci:=Ni/N, where N:=sum_i Ni

    Raises
    ------
    ValueError
        All elements of nums must be non-negative.
    """
    for i, Ni in enumerate(nums):
        if Ni < 0:
            error_message = (
                "All elements of nums must be non-negative. nums[{i}] = {Ni}"
            )
            raise ValueError(error_message)

    N = calc_total_num(nums)

    cs = []
    for Ni in nums:
        ci = Ni / N
        cs.append(ci)
    return cs
