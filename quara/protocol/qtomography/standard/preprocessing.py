from abc import abstractmethod
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


class StandardQTomographyPreprocessing:
    def __init__(self, sqt: StandardQTomography, eps_prob_zero: float = 10 ** (-12)):
        # Type of Standard Q Tomography
        type_estimate = type_standard_qtomography(sqt)
        self.set_type_estimate(t=type_estimate)
        self.set_sqt(sqt=sqt)
        self.set_eps_prob_zero(eps_prob_zero)

    def set_type_estimate(self, t: str):
        self._type_estimate = t

    def set_sqt(self, sqt: StandardQTomography):
        self._sqt = sqt

    def set_prob_dists(self, p: List[np.ndarray]):
        q = p
        for i, pi in enumerate(p):
            for x, pi_x in enumerate(pi):
                if pi_x < self._eps_prob_zero:
                    q[i][x] = 0
                elif pi_x > 1.0:
                    q[i][x] = 1
        self._prob_dists = q

    def set_eps_prob_zero(self, eps: float):
        self._eps_prob_zero = eps

    def set_from_empi_dists(self, empi_dists: List[Tuple[int, np.ndarray]]):
        self._nums_data = extract_nums_from_empi_dists(empi_dists)
        self._num_data_total = calc_total_num(self.nums_data)
        self._num_data_ratios = calc_num_ratios(self.nums_data)
        self.set_prob_dists(extract_prob_dists_from_empi_dists(empi_dists))
        # self._prob_dists = extract_prob_dists_from_empi_dists(empi_dists)

    @property
    def type_estimate(self):
        return self._type_estimate

    @property
    def sqt(self):
        return self._sqt

    @property
    def prob_dists(self):
        return self._prob_dists

    @property
    def eps_prob_zero(self):
        return self._eps_prob_zero

    @property
    def nums_data(self):
        return self._nums_data

    @property
    def num_data_total(self):
        return self._num_data_total

    @property
    def num_data_ratios(self):
        return self._num_data_ratios

    @property
    def composite_system(self):
        t = self.type_estimate
        if t == "state":
            c_sys = self.sqt._experiment.povms[0]._composite_system
        elif t == "povm":
            c_sys = self.sqt._experiment.states[0]._composite_system
        elif t == "gate":
            c_sys = self.sqt._experiment.povms[0]._composite_system
        elif t == "mprocess":
            c_sys = self.sqt._experiment.povms[0]._composite_system    
        return c_sys

    def basis(self):
        return self.composite_system().basis()

    def calc_prob_dist_from_var(self, var, i):
        vec = self.to_vec_from_var(var)
        p = self.sqt.get_coeffs_1st_mat(i) @ vec + self.sqt.get_coeffs_0th_vec(i)
        for x, px in enumerate(p):
            if px < 10 ** (-12):
                p[x] = 0.0

        return p

    def dim_sys(self):
        t = self.type_estimate
        if t == "state":
            d = self.sqt._experiment.povms[0].dim
        elif t == "povm":
            d = self.sqt._experiment.states[0].dim
        elif t == "gate":
            d = self.sqt._experiment.povms[0].dim
        elif t == "mprocess":
            d = self.sqt._experiment.povms[0].dim    
        return d

    def num_outcomes_estimate(self):
        assert self.type_estimate == "povm" or self.type_estimate == "mprocess"
        num = self.sqt.num_outcomes(schedule_index=0)
        return num



def type_standard_qtomography(sqt: StandardQTomography) -> str:
    """ Return Type of Standard Q Tomography """
    if type(sqt) == StandardQst:
        t = "state"
    elif type(sqt) == StandardPovmt:
        t = "povm"
    elif type(sqt) == StandardQpt:
        t = "gate"
    elif type(sqt) == StandardQmpt:
        t = "mprocess"
    else:
        raise ValueError(f"Type of StandardQTomography is invalid!")
    return t


def is_prob_dist(p: np.ndarray, eps: float = 10 ** (-12)) -> bool:
    """return True if p is a probability distribution and False if not. """
    assert p.dtype == float
    assert p.ndim == 1

    res = True
    for px in p:
        if px < -eps or px > 1 + eps:
            res = False

    s = np.sum(p)
    if abs(1 - s) > eps:
        res = False

    return res


def which_type_prob_dist(p: np.ndarray, eps: float = 10 ** (-12)) -> int:
    """return an integer 0, 1, 2 for specifying the type of a probability distribution.

    Type-0: includes 0, but not include 1.
    Type-1: includes 1.
    TYpe-2: otherwise.

    Parameters
    ----------
    p: np.ndarray
        1D, real
        required to be a probability distribution.

    eps: float = 10 ** (-12)

    Returns
    ----------
    int
        0, 1, or 2
    """
    assert is_prob_dist(p)

    t = 2
    is_zero = False
    for px in p:
        if px > 1 - eps:
            t = 1
            break
        elif px < eps:
            is_zero = True
    if t == 2 and is_zero == True:
        t = 0

    return t


def is_prob_dist(p: np.ndarray, eps: float = 10 ** (-12)) -> bool:
    """return True if p is a probability distribution and False if not. """
    assert p.dtype == float
    assert p.ndim == 1

    res = True
    for px in p:
        if px < -eps or px > 1 + eps:
            res = False

    s = np.sum(p)
    if abs(1 - s) > eps:
        res = False

    return res


def which_type_prob_dist(p: np.ndarray, eps: float = 10 ** (-12)) -> int:
    """return an integer 0, 1, 2 for specifying the type of a probability distribution.

    Type-0: includes 0, but not include 1.
    Type-1: includes 1.
    TYpe-2: otherwise.

    Parameters
    ----------
    p: np.ndarray
        1D, real
        required to be a probability distribution.

    eps: float = 10 ** (-12)

    Returns
    ----------
    int
        0, 1, or 2
    """
    assert is_prob_dist(p)

    t = 2
    is_zero = False
    for px in p:
        if px > 1 - eps:
            t = 1
            break
        elif px < eps:
            is_zero = True
    if t == 2 and is_zero == True:
        t = 0

    return t


def which_type_prob_dists(ps: List[np.ndarray], eps: float = 10 ** (-12)) -> List[int]:
    """return a list of types for list of probability distributions."""
    l = []
    for p in ps:
        t = which_type_prob_dist(p, eps)
        l.append(t)

    return l


# get_indices_removed


def get_indices_lists_removed(
    ps: List[np.ndarray], eps: np.float64 = 10 ** (-12)
) -> List[List[List[int]]]:
    """Return a list of indices lists to be removed.

    Parameters
    ==========
    ps: List[np.ndarray]
        a list of probaboilty distributions

    eps: np.float64
        a threshold value used at identifying 0 or 1.

    Returns
    =======
    List[List[List[int]]]
        list of indices lists
    """
    l = []
    for p in ps:
        t = which_type_prob_dist(p, eps)
        if t == 2:
            indices = get_indices_removed_type_two(p)
            l.append([indices])
        elif t == 1:
            indices = get_indices_removed_type_one(p, eps)
            l.append([indices])
        else:  # t == 0
            indices_list = get_indices_list_removed_type_zero(p, eps)
            l.append(indices_list)
    return l


def get_indices_removed(t: int, p: np.ndarray) -> List[int]:
    """return indices to be removed. p is assumed to be a probability distribution with type-0.

    Parameters
    ==========
    t: int
        type of a probability distribution, to be in [0, 1, 2]

    p: np.ndarray
        a probability distribution

    Returns
    =======
    List[int]
        the indices removed
    """
    assert t in [0, 1, 2]
    if t == 0:
        indices_removed = get_indices_removed_type_zero(p)
    elif t == 1:
        indices_removed = get_indices_removed_type_one(p)
    elif t == 2:
        indices_removed = get_indices_removed_type_two(p)
    return indices_removed


def get_indices_removed_type_zero(p: np.ndarray) -> List[int]:
    """return indices to be removed for type zero. p is assumed to be a probability distribution with type-0.

    Parameters
    ==========
    p: np.ndarray
        a probability distribution

    Returns
    =======
    List[int]
        the indices removed
    """
    indices_zero = get_indices_value_zero_type_zero(p)
    indices_removed = indices_zero
    length = len(p)
    index_last = length - 1
    for l in range(length - 1, -1, -1):
        if l not in indices_zero:
            index_last = l
            break
    indices_removed.append(index_last)
    return sorted(indices_removed)


def get_indices_removed_type_one(
    p: np.ndarray, eps: np.float64 = 10 ** (-12)
) -> List[int]:
    """p is assumed to be a probability distribution with type-1."""
    indices_removed = []
    for x, px in enumerate(p):
        if px <= 1 - eps:
            indices_removed.append(x)
    return sorted(indices_removed)


def get_indices_removed_type_two(p: np.ndarray) -> List[int]:
    """p is assumed to be a probability distribution with type-2."""
    length = len(p)
    indices_removed = [length - 1]
    return sorted(indices_removed)


def get_indices_list_removed_type_zero(
    p: np.ndarray, eps: np.float64 = 10 ** (-12)
) -> List[List[int]]:
    """p is assumed to be a probability distribution with type zero."""
    indices_list_remain = get_indices_list_remain_type_zero(p, eps)
    indices_list_removed = []
    for l in reversed(indices_list_remain):
        a = [k for k in list(range(len(p))) if k not in l]
        indices_list_removed.append(a)
    return indices_list_removed


# get_indices_remain


def get_indices_remain(t: int, p: np.ndarray) -> List[int]:
    """return indices to remain. p is assumed to be a probability distribution with type-0.

    Parameters
    ==========
    t: int
        type of a probability distribution, to be in [0, 1, 2]

    p: np.ndarray
        a probability distribution

    Returns
    =======
    List[int]
        the indices to remain
    """
    assert t in [0, 1, 2]
    if t == 0:
        indices_remain = get_indices_remain_type_zero(p)
    elif t == 1:
        indices_remain = get_indices_remain_type_one(p)
    elif t == 2:
        indices_remain = get_indices_remain_type_two(p)
    return indices_remain


def get_indices_remain_type_zero(p: np.ndarray) -> List[int]:
    """p is assumed to be a probability distribution with type-0."""
    indices = list(range(len(p)))
    indices_removed = get_indices_removed_type_zero(p)
    for i in indices_removed:
        indices.remove(i)
    return indices


def get_indices_remain_type_one(p: np.ndarray) -> List[int]:
    """p is assumed to be a probability distribution with type-1."""
    indices = list(range(len(p)))
    indices_removed = get_indices_removed_type_one(p)
    for i in indices_removed:
        indices.remove(i)
    return indices


def get_indices_remain_type_two(p: np.ndarray) -> List[int]:
    """p is assumed to be a probability distribution with type-2."""
    indices = list(range(len(p)))
    indices_removed = get_indices_removed_type_two(p)
    for i in indices_removed:
        indices.remove(i)
    return indices


def get_indices_value_one_type_one(
    p: np.ndarray, eps: np.float64 = 10 ** (-12)
) -> List[int]:
    """p is assumed to be a probability distribution in type one."""
    indices_one = []
    for x, px in enumerate(p):
        if px > 1 - eps:
            indices_one.append(x)
            break
    return indices_one


def get_indices_value_zero_type_zero(
    p: np.ndarray, eps: np.float64 = 10 ** (-12)
) -> List[int]:
    """p is assumed to be a probability distributionin type zero."""
    indices_zero = []
    for x, px in enumerate(p):
        if px < eps:
            indices_zero.append(x)
    return indices_zero


def get_indices_value_nonzero_type_zero(
    p: np.ndarray, eps: np.float64 = 10 ** (-12)
) -> List[int]:
    """p is assumed to be a probability distribution in type zero."""
    indices_value_zero = get_indices_value_zero_type_zero(p, eps)
    indices = list(range(len(p)))
    for i in indices_value_zero:
        indices.remove(i)
    return indices


def get_indices_list_remain_type_zero(
    p: np.ndarray, eps: np.float64 = 10 ** (-12)
) -> List[List[int]]:
    """p is assumed to be a probability distribution in type zero."""
    indices_list_remain_type_zero = []
    indices_value_nonzero = get_indices_value_nonzero_type_zero(p, eps)
    for j in reversed(indices_value_nonzero):
        l_dummy = indices_value_nonzero[:]
        l_dummy.remove(j)
        indices_list_remain_type_zero.append(l_dummy)
    return indices_list_remain_type_zero


def extract_nums_from_empi_dists(empi_dists: List[Tuple[int, np.ndarray]]) -> List[int]:
    """returns a list of numbers of data extracted from empirical distributions.

    Parameters
    ----------
    empi_dists: List[Tuple[int, np.array]]

    Returns
    -------
    List[int]
        A list of numbers of data
    """
    nums = []
    for empi_dist in empi_dists:
        nums.append(empi_dist[0])
    return nums


def extract_prob_dists_from_empi_dists(
    empi_dists: List[Tuple[int, np.ndarray]]
) -> List[np.array]:
    """returns a list of probbility distributions extracted from empirical distributions.

    Parameters
    ----------
    empi_dists: List[Tuple[int, np.array]]

    Returns
    -------
    List[np.array]
        A list of probability distributions
    """
    prob_dists = []
    for empi_dist in empi_dists:
        prob_dists.append(empi_dist[1])
    return prob_dists


def calc_total_num(nums: List[int]) -> int:
    """Return the total number in the list.

    Parameters
    ==========
    nums:List[int]
        a list of non-negative integers, Ni.

    Returns
    =======
    int
        sum_i Ni
    """
    n = sum(nums)
    return n


def calc_num_ratios(nums: List[int]) -> List[float]:
    """Return number ratios.

    Parameters
    ==========
    nums:List[int]
        a list of non-negative integers

    Returns
    =======
    List[float]
        a list of racios, ci:=Ni/N, where N:=sum_i Ni
    """
    for Ni in nums:
        assert Ni >= 0

    N = calc_total_num(nums)
    assert N > 0
    cs = []
    for Ni in nums:
        ci = Ni / N
        cs.append(ci)
    return cs


def calc_inverse_variance_matrix_from_vector(
    v: np.ndarray, eps: float = 10 ** (-12)
) -> np.ndarray:
    """Return the inverse variance matrix from a random variable vector.

    V^-1 := diag(v)^-1 + ones ones^T /(1 - ones^T v)

    Parameters
    ==========
    v:np.ndarray
        a random variable vector

    eps:float = 10**(-12)
        threshold for avoding a divergence of 1/v_i or 1/(1-ones^T v)

    Returns
    =======
    np.ndarray
        inverse variance matrix
    """
    s = np.sum(v)
    assert abs(1 - s) > eps

    for vi in v:
        assert abs(vi) > eps

    n = len(v)
    mat = np.diag(1 / v) + np.ones((n, n)) / (1 - s)

    return mat


def combine_nums_prob_dists(
    nums: List[int], prob_dists: List[np.ndarray]
) -> List[Tuple[int, np.ndarray]]:
    assert len(nums) == len(prob_dists)
    res = []
    for i in range(len(nums)):
        tup = (nums[i], prob_dists[i])
        res.append(tup)
    return res


def squared_distance_state(state1: State, state2: State) -> float:
    diff = state1.vec - state2.vec
    res = np.inner(diff, diff)
    return res


def squared_distance_povm(povm1: Povm, povm2: Povm) -> float:
    assert povm1.num_outcomes == povm2.num_outcomes
    res = 0.0
    for x in range(povm1.num_outcomes):
        diff = povm1.vecs[x] - povm2.vecs[x]
        res += np.inner(diff, diff)
    res = res / povm1.num_outcomes
    return res


def squared_distance_gate(gate1: Gate, gate2: Gate) -> float:
    diff = gate1.hs - gate2.hs
    res = np.trace(diff.T @ diff)
    return res

def squared_distance_mprocess(mprocess1: MProcess, mprocess2: MProcess) -> float:
    assert mprocess1.num_outcomes == mprocess2.num_outcomes
    res = 0.0
    for x in range(mprocess1.num_outcomes):
        diff = mprocess1.hss[x] - mprocess2.hss[x]
        res += np.trace(diff.T @ diff)
    res = res / mprocess1.num_outcomes
    return res        


def squared_distance_qoperation(qop1: QOperation, qop2: QOperation) -> float:
    assert type(qop1) == type(qop2)
    if type(qop1) == State:
        res = squared_distance_state(qop1, qop2)
    elif type(qop1) == Povm:
        res = squared_distance_povm(qop1, qop2)
    elif type(qop1) == Gate:
        res = squared_distance_gate(qop1, qop2)
    elif type(qop1) == MProcess:
        res = squared_distance_mprocess(qop1, qop2)    
    else:
        raise ValueError(f"Type of qop1 or qop2 is invalid!")
    return res
