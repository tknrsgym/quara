from typing import List, Union
import numpy as np

# quara
from quara.objects.composite_system import CompositeSystem
from quara.objects.qoperation import QOperation
from quara.objects.state import (
    State,
    convert_var_to_state,
)
from quara.objects.povm import (
    Povm,
)
from quara.objects.gate import (
    Gate,
)

# cvxpy
import cvxpy as cp
from cvxpy.expressions.variable import Variable as CvxpyVariable
from cvxpy.constraints.constraint import Constraint as CvxpyConstraint


def num_cvxpy_variable(t: str, dim: int, num_outcomes: int = None) -> int:
    """
    Returns number of optimization variable

    Parameters
    ----------
    t:str
        Type of QOperation. "state", "povm", "gate", or "mprocess"

    dim:int
        Dimension of the system

    num_outcoms:int=None
        Number of outcomes for t = "povm"

    Returns
    ----------
    int
        Number of optimization variable
    """
    d = dim
    if t == "state":
        num = d * d - 1
    elif t == "povm":
        num = (num_outcomes - 1) * d * d
    elif t == "gate":
        num = d ** 4 - d * d
    elif t == "mprocess":
        raise NotImplementedError
    return num


def generate_cvxpy_variable(
    t: str, dim: int, num_outcomes: int = None
) -> CvxpyVariable:
    """
    Returns number of optimization variable

    Parameters
    ----------
    t:str
        Type of QOperation. "state", "povm", "gate", or "mprocess"

    dim:int
        Dimension of the system

    num_outcoms:int=None
        Number of outcomes for t = "povm"

    Returns
    ----------
    CvxpyVariable
        Variable of CVXPY
    """
    num = num_cvxpy_variable(t, dim, num_outcomes)
    var = cp.Variable(num)
    return var


def generate_cvxpy_constraints_from_cvxpy_variable(
    c_sys: CompositeSystem, t: str, var: CvxpyVariable, num_outcomes: int = None
) -> CvxpyConstraint:
    """
    Returns Constraint of CVXPY

    Parameters
    ----------
    c_sys: CompositeSystem
        Composite system that the QOperation acts on

    t:str
        Type of QOperation. "state", "povm", "gate", or "mprocess"

    var:CvxpyVariable
        Variable of CVXPY

    num_outcoms:int=None
        Number of outcomes for t = "povm"

    Returns
    ----------
    CvxpyConstraint
        Constraint of CVXPY
    """
    if t == "state":
        constraints = [dmat_from_var(c_sys, var) >> 0]
    elif t == "povm":
        constraints = []
        for x in range(num_outcomes):
            constraints.append(povm_element_from_var(c_sys, num_outcomes, x, var) >> 0)
    elif t == "gate":
        constraints = [choi_from_var(c_sys, var) >> 0]
    return constraints


# Conversion from Variable of quara or cvxpy to Matrix Representation


def dmat_from_var(
    c_sys: CompositeSystem, var: Union[np.ndarray, CvxpyVariable]
) -> Union[np.ndarray, CvxpyVariable]:
    d = c_sys.dim
    basis = c_sys.basis()
    expr = np.eye(d) / d
    num_variable = d * d - 1
    for a in range(num_variable):
        expr = expr + var[a] * basis[a + 1]
    return expr


def povm_element_from_var(
    c_sys: CompositeSystem,
    num_outcomes: int,
    x: int,
    var: Union[np.ndarray, CvxpyVariable],
) -> Union[np.ndarray, CvxpyVariable]:
    d = c_sys.dim
    m = num_outcomes
    if 0 <= x and x < m - 1:
        vec = var[x * d * d : (x + 1) * d * d]
    elif x == m - 1:
        vec = np.zeros(d * d)
        vec[0] = np.sqrt(d)
        for y in range(m - 1):
            vec = vec - var[y * d * d : (y + 1) * d * d]
    basis = c_sys.basis()
    expr = np.zeros(shape=(d, d), dtype=np.complex128)
    for a in range(d * d):
        expr = expr + vec[a] * basis[a]
    return expr


def choi_from_var(
    c_sys: CompositeSystem, var: Union[np.ndarray, CvxpyVariable]
) -> Union[np.ndarray, CvxpyVariable]:
    d = c_sys.dim
    choi = np.eye(d * d, dtype=np.complex128) / d
    for a in range(1, d * d):
        for b in range(d * d):
            mat = c_sys.basis_basisconjugate((a, b))
            coeff = var[(a - 1) * d * d + b]
            choi += coeff * mat
    return choi


# Conversion from CVXPY.Variavle to Quara.QOparation
def convert_cxvpy_variable_to_state_vec(dim: int, var: CvxpyVariable) -> np.ndarray:
    d = dim
    l = [1 / np.sqrt(d)]
    l.extend(var.value)
    vec = np.array(l)
    return vec


def convert_cxvpy_variable_to_povm_vecs(
    dim: int, num_outcomes: int, var: CvxpyVariable
) -> np.ndarray:
    d = dim
    vec_sum = np.zeros(d * d)
    vec_sum[0] = np.sqrt(d)
    vecs = []
    # x = 0 ~ m-2
    for x in range(num_outcomes - 1):
        l = var.value[x * d * d : (x + 1) * d * d]
        vec = np.array(l)
        vecs.append(vec)
        vec_sum = vec_sum - vec
    # x = m-1
    vecs.append(vec_sum)
    return vecs


def convert_cxvpy_variable_to_gate_hs(dim: int, var: CvxpyVariable) -> np.ndarray:
    d = dim
    ll = []
    vec = np.zeros(d * d, dtype=np.float64)
    vec[0] = 1
    ll.append(vec)
    for a in range(d * d - 1):
        vec = var[a * d * d : (a + 1) * d * d].value
        ll.append(vec)
    hs = np.array(ll)
    return hs


def convert_cvxpy_variable_to_state(
    c_sys: CompositeSystem, var: CvxpyVariable
) -> State:
    vec = convert_cxvpy_variable_to_state_vec(c_sys, var)
    state = convert_var_to_state(
        c_sys=c_sys,
        var=vec,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return state


def convert_cvxpy_variable_to_povm(c_sys: CompositeSystem, var: CvxpyVariable) -> Povm:
    vecs = convert_cxvpy_variable_to_povm_vecs(c_sys, var)
    povm = Povm(
        c_sys=c_sys,
        vecs=vecs,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return povm


def convert_cvxpy_variable_to_gate(c_sys: CompositeSystem, var: CvxpyVariable) -> Gate:
    hs = convert_cxvpy_variable_to_gate_hs(c_sys, var)
    gate = Gate(
        c_sys=c_sys,
        hs=hs,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return gate


def convert_cvxpy_variable_to_qoperation(
    t: str, c_sys: CompositeSystem, var: CvxpyVariable
) -> QOperation:
    """
    t:str
        type of estimate, "state", "povm", "gate".
    """
    if t == "state":
        qop = convert_cvxpy_variable_to_state(c_sys, var)
    elif t == "povm":
        qop = convert_cvxpy_variable_to_povm(c_sys, var)
    elif t == "gate":
        qop = convert_cvxpy_variable_to_gate(c_sys, var)
    return qop


# Conversion from Variavle in Quara to Quara.QOparation
def convert_quara_variable_to_state_vec(dim: int, var: np.ndarray) -> np.ndarray:
    d = dim
    l = [1 / np.sqrt(d)]
    l.extend(var)
    vec = np.array(l)
    return vec


def convert_quara_variable_to_povm_vecs(
    dim: int, num_outcomes: int, var: np.ndarray
) -> List[np.ndarray]:
    d = dim
    vec_sum = np.zeros(d * d)
    vec_sum[0] = np.sqrt(d)
    vecs = []
    # x = 0 ~ m-2
    for x in range(num_outcomes - 1):
        l = var[x * d * d : (x + 1) * d * d]
        vec = np.array(l)
        vecs.append(vec)
        vec_sum = vec_sum - vec
    # x = m-1
    vecs.append(vec_sum)
    return vecs


def convert_quara_variable_to_gate_hs(dim: int, var: np.ndarray) -> np.ndarray:
    d = dim
    ll = []
    vec = np.zeros(d * d, dtype=np.float64)
    vec[0] = 1
    ll.append(vec)
    for a in range(d * d - 1):
        vec = var[a * d * d : (a + 1) * d * d]
        ll.append(vec)
    hs = np.array(ll)
    return hs


def convert_quara_variable_to_state(c_sys: CompositeSystem, var: np.ndarray) -> State:
    vec = convert_quara_variable_to_state_vec(c_sys, var)
    state = convert_var_to_state(
        c_sys=c_sys,
        var=vec,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return state


def convert_quara_variable_to_povm(c_sys: CompositeSystem, var: np.ndarray) -> Povm:
    vecs = convert_quara_variable_to_povm_vecs(c_sys, var)
    povm = Povm(
        c_sys=c_sys,
        vecs=vecs,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return povm


def convert_quara_variable_to_gate(c_sys: CompositeSystem, var) -> Gate:
    hs = convert_quara_variable_to_gate_hs(c_sys, var)
    gate = Gate(
        c_sys=c_sys,
        hs=hs,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return gate


def convert_quara_variable_to_qoperation(
    t: str, c_sys: CompositeSystem, var
) -> QOperation:
    if t == "state":
        qop = convert_quara_variable_to_state(c_sys, var)
    elif t == "povm":
        qop = convert_quara_variable_to_povm(c_sys, var)
    elif t == "gate":
        qop = convert_quara_variable_to_gate(c_sys, var)
    return qop
