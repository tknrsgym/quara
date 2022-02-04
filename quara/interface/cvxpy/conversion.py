from typing import List, Union

import numpy as np
import cvxpy as cp
from cvxpy.expressions.variable import Variable as CvxpyVariable
from cvxpy.constraints.constraint import Constraint as CvxpyConstraint

from quara.objects.composite_system import CompositeSystem
from quara.objects.qoperation import QOperation
from quara.objects.state import (
    State,
    convert_var_to_state,
    to_density_matrix_from_var,
)
from quara.objects.povm import (
    Povm,
    to_matrices_from_var,
)
from quara.objects.gate import (
    Gate,
    to_choi_from_var,
)


def get_valid_qopeartion_type() -> List[str]:
    """Returns the list of valid QOperation types.

    Returns
    -------
    List[str]
        the list of valid QOperation types.
    """
    l = ["state", "povm", "gate", "mprocess"]
    # l.append('diagonal_state')
    # l.append('nontp_gate')
    return l


def num_cvxpy_variable(t: str, dim: int, num_outcomes: int = None) -> int:
    """
    Returns number of optimization variable

    Parameters
    ----------
    t : str
        Type of QOperation. "state", "povm", "gate", "mprocess"
    dim : int
        Dimension of the system
    num_outcoms : int, optional
        Number of outcomes for t = "povm" or "mprocess", by default None

    Returns
    -------
    int
        Number of optimization variable

    Raises
    ------
    ValueError
        Unsupported type of QOperation is specified.
    ValueError
        dim is not a posivite number.
    ValueError
        Type of QOperation is povm, but num_outcomes is not specified.
    ValueError
        Type of QOperation is mprocess, but num_outcomes is not specified.
    """
    if t not in get_valid_qopeartion_type():
        raise ValueError(f"Unsupported type of QOperation is specified. type={t}")
    if dim <= 0:
        raise ValueError(f"dim must be a posivite number. dim={dim}")
    d = dim

    if t == "state":
        num = d * d - 1
    elif t == "povm":
        if num_outcomes is None:
            raise ValueError(
                "Type of QOperation is povm, but num_outcomes is not specified."
            )
        num = (num_outcomes - 1) * d * d
    elif t == "gate":
        num = d ** 4 - d * d
    elif t == "mprocess":
        if num_outcomes is None:
            raise ValueError(
                "Type of QOperation is mprocess, but num_outcomes is not specified."
            )
        num = num_outcomes * (d ** 4) - d ** 2

    return num


def generate_cvxpy_variable(
    t: str, dim: int, num_outcomes: int = None
) -> CvxpyVariable:
    """
    Returns number of optimization variable

    Parameters
    ----------
    t : str
        Type of QOperation. "state", "povm", "gate", "mprocess"
    dim : int
        Dimension of the system
    num_outcoms: int, optional
        Number of outcomes for t = "povm" or  "mprocess", by default None

    Returns
    -------
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
    assert t in get_valid_qopeartion_type()

    if t == "state":
        constraints = [dmat_from_var(c_sys, var) >> 0]
    elif t == "povm":
        constraints = []
        for x in range(num_outcomes):
            constraints.append(povm_element_from_var(c_sys, num_outcomes, x, var) >> 0)
    elif t == "gate":
        constraints = [choi_from_var(c_sys, var) >> 0]
    elif t == "mprocess":
        constraints = []
        for x in range(num_outcomes):
            constraints.append(
                mprocess_element_choi_from_var(c_sys, num_outcomes, x, var) >> 0
            )

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


def mprocess_element_choi_from_var(
    c_sys: CompositeSystem,
    num_outcomes: int,
    x: int,
    var: Union[np.ndarray, CvxpyVariable],
) -> Union[np.ndarray, CvxpyVariable]:
    d = c_sys.dim
    m = num_outcomes
    if 0 <= x and x < m - 1:
        vec = var[x * (d ** 4) : (x + 1) * (d ** 4)]
    elif x == m - 1:
        v = np.zeros(d ** 2)
        for y in range(m - 2):
            v += var[y * (d ** 4) : y * (d ** 4) + d ** 2]
        w = var[(m - 1) * (d ** 4) : m * (d ** 4)]
        vec = cp.hstack([v, w])

    choi = np.zeros((d ** 2, d ** 2))
    for a in range(d * d):
        for b in range(d * d):
            mat = c_sys.basis_basisconjugate((a, b))
            coeff = vec[a * d * d + b]
            choi += coeff * mat
    return choi


# Conversion to variable to matrix representation with sparsity of matrix basis


def generate_cvxpy_constraints_from_cvxpy_variable_with_sparsity(
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
    assert t in get_valid_qopeartion_type()

    if t == "state":
        constraints = [dmat_from_var_with_sparsity(c_sys, var) >> 0]
    elif t == "povm":
        matrices = povm_matrices_from_var_with_sparsity(c_sys, var)
        constraints = []
        for x in range(num_outcomes):
            constraints.append(matrices[x] >> 0)
    elif t == "gate":
        constraints = [choi_from_var_with_sparsity(c_sys, var) >> 0]
    elif t == "mprocess":
        constraints = []
        for x in range(num_outcomes):
            constraints.append(
                mprocess_element_choi_from_var_with_sparsity(
                    c_sys, num_outcomes, x, var
                )
                >> 0
            )
    return constraints


def dmat_from_var_with_sparsity(
    c_sys: CompositeSystem, var: Union[np.ndarray, CvxpyVariable]
) -> Union[np.ndarray, CvxpyVariable]:
    vec = cp.hstack([1 / np.sqrt(c_sys.dim), var])
    # density_vec = c_sys._basis_T_sparse.dot(vec)
    density_vec = c_sys._basis_T_sparse @ vec
    expr = cp.reshape(density_vec, (c_sys.dim, c_sys.dim))
    return expr


def povm_matrices_from_var_with_sparsity(
    c_sys: CompositeSystem,
    var: Union[np.ndarray, CvxpyVariable],
) -> Union[List[np.ndarray], CvxpyVariable]:
    dim = c_sys.dim
    num_outcome = var.shape[0] // (dim ** 2) + 1
    var_reshaped = cp.reshape(var, (num_outcome - 1, dim ** 2), order="C")
    vecI = np.zeros(dim ** 2)
    vecI[0] = np.sqrt(dim)
    total_vec = cp.sum(var_reshaped, axis=0)
    last_vec = vecI - total_vec
    last_vec = cp.reshape(last_vec, (1, dim ** 2))
    vecs = cp.vstack([var_reshaped, last_vec])

    matrices = []
    for vec in vecs:
        # new_vec = c_sys._basis_T_sparse.dot(vec)
        new_vec = c_sys._basis_T_sparse @ vec
        matrix = cp.reshape(new_vec, (c_sys.dim, c_sys.dim))
        matrices.append(matrix)
    return matrices


def choi_from_var_with_sparsity(
    c_sys: CompositeSystem, var: Union[np.ndarray, CvxpyVariable]
) -> Union[np.ndarray, CvxpyVariable]:
    dim = c_sys.dim
    c = np.zeros(dim ** 2)
    c[0] = 1
    hs_vec = cp.hstack([c, var])
    choi_vec = c_sys._basis_basisconjugate_T_sparse @ hs_vec
    choi = cp.reshape(choi_vec, (dim ** 2, dim ** 2))
    return choi


def mprocess_element_choi_from_var_with_sparsity(
    c_sys: CompositeSystem,
    num_outcomes: int,
    x: int,
    var: Union[np.ndarray, CvxpyVariable],
) -> Union[np.ndarray, CvxpyVariable]:
    d = c_sys.dim
    m = num_outcomes
    if 0 <= x and x < m - 1:
        vec = var[x * (d ** 4) : (x + 1) * (d ** 4)]
    elif x == m - 1:
        v = np.zeros(d ** 2)
        v[0] = 1.0
        for y in range(m - 1):
            v -= var[y * (d ** 4) : y * (d ** 4) + d ** 2]
        w = var[(m - 1) * (d ** 4) : m * (d ** 4)]
        vec = cp.hstack([v, w])

    # for debug
    # print(vec.value)

    choi_vec = c_sys._basis_basisconjugate_T_sparse @ vec
    choi = cp.reshape(choi_vec, (d ** 2, d ** 2))
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


def convert_cvxpy_variable_to_gate_hs(dim: int, var: CvxpyVariable) -> np.ndarray:
    # Probably this function does not work correctly because shape of hs is invalid. It must be (d^2, d^2).
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


def convert_cvxpy_variable_to_mprocess_hss(
    dim: int, num_outcomes: int, var: CvxpyVariable
) -> np.ndarray:
    raise NotImplementedError


def convert_cvxpy_variable_to_state(
    c_sys: CompositeSystem, var: CvxpyVariable
) -> State:
    vec = convert_cxvpy_variable_to_state_vec(c_sys.dim, var)
    state = convert_var_to_state(
        c_sys=c_sys,
        var=vec,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return state


def convert_cvxpy_variable_to_povm(c_sys: CompositeSystem, var: CvxpyVariable) -> Povm:
    vecs = convert_cxvpy_variable_to_povm_vecs(c_sys.dim, var)
    povm = Povm(
        c_sys=c_sys,
        vecs=vecs,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return povm


def convert_cvxpy_variable_to_gate(c_sys: CompositeSystem, var: CvxpyVariable) -> Gate:
    # Probably this function does not work correctly.
    hs = convert_cvxpy_variable_to_gate_hs(c_sys.dim, var)
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
    # Probably this function does not work correctly because hs must be d^2 times d^2 matrix.
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
    vec = convert_quara_variable_to_state_vec(c_sys.dim, var)
    state = convert_var_to_state(
        c_sys=c_sys,
        var=vec,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return state


def convert_quara_variable_to_povm(c_sys: CompositeSystem, var: np.ndarray) -> Povm:
    vecs = convert_quara_variable_to_povm_vecs(c_sys.dim, var)
    povm = Povm(
        c_sys=c_sys,
        vecs=vecs,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return povm


def convert_quara_variable_to_gate(c_sys: CompositeSystem, var) -> Gate:
    # Probably this function does not work correctly.
    hs = convert_quara_variable_to_gate_hs(c_sys.dim, var)
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
    elif t == "mprocess":
        pass
        # qop = convert_quara_variable_to_mprocess(c_sys, var)
    return qop
