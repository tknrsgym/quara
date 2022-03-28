import copy
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
from quara.objects.mprocess import MProcess


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
        Type of QOperation. "state", "povm", "gate", or "mprocess"
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
        Type of QOperation. "state", "povm", "gate", or "mprocess"
    dim : int
        Dimension of the system
    num_outcoms : int, optional
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
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    t : str
        Type of QOperation. "state", "povm", "gate", or "mprocess"
    var  :CvxpyVariable
        Variable of CVXPY
    num_outcoms : int, optional
        Number of outcomes for t = "povm" or  "mprocess", by default None

    Returns
    -------
    CvxpyConstraint
        Constraint of CVXPY

    Raises
    ------
    ValueError
        Unsupported type of QOperation is specified.
    """
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
    else:
        raise ValueError(f"Unsupported type of QOperation is specified. t={t}")

    return constraints


# Conversion from Variable of quara or cvxpy to Matrix Representation
def dmat_from_var(
    c_sys: CompositeSystem, var: Union[np.ndarray, CvxpyVariable]
) -> Union[np.ndarray, CvxpyVariable]:
    """Converts from Variable of quara or cvxpy to density matrix.

    Parameters
    ----------
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    var : Union[np.ndarray, CvxpyVariable]
        Variable of CVXPY

    Returns
    -------
    Union[np.ndarray, CvxpyVariable]
        Density matrix
    """
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
    """Converts from Variable of quara or cvxpy to povm element.

    Parameters
    ----------
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    num_outcomes : int
        Number of outcomes
    x : int
        Index of povm element
    var : Union[np.ndarray, CvxpyVariable]
        Variable of CVXPY

    Returns
    -------
    Union[np.ndarray, CvxpyVariable]
        povm element
    """
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
    """Converts from Variable of quara or cvxpy to Choi matrix.

    Parameters
    ----------
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    var : Union[np.ndarray, CvxpyVariable]
        Variable of CVXPY

    Returns
    -------
    Union[np.ndarray, CvxpyVariable]
        Choi matrix
    """
    d = c_sys.dim
    matrices = [np.eye(d * d, dtype=np.complex128) / d]
    for a in range(1, d * d):
        for b in range(d * d):
            mat = c_sys.basis_basisconjugate((a, b))
            coeff = var[(a - 1) * d * d + b]
            matrices.append(coeff * mat)
    choi = cp.sum(matrices)
    return choi


def mprocess_element_choi_from_var(
    c_sys: CompositeSystem,
    num_outcomes: int,
    x: int,
    var: Union[np.ndarray, CvxpyVariable],
) -> Union[np.ndarray, CvxpyVariable]:
    """Converts from Variable of quara or cvxpy to mprocess element.

    Parameters
    ----------
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    num_outcomes : int
        Number of outcomes
    x : int
        Index of mprocess element
    var : Union[np.ndarray, CvxpyVariable]
        Variable of CVXPY

    Returns
    -------
    Union[np.ndarray, CvxpyVariable]
        mprocess element
    """
    d = c_sys.dim
    m = num_outcomes
    if 0 <= x and x < m - 1:
        vec = var[x * (d ** 4) : (x + 1) * (d ** 4)]
    elif x == m - 1:
        vectors = [np.zeros(d ** 2)]
        for y in range(m - 2):
            vectors.append(var[y * (d ** 4) : y * (d ** 4) + d ** 2])
        v = cp.sum(vectors)
        w = var[(m - 1) * (d ** 4) : m * (d ** 4)]
        vec = cp.hstack([v, w])

    matrices = []
    for a in range(d * d):
        for b in range(d * d):
            mat = c_sys.basis_basisconjugate((a, b))
            coeff = vec[a * d * d + b]
            matrices.append(coeff * mat)
    choi = cp.sum(matrices)
    return choi


# Conversion to variable to matrix representation with sparsity of matrix basis


def generate_cvxpy_constraints_from_cvxpy_variable_with_sparsity(
    c_sys: CompositeSystem, t: str, var: CvxpyVariable, num_outcomes: int = None
) -> List[CvxpyConstraint]:
    """
    Returns list of Constraint of CVXPY

    Parameters
    ----------
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    t : str
        Type of QOperation. "state", "povm", "gate", or "mprocess"
    var : CvxpyVariable
        Variable of CVXPY
    num_outcomes : int, optional
        Number of outcomes for t = "povm" or  "mprocess", by default None

    Returns
    -------
    List[CvxpyConstraint]
        List of Constraint of CVXPY

    Raises
    ------
    ValueError
        Unsupported type of QOperation is specified.
    """
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
    else:
        raise ValueError(f"Unsupported type of QOperation is specified. t={t}")

    return constraints


def dmat_from_var_with_sparsity(
    c_sys: CompositeSystem, var: Union[np.ndarray, CvxpyVariable]
) -> Union[np.ndarray, CvxpyVariable]:
    """Converts from Variable of quara or cvxpy to density matrix with sparsity.

    Parameters
    ----------
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    var : Union[np.ndarray, CvxpyVariable]
        Variable of CVXPY

    Returns
    -------
    Union[np.ndarray, CvxpyVariable]
        Density matrix
    """
    vec = cp.hstack([1 / np.sqrt(c_sys.dim), var])
    density_vec = c_sys.basis_T_sparse @ vec
    expr = cp.reshape(density_vec, (c_sys.dim, c_sys.dim))
    return expr


def povm_matrices_from_var_with_sparsity(
    c_sys: CompositeSystem,
    var: Union[np.ndarray, CvxpyVariable],
) -> Union[List[np.ndarray], List[CvxpyVariable]]:
    """Converts from Variable of quara or cvxpy to povm matrices with sparsity.

    Parameters
    ----------
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    var : Union[np.ndarray, CvxpyVariable]
        Variable of CVXPY

    Returns
    -------
    Union[List[np.ndarray], List[CvxpyVariable]]
        List of povm matrices
    """
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
        new_vec = c_sys.basis_T_sparse @ vec
        matrix = cp.reshape(new_vec, (c_sys.dim, c_sys.dim))
        matrices.append(matrix)
    return matrices


def choi_from_var_with_sparsity(
    c_sys: CompositeSystem, var: Union[np.ndarray, CvxpyVariable]
) -> Union[np.ndarray, CvxpyVariable]:
    """Converts from Variable of quara or cvxpy to Choi matrix with sparsity.

    Parameters
    ----------
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    var : Union[np.ndarray, CvxpyVariable]
        Variable of CVXPY

    Returns
    -------
    Union[np.ndarray, CvxpyVariable]
        Choi matrix
    """
    dim = c_sys.dim
    c = np.zeros(dim ** 2)
    c[0] = 1
    hs_vec = cp.hstack([c, var])
    choi_vec = c_sys.basis_basisconjugate_T_sparse @ hs_vec
    choi = cp.reshape(choi_vec, (dim ** 2, dim ** 2))
    return choi


def mprocess_element_choi_from_var_with_sparsity(
    c_sys: CompositeSystem,
    num_outcomes: int,
    x: int,
    var: Union[np.ndarray, CvxpyVariable],
) -> Union[np.ndarray, CvxpyVariable]:
    """Converts from Variable of quara or cvxpy to mprocess element with sparsity.

    Parameters
    ----------
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    num_outcomes : int
        Number of outcomes
    x : int
        Index of mprocess element
    var : Union[np.ndarray, CvxpyVariable]
        Variable of CVXPY

    Returns
    -------
    Union[np.ndarray, CvxpyVariable]
        mprocess element
    """
    d = c_sys.dim
    m = num_outcomes
    if 0 <= x and x < m - 1:
        vec = var[x * (d ** 4) : (x + 1) * (d ** 4)]
    elif x == m - 1:
        v0 = np.zeros(d ** 2)
        v0[0] = 1.0
        vectors = [v0]
        for y in range(m - 1):
            vectors.append(-var[y * (d ** 4) : y * (d ** 4) + d ** 2])
        v = cp.sum(vectors)
        w = var[(m - 1) * (d ** 4) : m * (d ** 4)]
        vec = cp.hstack([v, w])

    choi_vec = c_sys.basis_basisconjugate_T_sparse @ vec
    choi = cp.reshape(choi_vec, (d ** 2, d ** 2))
    return choi


# Conversion from CVXPY.Variable to Quara.QOparation
def convert_cxvpy_variable_to_state_vec(dim: int, var: CvxpyVariable) -> np.ndarray:
    """Converts from Variable of CVXPY to vec of state.

    Parameters
    ----------
    dim : int
        Dimension of the system
    var : CvxpyVariable
        Variable of CVXPY

    Returns
    -------
    np.ndarray
        vec of state
    """
    d = dim
    l = [1 / np.sqrt(d)]
    l.extend(var.value)
    vec = np.array(l)
    return vec


def convert_cxvpy_variable_to_povm_vecs(
    dim: int, num_outcomes: int, var: CvxpyVariable
) -> np.ndarray:
    """Converts from Variable of CVXPY to vecs of povm.

    Parameters
    ----------
    dim : int
        Dimension of the system
    num_outcomes : int
        Number of outcomes
    var : CvxpyVariable
        Variable of CVXPY

    Returns
    -------
    np.ndarray
        vecs of povm
    """
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
    """Converts from Variable of CVXPY to HS matrix of gate.

    Parameters
    ----------
    dim : int
        Dimension of the system
    var : CvxpyVariable
        Variable of CVXPY

    Returns
    -------
    np.ndarray
        HS matrix of gate
    """
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
) -> List[np.ndarray]:
    """Converts from Variable of CVXPY to HS matrices of mprocess.

    Parameters
    ----------
    dim : int
        Dimension of the system
    num_outcomes : int
        Number of outcomes
    var : CvxpyVariable
        Variable of CVXPY

    Returns
    -------
    List[np.ndarray]
        HS matrices of mprocess
    """
    hs_size = dim ** 2 * dim ** 2

    vector = copy.copy(var.value)
    num_outcomes = vector.shape[0] // hs_size + 1

    one = np.zeros(dim ** 2, dtype=np.float64)
    one[0] = 1

    sum_first_row = np.zeros(dim ** 2, dtype=np.float64)
    for outcome in range(num_outcomes - 1):
        sum_first_row += vector[hs_size * outcome : hs_size * outcome + dim ** 2]
    first_row_of_last_hs = one - sum_first_row

    vector = np.insert(vector, hs_size * (num_outcomes - 1), first_row_of_last_hs)

    vec_list = []
    reshaped_vecs = vector.reshape((num_outcomes, dim ** 2, dim ** 2))
    # convert np.ndarray to list of np.ndarray
    for vec in reshaped_vecs:
        vec_list.append(vec)
    return vec_list


def convert_cvxpy_variable_to_state(
    c_sys: CompositeSystem, var: CvxpyVariable
) -> State:
    """Converts from Variable of CVXPY to state.

    Parameters
    ----------
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    var : CvxpyVariable
        Variable of CVXPY

    Returns
    -------
    State
        state
    """
    vec = convert_cxvpy_variable_to_state_vec(c_sys.dim, var)
    state = convert_var_to_state(
        c_sys=c_sys,
        var=vec,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return state


def convert_cvxpy_variable_to_povm(
    c_sys: CompositeSystem, num_outcomes: int, var: CvxpyVariable
) -> Povm:
    """Converts from Variable of CVXPY to povm.

    Parameters
    ----------
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    num_outcomes : int
        Number of outcomes
    var : CvxpyVariable
        Variable of CVXPY

    Returns
    -------
    Povm
        povm
    """
    vecs = convert_cxvpy_variable_to_povm_vecs(c_sys.dim, num_outcomes, var)
    povm = Povm(
        c_sys=c_sys,
        vecs=vecs,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return povm


def convert_cvxpy_variable_to_gate(c_sys: CompositeSystem, var: CvxpyVariable) -> Gate:
    """Converts from Variable of CVXPY to gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    var : CvxpyVariable
        Variable of CVXPY

    Returns
    -------
    Gate
        gate
    """
    hs = convert_cvxpy_variable_to_gate_hs(c_sys.dim, var)
    gate = Gate(
        c_sys=c_sys,
        hs=hs,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return gate


def convert_cvxpy_variable_to_mprocess(
    c_sys: CompositeSystem, num_outcomes: int, var: CvxpyVariable
) -> MProcess:
    """Converts from Variable of CVXPY to mprocess.

    Parameters
    ----------
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    num_outcomes : int
        Number of outcomes
    var : CvxpyVariable
        Variable of CVXPY

    Returns
    -------
    MProcess
        mprocess
    """
    hss = convert_cvxpy_variable_to_mprocess_hss(c_sys.dim, num_outcomes, var)
    mprocess = MProcess(
        c_sys=c_sys,
        hss=hss,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return mprocess


def convert_cvxpy_variable_to_qoperation(
    t: str, c_sys: CompositeSystem, var: CvxpyVariable, num_outcomes: int = None
) -> QOperation:
    """Converts from Variable of CVXPY to QOperation.

    Parameters
    ----------
    t : str
        Type of estimate, "state", "povm", "gate", or "mprocess".
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    var : CvxpyVariable
        Variable of CVXPY
    num_outcoms : int, optional
        Number of outcomes for t = "povm" or  "mprocess", by default None

    Returns
    -------
    QOperation
        QOperation

    Raises
    ------
    ValueError
        Unsupported type of QOperation is specified.
    """
    if t == "state":
        qop = convert_cvxpy_variable_to_state(c_sys, var)
    elif t == "povm":
        qop = convert_cvxpy_variable_to_povm(c_sys, num_outcomes, var)
    elif t == "gate":
        qop = convert_cvxpy_variable_to_gate(c_sys, var)
    elif t == "mprocess":
        qop = convert_cvxpy_variable_to_mprocess(c_sys, num_outcomes, var)
    else:
        raise ValueError(f"Unsupported type of estimate is specified. t={t}")
    return qop


# Conversion from Variable in Quara to Quara.QOparation
def convert_quara_variable_to_state_vec(dim: int, var: np.ndarray) -> np.ndarray:
    """Converts from Variable in quara to vec of state.

    Parameters
    ----------
    dim : int
        Dimension of the system
    var : np.ndarray
        Variable in quara

    Returns
    -------
    np.ndarray
        vec of state.
    """
    d = dim
    l = [1 / np.sqrt(d)]
    l.extend(var)
    vec = np.array(l)
    return vec


def convert_quara_variable_to_povm_vecs(
    dim: int, num_outcomes: int, var: np.ndarray
) -> List[np.ndarray]:
    """Converts from Variable in quara to vecs of povm.

    Parameters
    ----------
    dim : int
        Dimension of the system
    num_outcomes : int
        Number of outcomes
    var : np.ndarray
        Variable in quara

    Returns
    -------
    List[np.ndarray]
        vecs of povm
    """
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
    """Converts from Variable in quara to HS matrix of gate.

    Parameters
    ----------
    dim : int
        Dimension of the system
    var : np.ndarray
        Variable in quara

    Returns
    -------
    np.ndarray
        HS matrix of gate
    """
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


def convert_quara_variable_to_mprocess_hss(
    dim: int, num_outcomes: int, var: np.ndarray
) -> List[np.ndarray]:
    """Converts from Variable in quara to HS matrices of mprocess.

    Parameters
    ----------
    dim : int
        Dimension of the system
    num_outcomes : int
        Number of outcomes
    var : np.ndarray
        Variable in quara

    Returns
    -------
    List[np.ndarray]
        HS matrices of mprocess
    """
    hs_size = dim ** 2 * dim ** 2

    vector = copy.copy(var)

    one = np.zeros(dim ** 2, dtype=np.float64)
    one[0] = 1

    sum_first_row = np.zeros(dim ** 2, dtype=np.float64)
    for outcome in range(num_outcomes - 1):
        sum_first_row += vector[hs_size * outcome : hs_size * outcome + dim ** 2]
    first_row_of_last_hs = one - sum_first_row

    vector = np.insert(vector, hs_size * (num_outcomes - 1), first_row_of_last_hs)

    vec_list = []
    reshaped_vecs = vector.reshape((num_outcomes, dim ** 2, dim ** 2))
    # convert np.ndarray to list of np.ndarray
    for vec in reshaped_vecs:
        vec_list.append(vec)
    return vec_list


def convert_quara_variable_to_state(c_sys: CompositeSystem, var: np.ndarray) -> State:
    """Converts from Variable in quara to state.

    Parameters
    ----------
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    var : np.ndarray
        Variable in quara

    Returns
    -------
    State
        state
    """
    vec = convert_quara_variable_to_state_vec(c_sys.dim, var)
    state = convert_var_to_state(
        c_sys=c_sys,
        var=vec,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return state


def convert_quara_variable_to_povm(
    c_sys: CompositeSystem, num_outcomes: int, var: np.ndarray
) -> Povm:
    """Converts from Variable in quara to povm.

    Parameters
    ----------
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    num_outcomes : int
        Number of outcomes
    var : np.ndarray
        Variable in quara

    Returns
    -------
    Povm
        povm
    """
    vecs = convert_quara_variable_to_povm_vecs(c_sys.dim, num_outcomes, var)
    povm = Povm(
        c_sys=c_sys,
        vecs=vecs,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return povm


def convert_quara_variable_to_gate(c_sys: CompositeSystem, var) -> Gate:
    """Converts from Variable in quara to gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    var : [type]
        Variable in quara

    Returns
    -------
    Gate
        gate
    """
    hs = convert_quara_variable_to_gate_hs(c_sys.dim, var)
    gate = Gate(
        c_sys=c_sys,
        hs=hs,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return gate


def convert_quara_variable_to_mprocess(
    c_sys: CompositeSystem, num_outcomes: int, var: np.ndarray
) -> MProcess:
    """Converts from Variable in quara to mprocess.

    Parameters
    ----------
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    num_outcomes : int
        Number of outcomes
    var : np.ndarray
        Variable in quara

    Returns
    -------
    MProcess
        mprocess
    """
    hss = convert_quara_variable_to_mprocess_hss(c_sys.dim, num_outcomes, var)
    mprocess = MProcess(
        c_sys=c_sys,
        hss=hss,
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    return mprocess


def convert_quara_variable_to_qoperation(
    t: str, c_sys: CompositeSystem, var: np.ndarray, num_outcomes: int = None
) -> QOperation:
    """Converts from Variable in quara to QOperation.

    Parameters
    ----------
    t : str
        Type of QOperation. "state", "povm", "gate", or "mprocess"
    c_sys : CompositeSystem
        Composite system that the QOperation acts on
    var : np.ndarray
        Variable in quara
    num_outcomes : int, optional
        Number of outcomes for t = "povm" or  "mprocess", by default None

    Returns
    -------
    QOperation
        QOperation

    Raises
    ------
    ValueError
        Unsupported type of QOperation is specified.
    """
    if t == "state":
        qop = convert_quara_variable_to_state(c_sys, var)
    elif t == "povm":
        qop = convert_quara_variable_to_povm(c_sys, num_outcomes, var)
    elif t == "gate":
        qop = convert_quara_variable_to_gate(c_sys, var)
    elif t == "mprocess":
        qop = convert_quara_variable_to_mprocess(c_sys, num_outcomes, var)
    else:
        raise ValueError(f"Unsupported type of estimate is specified. t={t}")

    return qop
