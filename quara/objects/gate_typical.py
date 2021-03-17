import numpy as np
from typing import List, Dict, Tuple
from itertools import product

from scipy.linalg import expm

from quara.objects.matrix_basis import MatrixBasis
from quara.objects.matrix_basis import (
    get_comp_basis,
    get_pauli_basis,
    get_normalized_pauli_basis,
    get_normalized_gell_mann_basis,
)
from quara.objects.composite_system import CompositeSystem
from quara.objects.gate import Gate
from quara.objects.gate import (
    convert_hs,
)
from quara.objects.effective_lindbladian import (
    _truncate_hs,
)


def get_gate_names() -> List[str]:
    """Return the list of valid gate names."""
    names = []
    names.extend(["identity"])
    names.extend(get_gate_names_1qubit())
    names.extend(get_gate_names_2qubit())
    # names.extend(get_gate_names_3qubit())
    names.extend(get_gate_names_1qutrit())
    # names.extend(get_gate_names_2qutrit())
    return names


def get_gate_names_1qubit() -> List[str]:
    """Return the list of valid gate names of 1-qubit gates."""
    names = []
    names.append("x90")
    names.append("x180")
    names.append("x")
    names.append("y90")
    names.append("y180")
    names.append("y")
    names.append("z90")
    names.append("z180")
    names.append("z")
    names.append("phase")
    names.append("phase_daggered")
    names.append("piover8")
    names.append("piover8_daggered")
    names.append("hadamard")

    return names


def get_gate_names_2qubit() -> List[str]:
    """Return the list of valid gate names of 2-qubit gates."""
    names = []
    names.append("cx")
    names.append("cz")
    names.append("swap")
    names.append("zx90")
    names.append("zz90")

    return names


def get_gate_names_2qubit_asymmetric() -> List[str]:
    """Return the list of valid gate names of 2-qubit gates that are asymmetric with respect to ids of elemental systems."""
    names = []
    names.append("cx")
    names.append("zx90")

    return names


def get_gate_names_1qutrit() -> List[str]:
    """return the list of valid (implemented) gate names of 1-qutrit gates."""
    names = []
    names.extend(get_gate_names_1qutrit_single_gellmann())

    return names


def get_gate_names_1qutrit_single_gellmann() -> List[str]:
    """return the list of valid (implemented) gate names of 1-qutrit single Gell-Mann gates."""
    names = []
    # angle = 90
    names.append("01x90")
    names.append("01y90")
    names.append("01z90")
    names.append("12x90")
    names.append("12y90")
    names.append("12z90")
    names.append("02x90")
    names.append("02y90")
    names.append("02z90")
    # angle = 180
    names.append("01x180")
    names.append("01y180")
    names.append("01z180")
    names.append("12x180")
    names.append("12y180")
    names.append("12z180")
    names.append("02x180")
    names.append("02y180")
    names.append("02z180")

    return names


def _is_valid_dims_ids(dims: List[int], ids: List[int]) -> bool:
    res = True

    # whether components of dims are integers larger than 1
    for d in dims:
        if d <= 1:
            res = False
            raise ValueError(f"Component of dims must be larger than 1.")

    # whether components of id_sys_list are non-negative integers
    for i in ids:
        if i < 0:
            res = False
            raise ValueError(f"Component of ids must be non-negative.")

    return res


def _dim_total_from_dims(dims: List[int]) -> int:
    dim_total = 1
    for d in dims:
        dim_total = dim_total * d
    return dim_total


def generate_unitary_mat_from_gate_name(
    gate_name: str, dims: List[int] = [], ids: List[int] = []
):
    """returns the unitary matrix of a gate.

    Parameters
    ----------
    gate_name : str
        name of gate

    dims : List[int]
        list of dimentions of elemental systems that the gate acts on.

    ids : List[int] (optional)
        list of ids for elemental systems

    Returns
    ----------
    np.array
        The unitary matrix of the gate, to be complex.
    """
    _is_valid_dims_ids(dims, ids)
    assert gate_name in get_gate_names()

    if gate_name == "identity":
        dim_total = _dim_total_from_dims(dims)
        if dim_total <= 1:
            raise ValueError(f"dim_total must be larger than 1.")
        method_name = "generate_gate_" + gate_name + "_unitary_mat"
        method = eval(method_name)
        u = method(dim_total)
    # 1-qubit gate
    elif gate_name in get_gate_names_1qubit():
        method_name = "generate_gate_" + gate_name + "_unitary_mat"
        method = eval(method_name)
        u = method()
    # 2-qubit gate
    elif gate_name in get_gate_names_2qubit():
        method_name = "generate_gate_" + gate_name + "_unitary_mat"
        method = eval(method_name)
        if gate_name in get_gate_names_2qubit_asymmetric():
            u = method(ids)
        else:
            u = method()
    # 3-qubit gate
    # 1-qutrit
    elif gate_name in get_gate_names_1qutrit():
        if gate_name in get_gate_names_1qutrit_single_gellmann():
            method_name = "generate_gate_1qutrit_single_gellmann_unitary_mat"
            method = eval(method_name)
            u = method(gate_name)
    # 2-qutrit
    else:
        raise ValueError(f"gate_name is out of range.")

    return u


def generate_gate_mat_from_gate_name(
    gate_name: str, dims: List[int] = [], ids: List[int] = []
) -> np.array:
    """returns the Hilbert-Schmidt representation matrix of a gate.

    Parameters
    ----------
    gate_name : str
        name of gate

    dims : List[int]
        list of dimentions of elemental systems that the gate acts on.

    ids : List[int] (optional)
        list of ids for elemental systems

    Returns
    ----------
    np.array
        The HS matrix of the gate, to be real.
    """

    _is_valid_dims_ids(dims, ids)
    assert gate_name in get_gate_names()

    if gate_name == "identity":
        dim_total = _dim_total_from_dims(dims)
        if dim_total <= 1:
            raise ValueError(f"dim_total must be larger than 1.")
        method_name = "generate_gate_" + gate_name + "_mat"
        method = eval(method_name)
        mat = method(dim_total)
    # 1-qubit gate
    elif gate_name in get_gate_names_1qubit():
        method_name = "generate_gate_" + gate_name + "_mat"
        method = eval(method_name)
        mat = method()
    # 2-qubit gate
    elif gate_name in get_gate_names_2qubit():
        method_name = "generate_gate_" + gate_name + "_mat"
        method = eval(method_name)
        if gate_name in get_gate_names_2qubit_asymmetric():
            mat = method(ids)
        else:
            mat = method()
    # 3-qubit gate
    # 1-qutrit
    elif gate_name in get_gate_names_1qutrit():
        if gate_name in get_gate_names_1qutrit_single_gellmann():
            method_name = "generate_gate_1qutrit_single_gellmann_mat"
            method = eval(method_name)
            mat = method(gate_name)
    # 2-qutrit
    else:
        raise ValueError(f"gate_name is out of range.")

    return mat


def generate_gate_from_gate_name(
    gate_name: str, c_sys: CompositeSystem, ids: List[int] = []
) -> "Gate":
    """returns gate class.

    Parameters
    ----------
    gate_name : str
        name of gate

    c_sys : CompositeSystem

    ids : List[int] (optional)
        list of ids for elemental systems

    Returns
    ----------
    Gate
        The gate class for the input
    """
    assert gate_name in get_gate_names()

    if gate_name == "identity":
        method_name = "generate_gate_" + gate_name
        method = eval(method_name)
        gate = method(c_sys)
    # 1-qubit gate
    elif gate_name in get_gate_names_1qubit():
        method_name = "generate_gate_" + gate_name
        method = eval(method_name)
        gate = method(c_sys)
    # 2-qubit gate
    elif gate_name in get_gate_names_2qubit():
        method_name = "generate_gate_" + gate_name
        method = eval(method_name)
        if gate_name in get_gate_names_2qubit_asymmetric():
            gate = method(c_sys, ids)
        else:
            gate = method(c_sys)
    # 3-qubit gate
    # 1-qutrit gate
    elif gate_name in get_gate_names_1qutrit():
        if gate_name in get_gate_names_1qutrit_single_gellmann():
            method_name = "generate_gate_1qutrit_single_gellmann"
            method = eval(method_name)
            gate = method(c_sys, gate_name)
    # 2-qutrit gate
    else:
        raise ValueError(f"gate_name is out of range.")

    return gate


def calc_gate_mat_from_unitary_mat(from_u: np.array, to_basis: MatrixBasis) -> np.array:
    """Return the HS matrix for a gate represented by an unitary matrix.

    Parameters
    ----------
    from_u : np.array((dim, dim), dtype=np.complex128)
        The unitary matrix, to be square complex np.array.

    to_basis : MatrixBasis
        The matrix basis for representing the HS matrix, to be orthonormal.

    Returns
    ----------
    np.array((dim^2, dim^2), dtype=np.complex128)
        The HS matrix of the gate corresponding to the unitary matrix.
    """
    shape = from_u.shape
    assert shape[0] == shape[1]
    dim = shape[0]

    assert to_basis.dim == dim
    assert to_basis.is_orthogonal() == True
    assert to_basis.is_normal() == True

    hs_comp = np.kron(from_u, np.conjugate(from_u))
    basis_comp = get_comp_basis(dim)
    hs = convert_hs(from_hs=hs_comp, from_basis=basis_comp, to_basis=to_basis)

    return hs


def calc_gate_mat_from_unitary_mat_with_hermitian_basis(
    from_u: np.array, to_basis: MatrixBasis
) -> np.array:
    """Return the HS matrix w.r.t. a Hermitian (orthonormal) matrix basis for a gate represented by an unitary matrix.

    Parameters
    ----------
    from_u : np.array((dim, dim), dtype=np.complex128)
        The unitary matrix, to be square complex np.array.

    to_basis : MatrixBasis
        The matrix basis for representing the HS matrix

    Returns
    ----------
    np.array((dim^2, dim^2), dtype=np.float64)
        The HS matrix of the gate corresponding to the unitary matrix, to be real.
    """
    assert to_basis.is_hermitian() == True
    hs_complex = calc_gate_mat_from_unitary_mat(from_u, to_basis)
    hs = _truncate_hs(hs_complex)
    return hs


# Identity gate


def generate_gate_identity_unitary_mat(dim: int) -> np.array:
    """Return the unitary matrix for an identity gate.

    The result is the dim times dim complex identity matrix.

    Parameters
    ----------
    dim : int
        The dimension of the quantum system on which the gate acts.

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    u = np.eye(dim, dtype=np.complex128)
    return u


def generate_gate_identity_mat(dim: int) -> np.array:
    """Return the Hilbert-Schmidt representation matrix for an Identity gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is the dim^2 times dim^2 real identity matrix.

    Parameters
    ----------
    dim : int
        The dimension of the quantum system on which the gate acts.

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate. It is the identity matrix in this case.
    """
    size = dim * dim
    mat = np.eye(size, size, dtype=np.float64)
    return mat


def generate_gate_identity(c_sys: CompositeSystem) -> "Gate":
    """Return the Gate class for the identity gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the identity gate on the composite system.
    """
    dim = c_sys.dim
    hs = generate_gate_identity_mat(dim)
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# X90 gate on 1-qubit


def generate_gate_x90_unitary_mat() -> np.array:
    """Return the unitary matrix for an X90 gate.

    The result is the 2 times 2 complex matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    u = np.array([[1 + 0j, 0 - 1j], [0 - 1j, 1 + 0j]], dtype=np.complex128) / np.sqrt(2)
    return u


def generate_gate_x90_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for an X90 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    l = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]
    mat = np.array(l, dtype=np.float64)
    return mat


def generate_gate_x90(c_sys: CompositeSystem) -> "Gate":
    """Return the Gate class for the X90 gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the X90 gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_x90_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# X180 gate on 1-qubit


def generate_gate_x180_unitary_mat() -> np.array:
    """Return the unitary matrix for an X180 gate.

    The result is the 2 times 2 complex matrix, -i X.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    u = np.array([[0, -1j], [-1j, 0]], dtype=np.complex128)
    return u


def generate_gate_x180_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for an X180 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    l = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
    mat = np.array(l, dtype=np.float64)
    return mat


def generate_gate_x180(c_sys: CompositeSystem) -> "Gate":
    """Return the Gate class for the X180 gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the identity gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_x180_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# X gate on 1-qubit


def generate_gate_x_unitary_mat() -> np.array:
    """Return the unitary matrix for an X gate.

    The result is the 2 times 2 complex matrix, X.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    num_qubit = 1
    b = get_pauli_basis(num_qubit)
    u = b[1]
    return u


def generate_gate_x_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for an X gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    l = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
    mat = np.array(l, dtype=np.float64)
    return mat


def generate_gate_x(c_sys: CompositeSystem) -> "Gate":
    """Return the Gate class for the X gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the X gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_x_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# Y90 on 1-qubit


def generate_gate_y90_unitary_mat() -> np.array:
    """Return the unitary matrix for an Y90 gate.

    The result is a 2 times 2 complex matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    u = np.array([[1, -1], [1, 1]], dtype=np.complex128) / np.sqrt(2)
    return u


def generate_gate_y90_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for a Y90 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    l = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0]]
    mat = np.array(l, dtype=np.float64)
    return mat


def generate_gate_y90(c_sys: CompositeSystem) -> "Gate":
    """Return the Gate class for the Y90 gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the Y90 gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_y90_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# Y180 gate on 1-qubit


def generate_gate_y180_unitary_mat() -> np.array:
    """Return the unitary matrix for a Y180 gate.

    The result is the 2 times 2 complex matrix, -i Y.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    u = np.array([[0, -1], [1, 0]], dtype=np.complex128)
    return u


def generate_gate_y180_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for a Y180 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    l = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
    mat = np.array(l, dtype=np.float64)
    return mat


def generate_gate_y180(c_sys: CompositeSystem) -> "Gate":
    """Return the Gate class for the Y180 gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the Y180 gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_y180_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# Y gate on 1-qubit


def generate_gate_y_unitary_mat() -> np.array:
    """Return the unitary matrix for a Y gate.

    The result is the 2 times 2 complex matrix, Y.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    num_qubit = 1
    b = get_pauli_basis(num_qubit)
    u = b[2]
    return u


def generate_gate_y_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for a Y gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    l = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
    mat = np.array(l, dtype=np.float64)
    return mat


def generate_gate_y(c_sys: CompositeSystem) -> "Gate":
    """Return the Gate class for the Y gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the Y gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_y_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# Z90 gate on 1-qubit


def generate_gate_z90_unitary_mat() -> np.array:
    """Return the unitary matrix for a Z90 gate.

    The result is a 2 times 2 complex matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    u = np.array([[1 - 1j, 0], [0, 1 + 1j]], dtype=np.complex128) / np.sqrt(2)
    return u


def generate_gate_z90_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for a Z90 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    l = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    mat = np.array(l, dtype=np.float64)
    return mat


def generate_gate_z90(c_sys: CompositeSystem) -> "Gate":
    """Return the Gate class for the Z90 gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the Z90 gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_z90_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# Z180 gate on 1-qubit


def generate_gate_z180_unitary_mat() -> np.array:
    """Return the unitary matrix for a Z180 gate.

    The result is the 2 times 2 complex matrix, -i Z.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    u = np.array([[-1j, 0], [0, 1j]], dtype=np.complex128)
    return u


def generate_gate_z180_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for a Z180 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    l = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    mat = np.array(l, dtype=np.float64)
    return mat


def generate_gate_z180(c_sys: CompositeSystem) -> "Gate":
    """Return the Gate class for the Z180 gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the Z180 gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_z180_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# Z gate on 1-qubit


def generate_gate_z_unitary_mat() -> np.array:
    """Return the unitary matrix for a Z gate.

    The result is the 2 times 2 complex matrix, Z.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    num_qubit = 1
    b = get_pauli_basis(num_qubit)
    u = b[3]
    return u


def generate_gate_z_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for a Z gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    l = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    mat = np.array(l, dtype=np.float64)
    return mat


def generate_gate_z(c_sys: CompositeSystem) -> "Gate":
    """Return the Gate class for the Z gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the Z gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_z_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# Phase (S) gate on 1-qubit


def generate_gate_phase_unitary_mat() -> np.array:
    """Return the unitary matrix for a Phase (S) gate.

    The result is the 2 times 2 complex matrix, S.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    u = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    return u


def generate_gate_phase_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for a Phase (S) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    l = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    mat = np.array(l, dtype=np.float64)
    return mat


def generate_gate_phase(c_sys: CompositeSystem) -> "Gate":
    """Return the Gate class for the Phase (S) gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the Phase (S) gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_phase_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# Phase daggered (S^dagger) gate on 1-qubit


def generate_gate_phase_daggered_unitary_mat() -> np.array:
    """Return the unitary matrix for a Phase daggerd (S^dagger) gate.

    The result is the 2 times 2 complex matrix, S^dagger.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    u = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    return u


def generate_gate_phase_daggered_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for a Phase daggerd (S^dagger) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    l = [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    mat = np.array(l, dtype=np.float64)
    return mat


def generate_gate_phase_daggered(c_sys: CompositeSystem) -> "Gate":
    """Return the Gate class for the Phase daggered (S^dagger) gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the Phase daggered (S^dagger) gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_phase_daggered_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# pi/8 (T) gate on 1-qubit


def generate_gate_piover8_unitary_mat() -> np.array:
    """Return the unitary matrix for a pi/8 (T) gate.

    The result is the 2 times 2 complex matrix, T.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    u = np.array(
        [[1, 0], [0, 0.50 * np.sqrt(2) + 0.50 * np.sqrt(2) * 1j]], dtype=np.complex128
    )
    return u


def generate_gate_piover8_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for a pi/8 (T) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    l = [
        [1, 0, 0, 0],
        [0, 0.50 * np.sqrt(2), -0.50 * np.sqrt(2), 0],
        [0, 0.50 * np.sqrt(2), 0.50 * np.sqrt(2), 0],
        [0, 0, 0, 1],
    ]
    mat = np.array(l, dtype=np.float64)
    return mat


def generate_gate_piover8(c_sys: CompositeSystem) -> "Gate":
    """Return the Gate class for the pi/8 (T) gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the pi/8 (T) gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_piover8_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# pi/8 daggered (T^dagger) gate on 1-qubit


def generate_gate_piover8_daggered_unitary_mat() -> np.array:
    """Return the unitary matrix for a pi/8 daggerd (T^dagger) gate.

    The result is the 2 times 2 complex matrix, T^dagger.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    u = np.array(
        [[1, 0], [0, 0.50 * np.sqrt(2) - 1j * 0.50 * np.sqrt(2)]], dtype=np.complex128
    )
    return u


def generate_gate_piover8_daggered_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for a pi/8 daggerd (T^dagger) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    l = [
        [1, 0, 0, 0],
        [0, 0.50 * np.sqrt(2), 0.50 * np.sqrt(2), 0],
        [0, -0.50 * np.sqrt(2), 0.50 * np.sqrt(2), 0],
        [0, 0, 0, 1],
    ]
    mat = np.array(l, dtype=np.float64)
    return mat


def generate_gate_piover8_daggered(c_sys: CompositeSystem) -> "Gate":
    """Return the Gate class for the pi/8 daggered (T^dagger) gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the pi/8 daggered (T^dagger) gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_piover8_daggered_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# Hadamard (H) gate on 1-qubit


def generate_gate_hadamard_unitary_mat() -> np.array:
    """Return the unitary matrix for an Hadamard (H) gate.

    The result is the 2 times 2 complex matrix, H.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    u = 0.50 * np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    return u


def generate_gate_hadamard_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for an Hadamard (H) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    l = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0]]
    mat = np.array(l, dtype=np.float64)
    return mat


def generate_gate_hadamard(c_sys: CompositeSystem) -> "Gate":
    """Return the Gate class for the Hadamard (H) gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the Hadamard (H) gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_hadamard_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# 2-qubit gates

# Control-X gate on 2-qubit


def generate_gate_cx_unitary_mat(ids: List[int]) -> np.array:
    """Return the unitary matrix for a Control-X (CX) gate.

    The result is the 4 times 4 complex matrix.

    Parameters
    ----------
    ids : List[int]
        ids[0] for control system id, and ids[1] for target system id.

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    assert len(ids) == 2
    assert ids[0] != ids[1]
    if ids[0] < ids[1]:
        mat = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=np.complex128,
        )
    else:
        mat = np.array(
            [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
            dtype=np.complex128,
        )

    return mat


def generate_gate_cx_mat(ids: List[int]) -> np.array:
    """Return the Hilbert-Schmidt representation matrix for a Control-X (CX) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 16 times 16 real matrix.

    Parameters
    ----------
    ids : List[int]
        ids[0] for the control system id, ids[1] for the target system id.

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    assert len(ids) == 2
    assert ids[0] != ids[1]

    u = generate_gate_cx_unitary_mat(ids)
    b = get_normalized_pauli_basis(n_qubit=2)
    mat = calc_gate_mat_from_unitary_mat_with_hermitian_basis(from_u=u, to_basis=b)
    return mat


def generate_gate_cx(c_sys: "CompositeSystem", ids: List[int]) -> np.array:
    """Return the Gate class for the Control-X (CX) gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    ids: List[int]
        ids[0] for control system index
        ids[1] for target system index

    Returns
    ----------
    Gate
        The Gate class for the Control-X (CX) gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 2
    hs = generate_gate_cx_mat(ids)
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# Control-Z gate on 2-qubit


def generate_gate_cz_unitary_mat() -> np.array:
    """Return the unitary matrix for a Control-Z (CZ) gate.

    The result is the 4 times 4 complex matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    mat = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
        dtype=np.complex128,
    )

    return mat


def generate_gate_cz_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for a Control-Z (CZ) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 16 times 16 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    u = generate_gate_cz_unitary_mat()
    b = get_normalized_pauli_basis(n_qubit=2)
    mat = calc_gate_mat_from_unitary_mat_with_hermitian_basis(from_u=u, to_basis=b)
    return mat


def generate_gate_cz(c_sys: "CompositeSystem") -> np.array:
    """Return the Gate class for the Control-Z (CZ) gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the Control-Z (CZ) gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 2
    hs = generate_gate_cz_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# SWAP gate on 2-qubit


def generate_gate_swap_unitary_mat() -> np.array:
    """Return the unitary matrix for a SWAP gate.

    The result is the 4 times 4 complex matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    mat = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.complex128,
    )

    return mat


def generate_gate_swap_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for a SWAP gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 16 times 16 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    u = generate_gate_swap_unitary_mat()
    b = get_normalized_pauli_basis(n_qubit=2)
    mat = calc_gate_mat_from_unitary_mat_with_hermitian_basis(from_u=u, to_basis=b)
    return mat


def generate_gate_swap(c_sys: "CompositeSystem") -> np.array:
    """Return the Gate class for the SWAP gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the SWAP gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 2
    hs = generate_gate_swap_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# ZX90 gate on 2-qubit system


def generate_gate_zx90_unitary_mat(ids: List[int]) -> np.array:
    """Return the unitary matrix for a ZX90 gate.

    The result is the 4 times 4 complex matrix.

    Parameters
    ----------
    ids : List[int]
        ids[0] for control system id, and ids[1] for target system id.

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    assert len(ids) == 2
    assert ids[0] != ids[1]
    if ids[0] < ids[1]:
        mat = (
            0.50
            * np.sqrt(2)
            * np.array(
                [[1, -1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 1, 1j], [0, 0, 1j, 1]],
                dtype=np.complex128,
            )
        )
    else:
        mat = (
            0.50
            * np.sqrt(2)
            * np.array(
                [[1, 0, -1j, 0], [0, 1, 0, 1j], [-1j, 0, 1, 0], [0, 1j, 0, 1]],
                dtype=np.complex128,
            )
        )

    return mat


def generate_gate_zx90_mat(ids: List[int]) -> np.array:
    """Return the Hilbert-Schmidt representation matrix for a ZX90 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 16 times 16 real matrix.

    Parameters
    ----------
    ids : List[int]
        ids[0] for the control system id, ids[1] for the target system id.

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    assert len(ids) == 2
    assert ids[0] != ids[1]

    u = generate_gate_zx90_unitary_mat(ids)
    b = get_normalized_pauli_basis(n_qubit=2)
    mat = calc_gate_mat_from_unitary_mat_with_hermitian_basis(from_u=u, to_basis=b)
    return mat


def generate_gate_zx90(c_sys: "CompositeSystem", ids: List[int]) -> np.array:
    """Return the Gate class for the ZX90 gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    ids: List[int]
        ids[0] for control system index
        ids[1] for target system index

    Returns
    ----------
    Gate
        The Gate class for the ZX90 gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 2
    hs = generate_gate_zx90_mat(ids)
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# ZZ90 gate on 2-qubit system


def generate_gate_zz90_unitary_mat() -> np.array:
    """Return the unitary matrix for a ZZ90 gate.

    The result is the 4 times 4 complex matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    mat = (
        0.50
        * np.sqrt(2)
        * np.array(
            [
                [1 - 1j, 0, 0, 0],
                [0, 1 + 1j, 0, 0],
                [0, 0, 1 + 1j, 0],
                [0, 0, 0, 1 - 1j],
            ],
            dtype=np.complex128,
        )
    )

    return mat


def generate_gate_zz90_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for a ZZ90 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 16 times 16 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the gate.
    """
    u = generate_gate_zz90_unitary_mat()
    b = get_normalized_pauli_basis(n_qubit=2)
    mat = calc_gate_mat_from_unitary_mat_with_hermitian_basis(from_u=u, to_basis=b)
    return mat


def generate_gate_zz90(c_sys: "CompositeSystem") -> np.array:
    """Return the Gate class for the ZZ90 gate on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    ----------
    Gate
        The Gate class for the ZZ90 gate on the composite system.
    """
    assert len(c_sys.elemental_systems) == 2
    hs = generate_gate_zz90_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


# 3-qubit gates


# 1-qutrit gates

# Base of Hamiltonian


def calc_base_matrix_1qutrit(levels: str, axis: str) -> np.array:
    """Return a base matrix for 1-qutrit Hamiltonian.

    Parameters
    ----------
    axis : str
        specifies "x", "y", or "z".

    levels : str
        specifies levels for the axis, limited to ["01", "12", or "02"].

    Returns
    ----------
    np.array((3,3), dtype=np.complex128)
        The base matrix corresponding to the axis and levels, to be complex.
    """
    assert axis in ["x", "y", "z"]
    assert len(levels) == 2
    assert levels in ["01", "12", "02"]

    method_str = "calc_base_matrix_1qutrit_" + axis + "_" + levels
    mat = eval(method_str)()
    return mat


def calc_base_matrix_1qutrit_x_01() -> np.array:
    """Return the base matrix corresponding to the x-axis w.r.t. levels 0 and 1."""
    l = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    mat = np.array(l, dtype=np.complex128)
    return mat


def calc_base_matrix_1qutrit_y_01() -> np.array:
    """Return the base matrix corresponding to the y-axis w.r.t. levels 0 and 1."""
    l = [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]]
    mat = np.array(l, dtype=np.complex128)
    return mat


def calc_base_matrix_1qutrit_z_01() -> np.array:
    """Return the base matrix corresponding to the z-axis w.r.t. levels 0 and 1."""
    l = [[1, 0, 0], [0, -1, 0], [0, 0, 0]]
    mat = np.array(l, dtype=np.complex128)
    return mat


def calc_base_matrix_1qutrit_x_12() -> np.array:
    """Return the base matrix corresponding to the x-axis w.r.t. levels 1 and 2."""
    l = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
    mat = np.array(l, dtype=np.complex128)
    return mat


def calc_base_matrix_1qutrit_y_12() -> np.array:
    """Return the base matrix corresponding to the y-axis w.r.t. levels 1 and 2."""
    l = [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]
    mat = np.array(l, dtype=np.complex128)
    return mat


def calc_base_matrix_1qutrit_z_12() -> np.array:
    """Return the base matrix corresponding to the z-axis w.r.t. levels 1 and 2."""
    l = [[0, 0, 0], [0, 1, 0], [0, 0, -1]]
    mat = np.array(l, dtype=np.complex128)
    return mat


def calc_base_matrix_1qutrit_x_02() -> np.array:
    """Return the base matrix corresponding to the x-axis w.r.t. levels 0 and 2."""
    l = [[0, 0, 1], [0, 0, 0], [1, 0, 0]]
    mat = np.array(l, dtype=np.complex128)
    return mat


def calc_base_matrix_1qutrit_y_02() -> np.array:
    """Return the base matrix corresponding to the y-axis w.r.t. levels 0 and 2."""
    l = [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]]
    mat = np.array(l, dtype=np.complex128)
    return mat


def calc_base_matrix_1qutrit_z_02() -> np.array:
    """Return the base matrix corresponding to the z-axis w.r.t. levels 0 and 2."""
    l = [[1, 0, 0], [0, 0, 0], [0, 0, -1]]
    mat = np.array(l, dtype=np.complex128)
    return mat


def get_base_matrices_1qutrit() -> Dict[Tuple[str, str], np.array]:
    """Return the dictionary object containing all base matrices for 1-qutrit Hamiltonian.

    Parameters
    ----------

    Returns
    ----------
    Dict[Tuple[str, str], np.array]
        The dictionary. The first string of the Tuple is for the levels, "01", "12", or "02". The second string of the Tuple is for the axis, "x", "y", or "z".
        For example, dict[("12", "x")] is the base matrix for the x-axis w.r.t. the levels 1 and 2.
    """
    levels_list = ["01", "12", "02"]
    axis_list = ["x", "y", "z"]

    l = []
    for p in product(levels_list, axis_list):
        levels = p[0]
        axis = p[1]
        mat = calc_base_matrix_1qutrit(axis, levels)
        l.append(((levels, axis), mat))

    d = dict(l)
    return d


# gate_name -> {"levels", "axis", "angle"}
def calc_levels_axis_angle_from_gate_name_1qutrit_single_gellmann(
    gate_name: str,
) -> Dict[str, str]:
    """return dictionary object containing three information for specifying a 1-qutrit Gell-Mann gate.

    Parameters
    ----------
    gate_name : str
        A name of gate, e.g., "12x90".

    Returns
    ----------
    Dict[str, str]
        Example: {"levels": "12", "axis": "x", "angle": "90"}
    """
    levels = gate_name[0:2]
    axis = gate_name[2]
    angle = gate_name[3:]
    res = {"levels": levels, "axis": axis, "angle": angle}
    return res


# "angle" -> float
def calc_angle_from_str_to_float(angle_str: str) -> float:
    """return angle value from angle string"""
    if angle_str == "90":
        angle = 0.50 * np.pi
    elif angle_str == "m90":
        angle = -0.50 * np.pi
    elif angle_str == "180":
        angle = np.pi
    elif angle_str == "m180":
        angle = -np.pi
    else:
        raise ValueError(f"angle_str is invalid!")

    return angle


def calc_coeff_from_angle_str(angle_str: str) -> float:
    "return coeff = 0.5 * angle from angle_str."
    angle = calc_angle_from_str_to_float(angle_str)
    return 0.50 * angle


def calc_1qutrit_single_gellmann_hamiltonian_mat_from_levels_axis_angle(
    levels: str, axis: str, angle: str
) -> np.array:
    """return a 1-qutrit Hamiltonian matrix for the axis, levels, and angle."""
    assert axis in ["x", "y", "z"]
    assert levels in ["01", "12", "02"]

    mat = calc_base_matrix_1qutrit(levels, axis)
    coeff = calc_coeff_from_angle_str(angle)
    h = coeff * mat
    return h


def generate_gate_1qutrit_single_gellmann_hamiltonian_mat(gate_name: str) -> np.array:
    """return a 1-qutrit Hamiltonian matrix for the gate name."""
    res = calc_levels_axis_angle_from_gate_name_1qutrit_single_gellmann(gate_name)
    levels = res["levels"]
    axis = res["axis"]
    angle = res["angle"]
    h = calc_1qutrit_single_gellmann_hamiltonian_mat_from_levels_axis_angle(
        levels=levels, axis=axis, angle=angle
    )
    return h


def generate_gate_1qutrit_single_gellmann_unitary_mat(gate_name: str) -> np.array:
    """return the unitary matrix for the gate."""
    h = generate_gate_1qutrit_single_gellmann_hamiltonian_mat(gate_name)
    u = expm(-1j * h)
    return u


def generate_gate_1qutrit_single_gellmann_mat(gate_name: str) -> np.array:
    """return the HS matrix for the gate."""
    u = generate_gate_1qutrit_single_gellmann_unitary_mat(gate_name)
    to_basis = get_normalized_gell_mann_basis()
    hs = calc_gate_mat_from_unitary_mat_with_hermitian_basis(
        from_u=u, to_basis=to_basis
    )
    return hs


def generate_gate_1qutrit_single_gellmann(
    c_sys: CompositeSystem, gate_name: str
) -> np.array:
    """return the Gate for the gate."""
    assert len(c_sys.elemental_systems) == 1
    assert c_sys.dim == 3
    hs = generate_gate_1qutrit_single_gellmann_mat(gate_name)
    G = Gate(c_sys=c_sys, hs=hs)
    return G
