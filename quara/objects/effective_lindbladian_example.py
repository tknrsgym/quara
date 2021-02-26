import math
import numpy as np
from typing import List, Union

from quara.objects.matrix_basis import (
    get_pauli_basis,
)
from quara.objects.composite_system import CompositeSystem
from quara.objects.gate import Gate
from quara.objects.effective_lindbladian import EffectiveLindbladian


# Unified functions


def get_gate_names() -> List[str]:
    """Return the list of valid gate names."""
    names = []
    names.extend(get_gate_names_1qubit())
    return names


def get_gate_names_1qubit() -> List[str]:
    """Return the list of valid gate names of 1-qubit gates."""
    names = []
    names.append("identity")
    names.append("x90")
    names.append("x180")
    names.append("x")
    names.append("y90")
    names.append("y180")
    names.append("y")
    names.append("z90")
    names.append("z180")
    names.append("z")

    return names


def object_names() -> List[str]:
    """Return the list of valid object names for generate_object_of_gate_from_gate_name_object_name()."""
    names = []
    names.append("hamiltonian_vec")
    names.append("hamiltonian_mat")
    names.append("unitary_mat")
    names.append("effective_lindbladian_mat")
    names.append("effective_lindbladian_class")
    names.append("gate_mat")
    names.append("gate_class")

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


def generate_gate_object_from_gate_name_object_name(
    gate_name: str,
    object_name: str,
    dims: List[int] = [],
    ids: List[int] = [],
    c_sys: CompositeSystem = None,
) -> Union[np.array, "EffectiveLindbladian", "Gate"]:
    if object_name == "hamiltonian_vec":
        obj = generate_hamiltonian_vec_from_gate_name(gate_name, dims, ids)
    elif object_name == "hamiltonian_mat":
        obj = generate_hamiltonian_mat_from_gate_name(gate_name, dims, ids)
    elif object_name == "unitary_mat":
        obj = generate_unitary_mat_from_gate_name(gate_name, dims, ids)
    elif object_name == "effective_lindbladian_mat":
        obj = generate_effective_lindbladian_mat_from_gate_name(gate_name, dims, ids)
    elif object_name == "gate_mat":
        obj = generate_gate_mat_from_gate_name(gate_name, dims, ids)
    elif object_name == "effective_lindbladian_class":
        obj = generate_effective_lindbladian_from_gate_name(gate_name, c_sys, ids)
    elif object_name == "gate_class":
        obj = generate_gate_from_gate_name(gate_name, c_sys, ids)
    else:
        raise ValueError(f"object_name is out of range.")
    return obj


def generate_hamiltonian_vec_from_gate_name(
    gate_name: str, dims: List[int] = [], ids: List[int] = []
) -> np.array:
    _is_valid_dims_ids(dims, ids)

    if gate_name == "identity":
        dim_total = _dim_total_from_dims(dims)
        if dim_total <= 1:
            raise ValueError(f"dim_total must be larger than 1.")
        vec = generate_gate_identity_hamiltonian_vec(dim_total)
    # 1-qubit gate
    elif gate_name == "x90":
        vec = generate_gate_x90_hamiltonian_vec()
    elif gate_name == "x180":
        vec = generate_gate_x180_hamiltonian_vec()
    elif gate_name == "x":
        vec = generate_gate_x_hamiltonian_vec()
    elif gate_name == "y90":
        vec = generate_gate_y90_hamiltonian_vec()
    elif gate_name == "y180":
        vec = generate_gate_y180_hamiltonian_vec()
    elif gate_name == "y":
        vec = generate_gate_y_hamiltonian_vec()
    elif gate_name == "z90":
        vec = generate_gate_z90_hamiltonian_vec()
    elif gate_name == "z180":
        vec = generate_gate_z180_hamiltonian_vec()
    elif gate_name == "z":
        vec = generate_gate_z_hamiltonian_vec()
    # 2-qubit gate
    # 3-qubit gate
    else:
        raise ValueError(f"gate_name is out of range.")

    return vec


def generate_hamiltonian_mat_from_gate_name(
    gate_name: str, dims: List[int] = [], ids: List[int] = []
) -> np.array:
    _is_valid_dims_ids(dims, ids)

    if gate_name == "identity":
        dim_total = _dim_total_from_dims(dims)
        if dim_total <= 1:
            raise ValueError(f"dim_total must be larger than 1.")
        vec = generate_gate_identity_hamiltonian_mat(dim_total)
    # 1-qubit gate
    elif gate_name == "x90":
        vec = generate_gate_x90_hamiltonian_mat()
    elif gate_name == "x180":
        vec = generate_gate_x180_hamiltonian_mat()
    elif gate_name == "x":
        vec = generate_gate_x_hamiltonian_mat()
    elif gate_name == "y90":
        vec = generate_gate_y90_hamiltonian_mat()
    elif gate_name == "y180":
        vec = generate_gate_y180_hamiltonian_mat()
    elif gate_name == "y":
        vec = generate_gate_y_hamiltonian_mat()
    elif gate_name == "z90":
        vec = generate_gate_z90_hamiltonian_mat()
    elif gate_name == "z180":
        vec = generate_gate_z180_hamiltonian_mat()
    elif gate_name == "z":
        vec = generate_gate_z_hamiltonian_mat()
    # 2-qubit gate
    # 3-qubit gate
    else:
        raise ValueError(f"gate_name is out of range.")

    return vec


def generate_unitary_mat_from_gate_name(
    gate_name: str, dims: List[int] = [], ids: List[int] = []
):
    _is_valid_dims_ids(dims, ids)

    if gate_name == "identity":
        dim_total = _dim_total_from_dims(dims)
        if dim_total <= 1:
            raise ValueError(f"dim_total must be larger than 1.")
        vec = generate_gate_identity_unitary_mat(dim_total)
    # 1-qubit gate
    elif gate_name == "x90":
        vec = generate_gate_x90_unitary_mat()
    elif gate_name == "x180":
        vec = generate_gate_x180_unitary_mat()
    elif gate_name == "x":
        vec = generate_gate_x_unitary_mat()
    elif gate_name == "y90":
        vec = generate_gate_y90_unitary_mat()
    elif gate_name == "y180":
        vec = generate_gate_y180_unitary_mat()
    elif gate_name == "y":
        vec = generate_gate_y_unitary_mat()
    elif gate_name == "z90":
        vec = generate_gate_z90_unitary_mat()
    elif gate_name == "z180":
        vec = generate_gate_z180_unitary_mat()
    elif gate_name == "z":
        vec = generate_gate_z_unitary_mat()
    # 2-qubit gate
    # 3-qubit gate
    else:
        raise ValueError(f"gate_name is out of range.")

    return vec


def generate_effective_lindbladian_mat_from_gate_name(
    gate_name: str, dims: List[int] = [], ids: List[int] = []
):
    _is_valid_dims_ids(dims, ids)

    if gate_name == "identity":
        dim_total = _dim_total_from_dims(dims)
        if dim_total <= 1:
            raise ValueError(f"dim_total must be larger than 1.")
        vec = generate_gate_identity_effective_lindbladian_mat(dim_total)
    # 1-qubit gate
    elif gate_name == "x90":
        vec = generate_gate_x90_effective_lindbladian_mat()
    elif gate_name == "x180":
        vec = generate_gate_x180_effective_lindbladian_mat()
    elif gate_name == "x":
        vec = generate_gate_x_effective_lindbladian_mat()
    elif gate_name == "y90":
        vec = generate_gate_y90_effective_lindbladian_mat()
    elif gate_name == "y180":
        vec = generate_gate_y180_effective_lindbladian_mat()
    elif gate_name == "y":
        vec = generate_gate_y_effective_lindbladian_mat()
    elif gate_name == "z90":
        vec = generate_gate_z90_effective_lindbladian_mat()
    elif gate_name == "z180":
        vec = generate_gate_z180_effective_lindbladian_mat()
    elif gate_name == "z":
        vec = generate_gate_z_effective_lindbladian_mat()
    # 2-qubit gate
    # 3-qubit gate
    else:
        raise ValueError(f"gate_name is out of range.")

    return vec


def generate_gate_mat_from_gate_name(
    gate_name: str, dims: List[int] = [], ids: List[int] = []
):
    _is_valid_dims_ids(dims, ids)

    if gate_name == "identity":
        dim_total = _dim_total_from_dims(dims)
        if dim_total <= 1:
            raise ValueError(f"dim_total must be larger than 1.")
        vec = generate_gate_identity_mat(dim_total)
    # 1-qubit gate
    elif gate_name == "x90":
        vec = generate_gate_x90_mat()
    elif gate_name == "x180":
        vec = generate_gate_x180_mat()
    elif gate_name == "x":
        vec = generate_gate_x_mat()
    elif gate_name == "y90":
        vec = generate_gate_y90_mat()
    elif gate_name == "y180":
        vec = generate_gate_y180_mat()
    elif gate_name == "y":
        vec = generate_gate_y_mat()
    elif gate_name == "z90":
        vec = generate_gate_z90_mat()
    elif gate_name == "z180":
        vec = generate_gate_z180_mat()
    elif gate_name == "z":
        vec = generate_gate_z_mat()
    # 2-qubit gate
    # 3-qubit gate
    else:
        raise ValueError(f"gate_name is out of range.")

    return vec


def generate_effective_lindbladian_from_gate_name(
    gate_name: str, c_sys: CompositeSystem, ids: List[int] = []
):
    if gate_name == "identity":
        vec = generate_gate_identity_effective_lindbladian(c_sys)
    # 1-qubit gate
    elif gate_name == "x90":
        vec = generate_gate_x90_effective_lindbladian(c_sys)
    elif gate_name == "x180":
        vec = generate_gate_x180_effective_lindbladian(c_sys)
    elif gate_name == "x":
        vec = generate_gate_x_effective_lindbladian(c_sys)
    elif gate_name == "y90":
        vec = generate_gate_y90_effective_lindbladian(c_sys)
    elif gate_name == "y180":
        vec = generate_gate_y180_effective_lindbladian(c_sys)
    elif gate_name == "y":
        vec = generate_gate_y_effective_lindbladian(c_sys)
    elif gate_name == "z90":
        vec = generate_gate_z90_effective_lindbladian(c_sys)
    elif gate_name == "z180":
        vec = generate_gate_z180_effective_lindbladian(c_sys)
    elif gate_name == "z":
        vec = generate_gate_z_effective_lindbladian(c_sys)
    # 2-qubit gate
    # 3-qubit gate
    else:
        raise ValueError(f"gate_name is out of range.")

    return vec


def generate_gate_from_gate_name(
    gate_name: str, c_sys: CompositeSystem, ids: List[int] = []
):
    if gate_name == "identity":
        vec = generate_gate_identity(c_sys)
    # 1-qubit gate
    elif gate_name == "x90":
        vec = generate_gate_x90(c_sys)
    elif gate_name == "x180":
        vec = generate_gate_x180(c_sys)
    elif gate_name == "x":
        vec = generate_gate_x(c_sys)
    elif gate_name == "y90":
        vec = generate_gate_y90(c_sys)
    elif gate_name == "y180":
        vec = generate_gate_y180(c_sys)
    elif gate_name == "y":
        vec = generate_gate_y(c_sys)
    elif gate_name == "z90":
        vec = generate_gate_z90(c_sys)
    elif gate_name == "z180":
        vec = generate_gate_z180(c_sys)
    elif gate_name == "z":
        vec = generate_gate_z(c_sys)
    # 2-qubit gate
    # 3-qubit gate
    else:
        raise ValueError(f"gate_name is out of range.")

    return vec


# Identity gate


def generate_gate_identity_hamiltonian_vec(dim: int) -> np.array:
    """Return the vector representation for the Hamiltonian of an identity gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is the real zero vector with size dim^2.

    Parameters
    ----------
    dim : int
        The dimension of the quantum system on which the gate acts.

    Returns
    ----------
    np.array
        The real vector representation of the Hamiltonian of the gate.
    """
    vec = np.zeros(dim * dim, dtype=np.float64)
    return vec


def generate_gate_identity_hamiltonian_mat(dim: int) -> np.array:
    """Return Hamiltonian matrix for an identity gate.

    The result is the dim times dim complex zero matrix.

    Parameters
    ----------
    dim : int
        The dimension of the quantum system on which the gate acts.

    Returns
    ----------
    np.array
        The Hamiltonian, which is a complex matrix.
    """
    mat = np.zeros((dim, dim), dtype=np.complex128)
    return mat


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


def generate_gate_identity_effective_lindbladian_mat(dim: int) -> np.array:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of an Identity gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is the dim^2 times dim^2 real zero matrix.

    Parameters
    ----------
    dim : int
        The dimension of the quantum system on which the gate acts.

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the effective lindbladian of the gate. It is the zero matrix in this case.
    """
    size = dim * dim
    mat = np.zeros((size, size), dtype=np.float64)
    return mat


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


def generate_gate_identity_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the identity gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    dim = c_sys.dim
    hs = generate_gate_identity_effective_lindbladian_mat(dim)
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# X90 gate on 1-qubit


def generate_gate_x90_hamiltonian_vec() -> np.array:
    """Return the vector representation for the Hamiltonian of an X90 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.25 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[1] = coeff
    return vec


def generate_gate_x90_hamiltonian_mat() -> np.array:
    """Return Hamiltonian matrix for an X90 gate.

    The result is the 2 times 2 complex matrix, 0.25 * pi * X.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    index = 1
    coeff = 0.25 * math.pi
    mat = coeff * get_pauli_basis(num_qubit)[index]
    return mat


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


def generate_gate_x90_effective_lindbladian_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of an X90 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the effective lindbladian of the gate.
    """
    l = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]
    coeff = 0.50 * math.pi
    mat = coeff * np.array(l, dtype=np.float64)
    return mat


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
    hs = generate_gate_x90_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


def generate_gate_x90_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the X90 gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    hs = generate_gate_x90_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# X180 gate on 1-qubit


def generate_gate_x180_hamiltonian_vec() -> np.array:
    """Return the vector representation for the Hamiltonian of an X180 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.5 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[1] = coeff
    return vec


def generate_gate_x180_hamiltonian_mat() -> np.array:
    """Return Hamiltonian matrix for an X180 gate.

    The result is the 2 times 2 complex matrix, 0.5 * pi * X.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    index = 1
    coeff = 0.5 * math.pi
    mat = coeff * get_pauli_basis(num_qubit)[index]
    return mat


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


def generate_gate_x180_effective_lindbladian_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of an X180 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the effective lindbladian of the gate.
    """
    l = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]
    coeff = math.pi
    mat = coeff * np.array(l, dtype=np.float64)
    return mat


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
    hs = generate_gate_x180_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


def generate_gate_x180_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the X180 gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    hs = generate_gate_x180_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# X gate on 1-qubit


def generate_gate_x_hamiltonian_vec() -> np.array:
    """Return the vector representation for the Hamiltonian of an X gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.5 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[0] = -coeff
    vec[1] = coeff
    return vec


def generate_gate_x_hamiltonian_mat() -> np.array:
    """Return Hamiltonian matrix for an X gate.

    The result is the 2 times 2 complex matrix, -0.5 * pi * I + 0.5 * pi * X.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    b = get_pauli_basis(num_qubit)
    coeff = 0.5 * math.pi
    mat = -coeff * b[0]
    mat += coeff * b[1]
    return mat


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


def generate_gate_x_effective_lindbladian_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of an X gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the effective lindbladian of the gate.
    """
    l = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]
    coeff = math.pi
    mat = coeff * np.array(l, dtype=np.float64)
    return mat


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
    hs = generate_gate_x_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


def generate_gate_x_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the X gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    hs = generate_gate_x_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Y90 on 1-qubit


def generate_gate_y90_hamiltonian_vec() -> np.array:
    """Return the vector representation for the Hamiltonian of a Y90 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.25 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[2] = coeff
    return vec


def generate_gate_y90_hamiltonian_mat() -> np.array:
    """Return Hamiltonian matrix for a Y90 gate.

    The result is the 2 times 2 complex matrix, 0.25 * pi * Y.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    index = 2
    coeff = 0.25 * math.pi
    mat = coeff * get_pauli_basis(num_qubit)[index]
    return mat


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


def generate_gate_y90_effective_lindbladian_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a Y90 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the effective lindbladian of the gate.
    """
    l = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, -1, 0, 0]]
    coeff = 0.50 * math.pi
    mat = coeff * np.array(l, dtype=np.float64)
    return mat


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
    hs = generate_gate_y90_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


def generate_gate_y90_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the Y90 gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    hs = generate_gate_y90_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Y180 gate on 1-qubit


def generate_gate_y180_hamiltonian_vec() -> np.array:
    """Return the vector representation for the Hamiltonian of a Y180 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.5 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[2] = coeff
    return vec


def generate_gate_y180_hamiltonian_mat() -> np.array:
    """Return Hamiltonian matrix for a Y180 gate.

    The result is the 2 times 2 complex matrix, 0.5 * pi * Y.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    index = 2
    coeff = 0.5 * math.pi
    mat = coeff * get_pauli_basis(num_qubit)[index]
    return mat


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


def generate_gate_y180_effective_lindbladian_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a Y180 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the effective lindbladian of the gate.
    """
    l = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, -1, 0, 0]]
    coeff = math.pi
    mat = coeff * np.array(l, dtype=np.float64)
    return mat


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
    hs = generate_gate_y180_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


def generate_gate_y180_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the Y180 gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    hs = generate_gate_y180_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Y gate on 1-qubit


def generate_gate_y_hamiltonian_vec() -> np.array:
    """Return the vector representation for the Hamiltonian of a Y gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.5 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[0] = -coeff
    vec[2] = coeff
    return vec


def generate_gate_y_hamiltonian_mat() -> np.array:
    """Return Hamiltonian for a Y gate.

    The result is the 2 times 2 complex matrix, -0.5 * pi * I + 0.5 * pi * Y.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    b = get_pauli_basis(num_qubit)
    coeff = 0.5 * math.pi
    mat = -coeff * b[0]
    mat += coeff * b[2]
    return mat


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


def generate_gate_y_effective_lindbladian_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a Y gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the effective lindbladian of the gate.
    """
    l = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, -1, 0, 0]]
    coeff = math.pi
    mat = coeff * np.array(l, dtype=np.float64)
    return mat


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
    hs = generate_gate_y_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


def generate_gate_y_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the Y180 gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    hs = generate_gate_y_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Z90 gate on 1-qubit


def generate_gate_z90_hamiltonian_vec() -> np.array:
    """Return the vector representation for the Hamiltonian of a Z90 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is the real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.25 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[3] = coeff
    return vec


def generate_gate_z90_hamiltonian_mat() -> np.array:
    """Return Hamiltonian matrix for a Z90 gate.

    The result is the 2 times 2 complex matrix, 0.25 * pi * Z.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    index = 3
    coeff = 0.25 * math.pi
    mat = coeff * get_pauli_basis(num_qubit)[index]
    return mat


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


def generate_gate_z90_effective_lindbladian_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a Z90 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the effective lindbladian of the gate.
    """
    l = [[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
    coeff = 0.50 * math.pi
    mat = coeff * np.array(l, dtype=np.float64)
    return mat


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
    hs = generate_gate_z90_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


def generate_gate_z90_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the Z90 gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    hs = generate_gate_z90_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Z180 gate on 1-qubit


def generate_gate_z180_hamiltonian_vec() -> np.array:
    """Return the vector representation for the Hamiltonian of a Z180 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.5 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[3] = coeff
    return vec


def generate_gate_z180_hamiltonian_mat() -> np.array:
    """Return Hamiltonian matrix for a Z180 gate.

    The result is the 2 times 2 complex matrix, 0.5 * pi * Z.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    index = 3
    coeff = 0.5 * math.pi
    mat = coeff * get_pauli_basis(num_qubit)[index]
    return mat


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


def generate_gate_z180_effective_lindbladian_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a Z180 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the effective lindbladian of the gate.
    """
    l = [[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
    coeff = math.pi
    mat = coeff * np.array(l, dtype=np.float64)
    return mat


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
    hs = generate_gate_z180_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


def generate_gate_z180_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the Z180 gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    hs = generate_gate_z180_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Z gate on 1-qubit


def generate_gate_z_hamiltonian_vec() -> np.array:
    """Return the vector representation for the Hamiltonian of a Z gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.5 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[0] = -coeff
    vec[3] = coeff
    return vec


def generate_gate_z_hamiltonian_mat() -> np.array:
    """Return Hamiltonian matrix for a Z gate.

    The result is the 2 times 2 complex matrix, -0.5 * pi * I + 0.5 * pi * Z.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    b = get_pauli_basis(num_qubit)
    coeff = 0.5 * math.pi
    mat = -coeff * b[0]
    mat += coeff * b[3]
    return mat


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


def generate_gate_z_effective_lindbladian_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a Z gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the effective lindbladian of the gate.
    """
    l = [[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
    coeff = math.pi
    mat = coeff * np.array(l, dtype=np.float64)
    return mat


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
    hs = generate_gate_z_mat()
    gate = Gate(c_sys=c_sys, hs=hs)
    return gate


def generate_gate_z_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the Z gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    hs = generate_gate_z180_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el
