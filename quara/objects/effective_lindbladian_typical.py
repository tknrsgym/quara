import math
import numpy as np
from typing import List, Union

from quara.objects.matrix_basis import (
    get_pauli_basis,
)
from quara.objects.composite_system import CompositeSystem
from quara.objects.gate import Gate
from quara.objects.gate_typical import (
    _is_valid_dims_ids,
    _dim_total_from_dims,
    get_gate_names,
    get_gate_names_1qubit,
)
from quara.objects.effective_lindbladian import EffectiveLindbladian


def generate_hamiltonian_vec_from_gate_name(
    gate_name: str, dims: List[int] = [], ids: List[int] = []
) -> np.array:
    """returns the vector representation of the Hamiltonian of a gate.

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
        The vector for the Hamiltonian matrix, to be real.
    """
    _is_valid_dims_ids(dims, ids)
    assert gate_name in get_gate_names()

    method_name = "generate_gate_" + gate_name + "_hamiltonian_vec"
    method = eval(method_name)

    if gate_name == "identity":
        dim_total = _dim_total_from_dims(dims)
        if dim_total <= 1:
            raise ValueError(f"dim_total must be larger than 1.")
        vec = method(dim_total)
    # 1-qubit gate
    elif gate_name in get_gate_names_1qubit():
        vec = method()
    # 2-qubit gate
    # 3-qubit gate
    else:
        raise ValueError(f"gate_name is out of range.")

    return vec


def generate_hamiltonian_mat_from_gate_name(
    gate_name: str, dims: List[int] = [], ids: List[int] = []
) -> np.array:
    """returns the Hamiltonian matrix of a gate.

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
        The Hamiltonian matrix the gate, to be complex.
    """
    _is_valid_dims_ids(dims, ids)
    assert gate_name in get_gate_names()

    method_name = "generate_gate_" + gate_name + "_hamiltonian_mat"
    method = eval(method_name)

    if gate_name == "identity":
        dim_total = _dim_total_from_dims(dims)
        if dim_total <= 1:
            raise ValueError(f"dim_total must be larger than 1.")
        mat = method(dim_total)
    # 1-qubit gate
    elif gate_name in get_gate_names_1qubit():
        mat = method()
    # 2-qubit gate
    # 3-qubit gate
    else:
        raise ValueError(f"gate_name is out of range.")

    return mat


def generate_effective_lindbladian_mat_from_gate_name(
    gate_name: str, dims: List[int] = [], ids: List[int] = []
) -> np.array:
    """returns the Hilbert-Schmidt representation matrix of an effective lindbladian.

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
        The HS matrix of the effective lindbladian, to be real.
    """
    _is_valid_dims_ids(dims, ids)
    assert gate_name in get_gate_names()

    method_name = "generate_gate_" + gate_name + "_effective_lindbladian_mat"
    method = eval(method_name)

    if gate_name == "identity":
        dim_total = _dim_total_from_dims(dims)
        if dim_total <= 1:
            raise ValueError(f"dim_total must be larger than 1.")
        mat = method(dim_total)
    # 1-qubit gate
    elif gate_name in get_gate_names_1qubit():
        mat = method()
    # 2-qubit gate
    # 3-qubit gate
    else:
        raise ValueError(f"gate_name is out of range.")

    return mat


def generate_effective_lindbladian_from_gate_name(
    gate_name: str, c_sys: CompositeSystem, ids: List[int] = []
) -> "EffectiveLindbladian":
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
    EffectiveLindbladian
        The effective lindbladian class object of the gate.
    """
    assert gate_name in get_gate_names()

    method_name = "generate_gate_" + gate_name + "_effective_lindbladian"
    method = eval(method_name)

    if gate_name == "identity":
        el = method(c_sys)
    # 1-qubit gate
    elif gate_name in get_gate_names_1qubit():
        el = method(c_sys)
    # 2-qubit gate
    # 3-qubit gate
    else:
        raise ValueError(f"gate_name is out of range.")

    return el


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
    hs = generate_gate_z_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Phase (S) gate on 1-qubit


def generate_gate_phase_hamiltonian_vec() -> np.array:
    """Return the vector representation for the Hamiltonian of a Phase (S) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

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
    vec[0] = -coeff
    vec[3] = coeff
    return vec


def generate_gate_phase_hamiltonian_mat() -> np.array:
    """Return Hamiltonian matrix for a Phase (S) gate.

    The result is the 2 times 2 complex matrix, -0.25 * pi * I + 0.25 * pi * Z.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    b = get_pauli_basis(num_qubit)
    coeff = 0.25 * math.pi
    mat = -coeff * b[0]
    mat += coeff * b[3]
    return mat


def generate_gate_phase_effective_lindbladian_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a Phase (S) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

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


def generate_gate_phase_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the Phase (S) gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    hs = generate_gate_phase_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Phase daggered (S^dagger) gate on 1-qubit


def generate_gate_phase_daggered_hamiltonian_vec() -> np.array:
    """Return the vector representation for the Hamiltonian of a Phase daggered (S^dagger) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

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
    vec[0] = coeff
    vec[3] = -coeff
    return vec


def generate_gate_phase_daggered_hamiltonian_mat() -> np.array:
    """Return Hamiltonian matrix for a Phase daggerd (S^dagger) gate.

    The result is the 2 times 2 complex matrix, -0.25 * pi * I + 0.25 * pi * Z.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    b = get_pauli_basis(num_qubit)
    coeff = 0.25 * math.pi
    mat = coeff * b[0]
    mat -= coeff * b[3]
    return mat


def generate_gate_phase_daggered_effective_lindbladian_mat() -> np.array:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a Phase daggered (S^dagger) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The real Hilbert-Schmidt representation matrix for the effective lindbladian of the gate.
    """
    l = [[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
    coeff = -0.50 * math.pi
    mat = coeff * np.array(l, dtype=np.float64)
    return mat


def generate_gate_phase_daggered_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the Phase daggered (S^dagger) gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    hs = generate_gate_phase_daggered_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el
