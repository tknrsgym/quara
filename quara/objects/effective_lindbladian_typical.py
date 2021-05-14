import math
import numpy as np
from typing import List, Union

from quara.utils.matrix_util import is_hermitian
from quara.objects.matrix_basis import MatrixBasis
from quara.objects.matrix_basis import (
    get_comp_basis,
    get_pauli_basis,
    get_normalized_pauli_basis,
    get_normalized_gell_mann_basis,
    get_normalized_generalized_gell_mann_basis,
    calc_hermitian_matrix_expansion_coefficient_hermitian_basis,
)
from quara.objects.composite_system import CompositeSystem
from quara.objects.gate import Gate
from quara.objects.gate import convert_hs
from quara.objects.gate_typical import (
    _is_valid_dims_ids,
    _dim_total_from_dims,
    get_gate_names,
    get_gate_names_1qubit,
    get_gate_names_2qubit,
    get_gate_names_2qubit_asymmetric,
    get_gate_names_3qubit,
    generate_gate_toffoli_hamiltonian_mat,
    generate_gate_fredkin_hamiltonian_mat,
    # 1-qutrit
    get_gate_names_1qutrit,
    get_gate_names_1qutrit_single_gellmann,
    calc_base_matrix_1qutrit,
    calc_levels_axis_angle_from_gate_name_1qutrit_single_gellmann,
    generate_gate_1qutrit_single_gellmann_hamiltonian_mat,
    # 2-qutrit
    get_gate_names_2qutrit,
    generate_gate_2qutrit_hamiltonian_mat_from_gate_name,
)
from quara.objects.effective_lindbladian import EffectiveLindbladian
from quara.objects.effective_lindbladian import _truncate_hs


def generate_effective_lindbladian_object_from_gate_name_object_name(
    gate_name: str,
    object_name: str,
    dims: List[int] = None,
    ids: List[int] = None,
    c_sys: CompositeSystem = None,
) -> Union[np.ndarray, "EffectiveLindbladian"]:
    if object_name == "hamiltonian_vec":
        obj = generate_hamiltonian_vec_from_gate_name(gate_name, dims, ids)
    elif object_name == "hamiltonian_mat":
        obj = generate_hamiltonian_mat_from_gate_name(gate_name, dims, ids)
    elif object_name == "effective_lindbladian_mat":
        obj = generate_effective_lindbladian_mat_from_gate_name(gate_name, dims, ids)
    elif object_name == "effective_lindbladian":
        obj = generate_effective_lindbladian_from_gate_name(gate_name, c_sys, ids)
    else:
        raise ValueError(f"object_name is out of range.")
    return obj


def calc_effective_lindbladian_mat_comp_basis_from_hamiltonian(
    h: np.ndarray,
) -> np.ndarray:
    """return the HS matrix of an effective Lindbladian w.r.t. the computational basis from a given Hamiltonian.

    Parameters
    ----------
    h : np.ndarray((dim, dim), dtype=np.complex128)
        A Hamiltonian, to be an Hermitian matrix.

    Returns
    ----------
    np.ndarray((dim^2, dim^2), dtype=np.complex128)
        The HS matrix of an effective Lindbladian characterized by the Hamiltonian.
    """
    shape = h.shape
    assert shape[0] == shape[1]
    assert is_hermitian(h)

    I = np.eye(shape[0], dtype=np.complex128)
    L = -1j * np.kron(h, I) + 1j * np.kron(I, np.conjugate(h))
    return L


def calc_effective_lindbladian_mat_from_hamiltonian(
    h: np.ndarray, to_basis: MatrixBasis
) -> np.ndarray:
    """return the HS matrix of an effective Lindbladian w.r.t. the given matrix basis from a given Hamiltonian.

    Parameters
    ----------
    h : np.ndarray((dim, dim), dtype=np.complex128)
        A Hamiltonian, to be an Hermitian matrix.

    bo_basis : MatrixBasis
        An orthonormal matrix basis.

    Returns
    ----------
    np.ndarray((dim^2, dim^2), dtype=np.complex128)
        The HS matrix of an effective Lindbladian characterized by the Hamiltonian.

    """
    shape = h.shape
    dim = shape[0]
    assert to_basis.dim == dim
    assert to_basis.is_orthogonal() == True
    assert to_basis.is_normal() == True

    L_comp = calc_effective_lindbladian_mat_comp_basis_from_hamiltonian(h)
    basis_comp = get_comp_basis(dim)
    L = convert_hs(from_hs=L_comp, from_basis=basis_comp, to_basis=to_basis)
    return L


def calc_effective_lindbladian_mat_hermitian_basis_from_hamiltonian(
    h: np.ndarray, to_basis: MatrixBasis
) -> np.ndarray:
    """return the HS matrix of an effective Lindbladian w.r.t. the given Hermitian matrix basis from a given Hamiltonian.

    Parameters
    ----------
    h : np.ndarray((dim, dim), dtype=np.complex128)
        A Hamiltonian, to be an Hermitian matrix.

    bo_basis : MatrixBasis
        An orthonormal Hermitian matrix basis.

    Returns
    ----------
    np.ndarray((dim^2, dim^2), dtype=np.float128)
        The HS matrix of an effective Lindbladian characterized by the Hamiltonian.
    """
    assert to_basis.is_hermitian() == True
    L_complex = calc_effective_lindbladian_mat_from_hamiltonian(h, to_basis)
    L = _truncate_hs(L_complex)
    return L


def generate_hamiltonian_vec_from_gate_name(
    gate_name: str, dims: List[int] = None, ids: List[int] = None
) -> np.ndarray:
    """return the vector representation of the Hamiltonian of a gate.

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
    np.ndarray
        The vector for the Hamiltonian matrix, to be real.
    """
    _is_valid_dims_ids(dims, ids)
    assert gate_name in get_gate_names()

    if gate_name == "identity":
        dim_total = _dim_total_from_dims(dims)
        if dim_total <= 1:
            raise ValueError(f"dim_total must be larger than 1.")
        method_name = "generate_gate_" + gate_name + "_hamiltonian_vec"
        method = eval(method_name)
        vec = method(dim_total)
    # 1-qubit gate
    elif gate_name in get_gate_names_1qubit():
        method_name = "generate_gate_" + gate_name + "_hamiltonian_vec"
        method = eval(method_name)
        vec = method()
    # 2-qubit gate
    elif gate_name in get_gate_names_2qubit():
        method_name = "generate_gate_" + gate_name + "_hamiltonian_vec"
        method = eval(method_name)
        if gate_name in get_gate_names_2qubit_asymmetric():
            vec = method(ids)
        else:
            vec = method()
    # 3-qubit gate
    elif gate_name in get_gate_names_3qubit():
        b = get_normalized_pauli_basis(n_qubit=3)
        method_name = "generate_gate_" + gate_name + "_hamiltonian_mat"
        method = eval(method_name)
        mat = method(ids)
        vec = calc_hermitian_matrix_expansion_coefficient_hermitian_basis(
            from_mat=mat, basis=b
        )
    # 1-qutrit gate
    elif gate_name in get_gate_names_1qutrit():
        if gate_name in get_gate_names_1qutrit_single_gellmann():
            method_name = "generate_gate_1qutrit_single_gellmann_hamiltonian_vec"
            method = eval(method_name)
            vec = method(gate_name)
    # 2-qutrit
    elif gate_name in get_gate_names_2qutrit():
        b = get_normalized_generalized_gell_mann_basis(n_qubit=2, dim=3)
        mat = generate_gate_2qutrit_hamiltonian_mat_from_gate_name(gate_name, ids)
        vec = calc_hermitian_matrix_expansion_coefficient_hermitian_basis(
            from_mat=mat, basis=b
        )
    else:
        raise ValueError(f"gate_name is out of range.")

    return vec


def generate_hamiltonian_mat_from_gate_name(
    gate_name: str, dims: List[int] = None, ids: List[int] = None
) -> np.ndarray:
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
    np.ndarray
        The Hamiltonian matrix the gate, to be complex.
    """
    _is_valid_dims_ids(dims, ids)
    assert gate_name in get_gate_names()

    if gate_name == "identity":
        dim_total = _dim_total_from_dims(dims)
        if dim_total <= 1:
            raise ValueError(f"dim_total must be larger than 1.")
        method_name = "generate_gate_" + gate_name + "_hamiltonian_mat"
        method = eval(method_name)
        mat = method(dim_total)
    # 1-qubit gate
    elif gate_name in get_gate_names_1qubit():
        method_name = "generate_gate_" + gate_name + "_hamiltonian_mat"
        method = eval(method_name)
        mat = method()
    # 2-qubit gate
    elif gate_name in get_gate_names_2qubit():
        method_name = "generate_gate_" + gate_name + "_hamiltonian_mat"
        method = eval(method_name)
        if gate_name in get_gate_names_2qubit_asymmetric():
            mat = method(ids)
        else:
            mat = method()
    # 3-qubit gate
    elif gate_name in get_gate_names_3qubit():
        method_name = "generate_gate_" + gate_name + "_hamiltonian_mat"
        method = eval(method_name)
        mat = method(ids)
    # 1-qutrit gate
    elif gate_name in get_gate_names_1qutrit():
        if gate_name in get_gate_names_1qutrit_single_gellmann():
            method_name = "generate_gate_1qutrit_single_gellmann_hamiltonian_mat"
            method = eval(method_name)
            mat = method(gate_name)
    # 2-qutrit
    elif gate_name in get_gate_names_2qutrit():
        mat = generate_gate_2qutrit_hamiltonian_mat_from_gate_name(gate_name, ids)
    else:
        raise ValueError(f"gate_name is out of range.")

    return mat


def generate_effective_lindbladian_mat_from_gate_name(
    gate_name: str, dims: List[int] = None, ids: List[int] = None
) -> np.ndarray:
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
    np.ndarray
        The HS matrix of the effective lindbladian, to be real.
    """
    _is_valid_dims_ids(dims, ids)
    assert gate_name in get_gate_names()

    if gate_name == "identity":
        dim_total = _dim_total_from_dims(dims)
        if dim_total <= 1:
            raise ValueError(f"dim_total must be larger than 1.")
        method_name = "generate_gate_" + gate_name + "_effective_lindbladian_mat"
        method = eval(method_name)
        mat = method(dim_total)
    # 1-qubit gate
    elif gate_name in get_gate_names_1qubit():
        method_name = "generate_gate_" + gate_name + "_effective_lindbladian_mat"
        method = eval(method_name)
        mat = method()
    # 2-qubit gate
    elif gate_name in get_gate_names_2qubit():
        method_name = "generate_gate_" + gate_name + "_effective_lindbladian_mat"
        method = eval(method_name)
        if gate_name in get_gate_names_2qubit_asymmetric():
            mat = method(ids)
        else:
            mat = method()
    # 3-qubit gate
    elif gate_name in get_gate_names_3qubit():
        basis = get_normalized_pauli_basis(n_qubit=3)
        method_name = "generate_gate_" + gate_name + "_hamiltonian_mat"
        method = eval(method_name)
        h = method(ids)
        mat = calc_effective_lindbladian_mat_hermitian_basis_from_hamiltonian(
            h=h, to_basis=basis
        )
    # 1-qutrit gate
    elif gate_name in get_gate_names_1qutrit():
        if gate_name in get_gate_names_1qutrit_single_gellmann():
            method_name = (
                "generate_gate_1qutrit_single_gellmann_effective_lindbladian_mat"
            )
            method = eval(method_name)
            mat = method(gate_name)
    # 2-qutrit
    elif gate_name in get_gate_names_2qutrit():
        h = generate_gate_2qutrit_hamiltonian_mat_from_gate_name(gate_name, ids)
        basis = get_normalized_generalized_gell_mann_basis(n_qubit=2, dim=3)
        mat = calc_effective_lindbladian_mat_hermitian_basis_from_hamiltonian(
            h=h, to_basis=basis
        )
    else:
        raise ValueError(f"gate_name is out of range.")

    return mat


def generate_effective_lindbladian_from_gate_name(
    gate_name: str, c_sys: CompositeSystem, ids: List[int] = None
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

    if gate_name == "identity":
        method_name = "generate_gate_" + gate_name + "_effective_lindbladian"
        method = eval(method_name)
        el = method(c_sys)
    # 1-qubit gate
    elif gate_name in get_gate_names_1qubit():
        method_name = "generate_gate_" + gate_name + "_effective_lindbladian"
        method = eval(method_name)
        el = method(c_sys)
    # 2-qubit gate
    elif gate_name in get_gate_names_2qubit():
        method_name = "generate_gate_" + gate_name + "_effective_lindbladian"
        method = eval(method_name)
        if gate_name in get_gate_names_2qubit_asymmetric():
            el = method(c_sys, ids)
        else:
            el = method(c_sys)
    # 3-qubit gate
    elif gate_name in get_gate_names_3qubit():
        basis = get_normalized_pauli_basis(n_qubit=3)
        method_name = "generate_gate_" + gate_name + "_hamiltonian_mat"
        method = eval(method_name)
        h = method(ids)
        mat = calc_effective_lindbladian_mat_hermitian_basis_from_hamiltonian(
            h=h, to_basis=basis
        )
        el = EffectiveLindbladian(c_sys=c_sys, hs=mat)
    # 1-qutrit gate
    elif gate_name in get_gate_names_1qutrit():
        if gate_name in get_gate_names_1qutrit_single_gellmann():
            method_name = "generate_gate_1qutrit_single_gellmann_effective_linabladian"
            method = eval(method_name)
            el = method(c_sys, gate_name)
    # 2-qutrit
    elif gate_name in get_gate_names_2qutrit():
        h = generate_gate_2qutrit_hamiltonian_mat_from_gate_name(gate_name, ids)
        basis = get_normalized_generalized_gell_mann_basis(n_qubit=2, dim=3)
        mat = calc_effective_lindbladian_mat_hermitian_basis_from_hamiltonian(
            h=h, to_basis=basis
        )
        el = EffectiveLindbladian(c_sys=c_sys, hs=mat)
    else:
        raise ValueError(f"gate_name is out of range.")

    return el


# Identity gate


def generate_gate_identity_hamiltonian_vec(dim: int) -> np.ndarray:
    """Return the vector representation for the Hamiltonian of an identity gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is the real zero vector with size dim^2.

    Parameters
    ----------
    dim : int
        The dimension of the quantum system on which the gate acts.

    Returns
    ----------
    np.ndarray
        The real vector representation of the Hamiltonian of the gate.
    """
    vec = np.zeros(dim * dim, dtype=np.float64)
    return vec


def generate_gate_identity_hamiltonian_mat(dim: int) -> np.ndarray:
    """Return Hamiltonian matrix for an identity gate.

    The result is the dim times dim complex zero matrix.

    Parameters
    ----------
    dim : int
        The dimension of the quantum system on which the gate acts.

    Returns
    ----------
    np.ndarray
        The Hamiltonian, which is a complex matrix.
    """
    mat = np.zeros((dim, dim), dtype=np.complex128)
    return mat


def generate_gate_identity_effective_lindbladian_mat(dim: int) -> np.ndarray:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of an Identity gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is the dim^2 times dim^2 real zero matrix.

    Parameters
    ----------
    dim : int
        The dimension of the quantum system on which the gate acts.

    Returns
    ----------
    np.ndarray
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


def generate_gate_x90_hamiltonian_vec() -> np.ndarray:
    """Return the vector representation for the Hamiltonian of an X90 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.25 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[1] = coeff
    return vec


def generate_gate_x90_hamiltonian_mat() -> np.ndarray:
    """Return Hamiltonian matrix for an X90 gate.

    The result is the 2 times 2 complex matrix, :math:`0.25 \\pi X`.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    index = 1
    coeff = 0.25 * math.pi
    mat = coeff * get_pauli_basis(num_qubit)[index]
    return mat


def generate_gate_x90_effective_lindbladian_mat() -> np.ndarray:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of an X90 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
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
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_x90_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# X180 gate on 1-qubit


def generate_gate_x180_hamiltonian_vec() -> np.ndarray:
    """Return the vector representation for the Hamiltonian of an X180 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.5 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[1] = coeff
    return vec


def generate_gate_x180_hamiltonian_mat() -> np.ndarray:
    """Return Hamiltonian matrix for an X180 gate.

    The result is the 2 times 2 complex matrix, :math:`0.5 \\pi X`.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    index = 1
    coeff = 0.5 * math.pi
    mat = coeff * get_pauli_basis(num_qubit)[index]
    return mat


def generate_gate_x180_effective_lindbladian_mat() -> np.ndarray:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of an X180 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
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
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_x180_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# X gate on 1-qubit


def generate_gate_x_hamiltonian_vec() -> np.ndarray:
    """Return the vector representation for the Hamiltonian of an X gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.5 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[0] = -coeff
    vec[1] = coeff
    return vec


def generate_gate_x_hamiltonian_mat() -> np.ndarray:
    """Return Hamiltonian matrix for an X gate.

    The result is the 2 times 2 complex matrix, :math:`-0.5 \\pi I + 0.5 \\pi X`.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    b = get_pauli_basis(num_qubit)
    coeff = 0.5 * math.pi
    mat = -coeff * b[0]
    mat += coeff * b[1]
    return mat


def generate_gate_x_effective_lindbladian_mat() -> np.ndarray:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of an X gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
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
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_x_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Y90 on 1-qubit


def generate_gate_y90_hamiltonian_vec() -> np.ndarray:
    """Return the vector representation for the Hamiltonian of a Y90 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.25 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[2] = coeff
    return vec


def generate_gate_y90_hamiltonian_mat() -> np.ndarray:
    """Return Hamiltonian matrix for a Y90 gate.

    The result is the 2 times 2 complex matrix, :math:`0.25 \\pi Y`.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    index = 2
    coeff = 0.25 * math.pi
    mat = coeff * get_pauli_basis(num_qubit)[index]
    return mat


def generate_gate_y90_effective_lindbladian_mat() -> np.ndarray:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a Y90 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
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
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_y90_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Y180 gate on 1-qubit


def generate_gate_y180_hamiltonian_vec() -> np.ndarray:
    """Return the vector representation for the Hamiltonian of a Y180 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.5 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[2] = coeff
    return vec


def generate_gate_y180_hamiltonian_mat() -> np.ndarray:
    """Return Hamiltonian matrix for a Y180 gate.

    The result is the 2 times 2 complex matrix, :math:`0.5 \\pi Y`.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    index = 2
    coeff = 0.5 * math.pi
    mat = coeff * get_pauli_basis(num_qubit)[index]
    return mat


def generate_gate_y180_effective_lindbladian_mat() -> np.ndarray:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a Y180 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
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
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_y180_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Y gate on 1-qubit


def generate_gate_y_hamiltonian_vec() -> np.ndarray:
    """Return the vector representation for the Hamiltonian of a Y gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.5 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[0] = -coeff
    vec[2] = coeff
    return vec


def generate_gate_y_hamiltonian_mat() -> np.ndarray:
    """Return Hamiltonian for a Y gate.

    The result is the 2 times 2 complex matrix, :math:`-0.5 \\pi I + 0.5 \\pi Y`.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    b = get_pauli_basis(num_qubit)
    coeff = 0.5 * math.pi
    mat = -coeff * b[0]
    mat += coeff * b[2]
    return mat


def generate_gate_y_effective_lindbladian_mat() -> np.ndarray:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a Y gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
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
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_y_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Z90 gate on 1-qubit


def generate_gate_z90_hamiltonian_vec() -> np.ndarray:
    """Return the vector representation for the Hamiltonian of a Z90 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is the real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.25 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[3] = coeff
    return vec


def generate_gate_z90_hamiltonian_mat() -> np.ndarray:
    """Return Hamiltonian matrix for a Z90 gate.

    The result is the 2 times 2 complex matrix, :math:`0.25 \\pi Z`.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    index = 3
    coeff = 0.25 * math.pi
    mat = coeff * get_pauli_basis(num_qubit)[index]
    return mat


def generate_gate_z90_effective_lindbladian_mat() -> np.ndarray:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a Z90 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
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
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_z90_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Z180 gate on 1-qubit


def generate_gate_z180_hamiltonian_vec() -> np.ndarray:
    """Return the vector representation for the Hamiltonian of a Z180 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.5 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[3] = coeff
    return vec


def generate_gate_z180_hamiltonian_mat() -> np.ndarray:
    """Return Hamiltonian matrix for a Z180 gate.

    The result is the 2 times 2 complex matrix, :math:`0.5 \\pi Z`.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    index = 3
    coeff = 0.5 * math.pi
    mat = coeff * get_pauli_basis(num_qubit)[index]
    return mat


def generate_gate_z180_effective_lindbladian_mat() -> np.ndarray:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a Z180 gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
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
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_z180_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Z gate on 1-qubit


def generate_gate_z_hamiltonian_vec() -> np.ndarray:
    """Return the vector representation for the Hamiltonian of a Z gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.5 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[0] = -coeff
    vec[3] = coeff
    return vec


def generate_gate_z_hamiltonian_mat() -> np.ndarray:
    """Return Hamiltonian matrix for a Z gate.

    The result is the 2 times 2 complex matrix, :math:`-0.5 \\pi I + 0.5 \\pi Z`.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    b = get_pauli_basis(num_qubit)
    coeff = 0.5 * math.pi
    mat = -coeff * b[0]
    mat += coeff * b[3]
    return mat


def generate_gate_z_effective_lindbladian_mat() -> np.ndarray:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a Z gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
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
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_z_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Phase (S) gate on 1-qubit


def generate_gate_phase_hamiltonian_vec() -> np.ndarray:
    """Return the vector representation for the Hamiltonian of a Phase (S) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.25 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[0] = -coeff
    vec[3] = coeff
    return vec


def generate_gate_phase_hamiltonian_mat() -> np.ndarray:
    """Return Hamiltonian matrix for a Phase (S) gate.

    The result is the 2 times 2 complex matrix, :math:`-0.25 \\pi I + 0.25 \\pi Z`.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    b = get_pauli_basis(num_qubit)
    coeff = 0.25 * math.pi
    mat = -coeff * b[0]
    mat += coeff * b[3]
    return mat


def generate_gate_phase_effective_lindbladian_mat() -> np.ndarray:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a Phase (S) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
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
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_phase_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Phase daggered (S^dagger) gate on 1-qubit


def generate_gate_phase_daggered_hamiltonian_vec() -> np.ndarray:
    """Return the vector representation for the Hamiltonian of a Phase daggered (S^dagger) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.25 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[0] = coeff
    vec[3] = -coeff
    return vec


def generate_gate_phase_daggered_hamiltonian_mat() -> np.ndarray:
    """Return Hamiltonian matrix for a Phase daggerd (S^dagger) gate.

    The result is the 2 times 2 complex matrix, :math:`0.25 \\pi I - 0.25 \\pi Z`.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    b = get_pauli_basis(num_qubit)
    coeff = 0.25 * math.pi
    mat = coeff * b[0]
    mat -= coeff * b[3]
    return mat


def generate_gate_phase_daggered_effective_lindbladian_mat() -> np.ndarray:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a Phase daggered (S^dagger) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real Hilbert-Schmidt representation matrix for the effective lindbladian of the gate.
    """
    l = [[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
    coeff = -0.50 * math.pi
    mat = coeff * np.array(l, dtype=np.float64)
    return mat


def generate_gate_phase_daggered_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the Phase daggered (:math:`S^\\dagger`) gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_phase_daggered_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# pi/8 (T) gate on 1-qubit


def generate_gate_piover8_hamiltonian_vec() -> np.ndarray:
    """Return the vector representation for the Hamiltonian of a pi/8 (T) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.125 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[0] = -coeff
    vec[3] = coeff
    return vec


def generate_gate_piover8_hamiltonian_mat() -> np.ndarray:
    """Return Hamiltonian matrix for a pi/8 (T) gate.

    The result is the 2 times 2 complex matrix, :math:`-\\frac{\\pi}{8} I + \\frac{\\pi}{8} Z`.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    b = get_pauli_basis(num_qubit)
    coeff = 0.125 * math.pi
    mat = -coeff * b[0]
    mat += coeff * b[3]
    return mat


def generate_gate_piover8_effective_lindbladian_mat() -> np.ndarray:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a pi/8 (T) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real Hilbert-Schmidt representation matrix for the effective lindbladian of the gate.
    """
    l = [[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
    coeff = 0.25 * math.pi
    mat = coeff * np.array(l, dtype=np.float64)
    return mat


def generate_gate_piover8_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the pi/8 (T) gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_piover8_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# pi/8 daggered (T^dagger) gate on 1-qubit


def generate_gate_piover8_daggered_hamiltonian_vec() -> np.ndarray:
    """Return the vector representation for the Hamiltonian of a pi/8 daggered (T^dagger) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.125 * math.pi * np.sqrt(2)
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[0] = coeff
    vec[3] = -coeff
    return vec


def generate_gate_piover8_daggered_hamiltonian_mat() -> np.ndarray:
    """Return Hamiltonian matrix for a pi/8 daggerd (T^dagger) gate.

    The result is the 2 times 2 complex matrix, :math:`\\frac{\\pi}{8} I - \\frac{\\pi}{8} Z`.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    b = get_pauli_basis(num_qubit)
    coeff = 0.125 * math.pi
    mat = coeff * b[0]
    mat -= coeff * b[3]
    return mat


def generate_gate_piover8_daggered_effective_lindbladian_mat() -> np.ndarray:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of a pi/8 daggered (T^dagger) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real Hilbert-Schmidt representation matrix for the effective lindbladian of the gate.
    """
    l = [[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
    coeff = -0.25 * math.pi
    mat = coeff * np.array(l, dtype=np.float64)
    return mat


def generate_gate_piover8_daggered_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the pi/8 daggered (T^dagger) gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_piover8_daggered_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Hadamard (H) gate on 1-qubit


def generate_gate_hadamard_hamiltonian_vec() -> np.ndarray:
    """Return the vector representation for the Hamiltonian of an Hadamard (H) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a real vector with size 4.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real vector representation of the Hamiltonian of the gate.
    """
    dim = 2
    coeff = 0.5 * math.pi
    vec = np.zeros(dim * dim, dtype=np.float64)
    vec[0] = -coeff * np.sqrt(2)
    vec[1] = coeff
    vec[3] = coeff
    return vec


def generate_gate_hadamard_hamiltonian_mat() -> np.ndarray:
    """Return Hamiltonian matrix for an Hadamard (H) gate.

    The result is the 2 times 2 complex matrix, :math:`-0.25 \\pi I + 0.25 \\pi X / \\sqrt{2}+ 0.25 \\pi Z / \\sqrt{2}`.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The Hamiltonian, which is a complex matrix.
    """
    num_qubit = 1
    b = get_pauli_basis(num_qubit)
    coeff = 0.5 * math.pi
    mat = -coeff * b[0]
    mat += coeff * b[1] / np.sqrt(2)
    mat += coeff * b[3] / np.sqrt(2)
    return mat


def generate_gate_hadamard_effective_lindbladian_mat() -> np.ndarray:
    """Return the Hilbert-Schmidt representation matrix for the effective Lindbladian of an Hadamard (H) gate with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is a 4 times 4 real matrix.

    Parameters
    ----------

    Returns
    ----------
    np.ndarray
        The real Hilbert-Schmidt representation matrix for the effective lindbladian of the gate.
    """
    l = [[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, -1], [0, 0, 1, 0]]
    coeff = 0.50 * np.sqrt(2) * math.pi
    mat = coeff * np.array(l, dtype=np.float64)
    return mat


def generate_gate_hadamard_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the Hadamard (H) gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    assert len(c_sys.elemental_systems) == 1
    hs = generate_gate_hadamard_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# 1-qubit system
# HS matrix of effective Lindbladian for Hamiltonians in the form of a Pauli matrix


def generate_effective_lindbladian_mat_for_hamiltonian_x() -> np.ndarray:
    """Return HS matrix of effective lindbladian for Hamiltonian X, which correspond to a linear map, :math:`f(A) := -i [ H, A ]`, with :math:`H = X`."""
    l = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -2], [0, 0, 2, 0]]
    mat = np.array(l, dtype=np.float64)
    return mat


def generate_effective_lindbladian_mat_for_hamiltonian_y() -> np.ndarray:
    """Return HS matrix of effective lindbladian for Hamiltonian Y, which correspond to a linear map, :math:`f(A) := -i [ H, A ]`, with :math:`H = Y`."""
    l = [[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 0], [0, -2, 0, 0]]
    mat = np.array(l, dtype=np.float64)
    return mat


def generate_effective_lindbladian_mat_for_hamiltonian_z() -> np.ndarray:
    """Return HS matrix of effective lindbladian for Hamiltonian Z, which correspond to a linear map, :math:`f(A) := -i [ H, A ]`, with :math:`H = Z`."""
    l = [[0, 0, 0, 0], [0, 0, -2, 0], [0, 2, 0, 0], [0, 0, 0, 0]]
    mat = np.array(l, dtype=np.float64)
    return mat


# HS matrices for commutator map and anti-commutator maps with 1-qubit Pauli matrix

# Pauli commutator maps on 1-qubit
def calc_hs_commutator_map_i() -> np.ndarray:
    """Return the HS matrix for an Hermiticity-preserving linear map, :math:`f_H(A) := {H, A} = HA + AH`, with :math:`H = I`."""
    size = 4
    mat = 2 * np.eye(size, dtype=np.float64)
    return mat


def calc_hs_commutator_map_x() -> np.ndarray:
    """Return the HS matrix for an Hermiticity-preserving linear map, :math:`f_H(A) := {H, A} = HA + AH`, with :math:`H = X`."""
    l = [[0, 2, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    mat = np.array(l, dtype=np.float64)
    return mat


def calc_hs_commutator_map_y() -> np.ndarray:
    """Return the HS matrix for an Hermiticity-preserving linear map, :math:`f_H(A) := {H, A} = HA + AH`, with :math:`H = Y`."""
    l = [[0, 0, 2, 0], [0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]]
    mat = np.array(l, dtype=np.float64)
    return mat


def calc_hs_commutator_map_z() -> np.ndarray:
    """Return the HS matrix for an Hermiticity-preserving linear map, :math:`f_H(A) := {H, A} = HA + AH`, with :math:`H = Z`."""
    l = [[0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]]
    mat = np.array(l, dtype=np.float64)
    return mat


# Pauli annti-commutator map on 1-qubit


def calc_hs_minus1j_anticommutator_map_i() -> np.ndarray:
    """Return the HS matrix for an Hermiticity-preserving linear map, :math:`f_H(A) := -i[H, A] = -i(HA - AH)`, with :math:`H = I`."""
    size = 4
    mat = np.zeros((size, size), dtype=np.float64)
    return mat


def calc_hs_minus1j_anticommutator_map_x() -> np.ndarray:
    """Return the HS matrix for an Hermiticity-preserving linear map, :math:`f_H(A) := -i[H, A] = -i(HA - AH)`, with :math:`H = X`."""
    l = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -2], [0, 0, 2, 0]]
    mat = np.array(l, dtype=np.float64)
    return mat


def calc_hs_minus1j_anticommutator_map_y() -> np.ndarray:
    """Return the HS matrix for an Hermiticity-preserving linear map, :math:`f_H(A) := -i[H, A] = -i(HA - AH)`, with :math:`H = Y`."""
    l = [[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 0], [0, -2, 0, 0]]
    mat = np.array(l, dtype=np.float64)
    return mat


def calc_hs_minus1j_anticommutator_map_z() -> np.ndarray:
    """Return the HS matrix for an Hermiticity-preserving linear map, :math:`f_H(A) := -i[H, A] = -i(HA - AH)`, with :math:`H = Z`."""
    l = [[0, 0, 0, 0], [0, 0, -2, 0], [0, 2, 0, 0], [0, 0, 0, 0]]
    mat = np.array(l, dtype=np.float64)
    return mat


# 2-qubit system
# HS matrix of effective Lindbladian for Hamiltonians in the form of Pauli tensor product


def calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(
    pauli_type: str,
) -> np.ndarray:
    """Return the HS matrix of effective lindbladian for Hamiltonian with the form of tensor product of two Pauli matrices"""
    assert len(pauli_type) == 2

    pauli_type_0 = pauli_type[0]
    pauli_type_1 = pauli_type[1]
    pauli_types = ["i", "x", "y", "z"]
    assert pauli_type_0 in pauli_types
    assert pauli_type_1 in pauli_types

    mat_anti_0 = eval("calc_hs_minus1j_anticommutator_map_" + pauli_type_0)()
    mat_anti_1 = eval("calc_hs_minus1j_anticommutator_map_" + pauli_type_1)()
    mat_comm_0 = eval("calc_hs_commutator_map_" + pauli_type_0)()
    mat_comm_1 = eval("calc_hs_commutator_map_" + pauli_type_1)()

    size = 16
    mat = np.zeros((size, size), np.float64)
    mat += 0.50 * np.kron(mat_anti_0, mat_comm_1)
    mat += 0.50 * np.kron(mat_comm_0, mat_anti_1)
    return mat


# Control-X gate on 2-qubit


def generate_gate_cx_hamiltonian_vec(ids: List[int]) -> np.ndarray:
    """Return the vector representation of the Hamiltonian of the Control-X gate. The Hamiltonian is :math:`H = \\frac{\\pi}{4} (- II + IX - ZI - ZX)` for ids[0] < ids[1], and :math:`H = \\frac{\\pi}{4} (- II + XI - IZ - XZ)` for ids[0] > ids[1], where ids[0] for control system index and ids[1] for target system index."""
    assert len(ids) == 2
    assert ids[0] != ids[1]
    coeff = 0.5 * math.pi
    # 0.5 = 2 /4 where 2 is the normalization factor of the matrix basis
    size = 16
    vec = np.zeros(size, dtype=np.float64)
    if ids[0] < ids[1]:
        # II
        i = int("00", 4)
        vec[i] = -coeff
        # IX
        i = int("01", 4)
        vec[i] = coeff
        # ZI
        i = int("30", 4)
        vec[i] = coeff
        # ZX
        i = int("31", 4)
        vec[i] = -coeff
    else:
        # II
        i = int("00", 4)
        vec[i] = -coeff
        # XI
        i = int("10", 4)
        vec[i] = coeff
        # IZ
        i = int("03", 4)
        vec[i] = coeff
        # XZ
        i = int("13", 4)
        vec[i] = -coeff

    return vec


def generate_gate_cx_hamiltonian_mat(ids: List[int]) -> np.ndarray:
    """Return the Hamiltonian of the Control-X gate. The Hamiltonian is :math:`H = \\frac{\\pi}{4} (- II + IX - ZI - ZX)` for ids[0] < ids[1], and :math:`H = \\frac{\\pi}{4} (- II + XI - IZ - XZ)` for ids[0] > ids[1], where ids[0] for control system index and ids[1] for target system index."""
    assert len(ids) == 2
    assert ids[0] != ids[1]
    coeff = 0.25 * math.pi
    num_qubit = 2
    b = get_pauli_basis(num_qubit)

    size = 4
    mat = np.zeros((size, size), dtype=np.complex128)
    if ids[0] < ids[1]:
        # II
        i = int("00", 4)
        mat += -coeff * b[i]
        # IX
        i = int("01", 4)
        mat += coeff * b[i]
        # ZI
        i = int("30", 4)
        mat += coeff * b[i]
        # ZX
        i = int("31", 4)
        mat += -coeff * b[i]
    else:
        # II
        i = int("00", 4)
        mat += -coeff * b[i]
        # XI
        i = int("10", 4)
        mat += coeff * b[i]
        # IZ
        i = int("03", 4)
        mat += coeff * b[i]
        # XZ
        i = int("13", 4)
        mat += -coeff * b[i]

    return mat


def generate_gate_cx_effective_lindbladian_mat(ids: List[int]) -> np.ndarray:
    """Return the HS matrix of the effective lindbladian for a Control-X gate"""
    assert len(ids) == 2
    assert ids[0] != ids[1]
    coeff = 0.25 * math.pi

    size = 16
    mat = np.zeros((size, size), dtype=np.float64)
    if ids[0] < ids[1]:
        # II
        pauli_type = "ii"
        m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
        mat += -coeff * m
        # IX
        pauli_type = "ix"
        m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
        mat += coeff * m
        # ZI
        pauli_type = "zi"
        m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
        mat += coeff * m
        # ZX
        pauli_type = "zx"
        m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
        mat += -coeff * m
    else:
        # II
        pauli_type = "ii"
        m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
        mat += -coeff * m
        # XI
        pauli_type = "xi"
        m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
        mat += coeff * m
        # IZ
        pauli_type = "iz"
        m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
        mat += coeff * m
        # XZ
        pauli_type = "xz"
        m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
        mat += -coeff * m

    return mat


def generate_gate_cx_effective_lindbladian(
    c_sys: "CompositeSystem", ids: List[int]
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the Control-X gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    assert len(c_sys.elemental_systems) == 2
    hs = generate_gate_cx_effective_lindbladian_mat(ids)
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# Control-Z gate on 2-qubit


def generate_gate_cz_hamiltonian_vec() -> np.ndarray:
    """Return the vector representation of the Hamiltonian of the Control-Z gate. The Hamiltonian is :math:`H = \\frac{\\pi}{4} (- II + IZ + ZI - ZZ)`."""
    coeff = 0.5 * math.pi
    # 0.5 = 2 /4 where 2 is the normalization factor of the matrix basis
    size = 16
    vec = np.zeros(size, dtype=np.float64)
    # II
    i = int("00", 4)
    vec[i] = -coeff
    # IZ
    i = int("03", 4)
    vec[i] = coeff
    # ZI
    i = int("30", 4)
    vec[i] = coeff
    # ZZ
    i = int("33", 4)
    vec[i] = -coeff

    return vec


def generate_gate_cz_hamiltonian_mat() -> np.ndarray:
    """Return the Hamiltonian of the Control-Z gate. The Hamiltonian is :math:`H = \\frac{\\pi}{4} (- II + IZ + ZI - ZZ)`."""
    coeff = 0.25 * math.pi
    num_qubit = 2
    b = get_pauli_basis(num_qubit)

    size = 4
    mat = np.zeros((size, size), dtype=np.complex128)
    # II
    i = int("00", 4)
    mat += -coeff * b[i]
    # IZ
    i = int("03", 4)
    mat += coeff * b[i]
    # ZI
    i = int("30", 4)
    mat += coeff * b[i]
    # ZZ
    i = int("33", 4)
    mat += -coeff * b[i]

    return mat


def generate_gate_cz_effective_lindbladian_mat() -> np.ndarray:
    """Return the HS matrix of the effective lindbladian for a Control-Z gate"""
    coeff = 0.25 * math.pi
    size = 16
    mat = np.zeros((size, size), dtype=np.float64)

    # II
    pauli_type = "ii"
    m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
    mat += -coeff * m
    # IZ
    pauli_type = "iz"
    m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
    mat += coeff * m
    # ZI
    pauli_type = "zi"
    m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
    mat += coeff * m
    # ZZ
    pauli_type = "zz"
    m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
    mat += -coeff * m

    return mat


def generate_gate_cz_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the Control-Z gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    assert len(c_sys.elemental_systems) == 2
    hs = generate_gate_cz_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# SWAP gate on 2-qubit


def generate_gate_swap_hamiltonian_vec() -> np.ndarray:
    """Return the vector representation of the Hamiltonian of the SWAP gate. The Hamiltonian is :math:`H = \\frac{\\pi}{4} (- II + XX + YY + ZZ)`."""
    coeff = 0.5 * math.pi
    # 0.5 = 2 /4 where 2 is the normalization factor of the matrix basis
    size = 16
    vec = np.zeros(size, dtype=np.float64)
    # II
    i = int("00", 4)
    vec[i] = -coeff
    # XX
    i = int("11", 4)
    vec[i] = coeff
    # YY
    i = int("22", 4)
    vec[i] = coeff
    # ZZ
    i = int("33", 4)
    vec[i] = coeff

    return vec


def generate_gate_swap_hamiltonian_mat() -> np.ndarray:
    """Return the Hamiltonian of the SWAP gate. The Hamiltonian is :math:`H = \\frac{\\pi}{4} (- II + XX + YY + ZZ)`."""
    coeff = 0.25 * math.pi
    num_qubit = 2
    b = get_pauli_basis(num_qubit)

    size = 4
    mat = np.zeros((size, size), dtype=np.complex128)
    # II
    i = int("00", 4)
    mat += -coeff * b[i]
    # XX
    i = int("11", 4)
    mat += coeff * b[i]
    # YY
    i = int("22", 4)
    mat += coeff * b[i]
    # ZZ
    i = int("33", 4)
    mat += coeff * b[i]

    return mat


def generate_gate_swap_effective_lindbladian_mat() -> np.ndarray:
    """Return the HS matrix of the effective lindbladian for a SWAP gate"""
    coeff = 0.25 * math.pi
    size = 16
    mat = np.zeros((size, size), dtype=np.float64)

    # II
    pauli_type = "ii"
    m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
    mat += -coeff * m
    # XX
    pauli_type = "xx"
    m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
    mat += coeff * m
    # YY
    pauli_type = "yy"
    m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
    mat += coeff * m
    # ZZ
    pauli_type = "zz"
    m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
    mat += coeff * m

    return mat


def generate_gate_swap_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the SWAP gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    assert len(c_sys.elemental_systems) == 2
    hs = generate_gate_swap_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# ZX90 gate on 2-qubit system


def generate_gate_zx90_hamiltonian_vec(ids: List[int]) -> np.ndarray:
    """Return the vector representation of the Hamiltonian of the ZX90 gate. The Hamiltonian is :math:`H = \\frac{\\pi}{4} ZX` for ids[0] < ids[1], and :math:`H = \\frac{\\pi}{4} XZ` for ids[0] > ids[1], where ids[0] for control system index and ids[1] for target system index."""
    assert len(ids) == 2
    assert ids[0] != ids[1]
    coeff = 0.5 * math.pi
    # 0.5 = 2 /4 where 2 is the normalization factor of the matrix basis
    size = 16
    vec = np.zeros(size, dtype=np.float64)
    if ids[0] < ids[1]:
        # ZX
        i = int("31", 4)
        vec[i] = coeff
    else:
        # XZ
        i = int("13", 4)
        vec[i] = coeff

    return vec


def generate_gate_zx90_hamiltonian_mat(ids: List[int]) -> np.ndarray:
    """Return the Hamiltonian of the ZX90 gate. The Hamiltonian is :math:`H = \\frac{\\pi}{4} ZX` for ids[0] < ids[1], and :math:`H = \\frac{\\pi}{4} XZ` for ids[0] > ids[1], where ids[0] for control system index and ids[1] for target system index."""
    assert len(ids) == 2
    assert ids[0] != ids[1]
    coeff = 0.25 * math.pi
    num_qubit = 2
    b = get_pauli_basis(num_qubit)

    size = 4
    mat = np.zeros((size, size), dtype=np.complex128)
    if ids[0] < ids[1]:
        # ZX
        i = int("31", 4)
        mat += coeff * b[i]
    else:
        # XZ
        i = int("13", 4)
        mat += coeff * b[i]

    return mat


def generate_gate_zx90_effective_lindbladian_mat(ids: List[int]) -> np.ndarray:
    """Return the HS matrix of the effective lindbladian for a ZX90 gate"""
    assert len(ids) == 2
    assert ids[0] != ids[1]
    coeff = 0.25 * math.pi

    size = 16
    mat = np.zeros((size, size), dtype=np.float64)
    if ids[0] < ids[1]:
        # ZX
        pauli_type = "zx"
        m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
        mat += coeff * m
    else:
        # XZ
        pauli_type = "xz"
        m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
        mat += coeff * m

    return mat


def generate_gate_zx90_effective_lindbladian(
    c_sys: "CompositeSystem", ids: List[int]
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the ZX90 gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    ids : List[int]
        ids[0] for control system id, and ids[1] for target system id

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    assert len(c_sys.elemental_systems) == 2
    hs = generate_gate_zx90_effective_lindbladian_mat(ids)
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# ZZ90 gate on 2-qubit system


def generate_gate_zz90_hamiltonian_vec() -> np.ndarray:
    """Return the vector representation of the Hamiltonian of a ZZ90 gate. The Hamiltonian is :math:`H = \\frac{\\pi}{4} ZZ`."""
    coeff = 0.5 * math.pi
    # 0.5 = 2 /4 where 2 is the normalization factor of the matrix basis
    size = 16
    vec = np.zeros(size, dtype=np.float64)
    # ZZ
    i = int("33", 4)
    vec[i] = coeff

    return vec


def generate_gate_zz90_hamiltonian_mat() -> np.ndarray:
    """Return the Hamiltonian of a ZZ90 gate. The Hamiltonian is :math:`H = \\frac{\\pi}{4} ZZ`."""
    coeff = 0.25 * math.pi
    num_qubit = 2
    b = get_pauli_basis(num_qubit)

    size = 4
    mat = np.zeros((size, size), dtype=np.complex128)
    # ZZ
    i = int("33", 4)
    mat += coeff * b[i]

    return mat


def generate_gate_zz90_effective_lindbladian_mat() -> np.ndarray:
    """Return the HS matrix of the effective lindbladian for a ZZ90 gate"""
    coeff = 0.25 * math.pi
    size = 16
    mat = np.zeros((size, size), dtype=np.float64)

    # ZZ
    pauli_type = "zz"
    m = calc_effective_lindbladian_mat_for_2qubit_hamiltonian_pauli(pauli_type)
    mat += coeff * m

    return mat


def generate_gate_zz90_effective_lindbladian(
    c_sys: "CompositeSystem",
) -> "EffectiveLindbladian":
    """Return the class EffectiveLindbladian for the ZZ90 gate on the composite system.

    Parameters
    ----------
    c_sys : CompositeSystem
        The class CompositeSystem on which the gate acts.

    Returns
    ----------
    EffectiveLindbladian
        The effective Lindbladian of the gate.
    """
    assert len(c_sys.elemental_systems) == 2
    hs = generate_gate_zz90_effective_lindbladian_mat()
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# 3-qubit gates


# 1-qutrit gates


def generate_gate_1qutrit_single_gellmann_hamiltonian_vec(gate_name: str) -> np.ndarray:
    """return the Hamiltonian vector for the gate."""
    h = generate_gate_1qutrit_single_gellmann_hamiltonian_mat(gate_name)
    basis = get_normalized_gell_mann_basis()
    vec = calc_hermitian_matrix_expansion_coefficient_hermitian_basis(h, basis)
    return vec


def generate_gate_1qutrit_single_gellmann_effective_lindbladian_mat(
    gate_name: str,
) -> np.ndarray:
    """return the effective Lindbladian matrix for the gate."""
    h = generate_gate_1qutrit_single_gellmann_hamiltonian_mat(gate_name)
    to_basis = get_normalized_gell_mann_basis()
    hs = calc_effective_lindbladian_mat_hermitian_basis_from_hamiltonian(h, to_basis)
    return hs


def generate_gate_1qutrit_single_gellmann_effective_linabladian(
    c_sys: CompositeSystem, gate_name: str
) -> np.ndarray:
    """return the EffectiveLindbladian for the gate."""
    assert len(c_sys.elemental_systems) == 1
    assert c_sys.dim == 3
    hs = generate_gate_1qutrit_single_gellmann_effective_lindbladian_mat(gate_name)
    el = EffectiveLindbladian(c_sys=c_sys, hs=hs)
    return el


# 2-qutrit gates
