import math
import numpy as np

from quara.objects.matrix_basis import (
    get_pauli_basis,
)
from quara.objects.effective_lindbladian import EffectiveLindbladian

# Identity gate


def generate_vec_identity_gate_hamiltonian(dim: int) -> np.array:
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


def generate_matrix_identity_gate_hamiltonian(dim: int) -> np.array:
    """Return Hamiltonian for an identity gate.

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


def generate_matrix_identity_gate_unitary(dim: int) -> np.array:
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


def generate_matrix_identity_gate_lindbladian(dim: int) -> np.array:
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


def generate_matrix_identity_gate(dim: int) -> np.array:
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


def generate_effective_lindbladian_identity_gate(
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
    dim = c_sys.dim()
    hs = generate_hsmatrix_identity_gate_lindbladian(dim)
    el = EffectiveLindbladian(c_sys, hs)
    return el


# X90 gate on 1-qubit


def generate_vec_x90_gate_hamiltonian() -> np.array:
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


def generate_matrix_x90_gate_hamiltonian() -> np.array:
    """Return Hamiltonian for an X90 gate.

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


def generate_matrix_x90_gate_unitary() -> np.array:
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


def generate_matrix_x90_gate_lindbladian() -> np.array:
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


def generate_matrix_x90_gate() -> np.array:
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


def generate_effective_lindbladian_x90_gate(
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
    hs = generate_hsmatrix_x90_gate_lindbladian()
    el = EffectiveLindbladian(c_sys, hs)
    return el


# X180 gate on 1-qubit


def generate_vec_x180_gate_hamiltonian() -> np.array:
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


def generate_matrix_x180_gate_hamiltonian() -> np.array:
    """Return Hamiltonian for an X180 gate.

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


def generate_matrix_x180_gate_unitary() -> np.array:
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


def generate_matrix_x180_gate_lindbladian() -> np.array:
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


def generate_matrix_x180_gate() -> np.array:
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


def generate_effective_lindbladian_x180_gate(
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
    hs = generate_hsmatrix_x180_gate_lindbladian()
    el = EffectiveLindbladian(c_sys, hs)
    return el


# X gate on 1-qubit


def generate_vec_x_gate_hamiltonian() -> np.array:
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


def generate_matrix_x_gate_hamiltonian() -> np.array:
    """Return Hamiltonian for an X gate.

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


def generate_matrix_x_gate_unitary() -> np.array:
    """Return the unitary matrix for an X gate.

    The result is the 2 times 2 complex matrix, X.

    Parameters
    ----------

    Returns
    ----------
    np.array
        The unitary matrix, which is a complex matrix.
    """
    u = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    return u


def generate_matrix_x_gate_lindbladian() -> np.array:
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


def generate_matrix_x_gate() -> np.array:
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


def generate_effective_lindbladian_x_gate(
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
    hs = generate_hsmatrix_x_gate_lindbladian()
    el = EffectiveLindbladian(c_sys, hs)
    return el


# Y90 on 1-qubit


def generate_vec_y90_gate_hamiltonian() -> np.array:
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


def generate_matrix_y90_gate_hamiltonian() -> np.array:
    """Return Hamiltonian for a Y90 gate.

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


def generate_matrix_y90_gate_unitary() -> np.array:
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


def generate_matrix_y90_gate_lindbladian() -> np.array:
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


def generate_matrix_y90_gate() -> np.array:
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


def generate_effective_lindbladian_y90_gate(
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
    hs = generate_hsmatrix_y90_gate_lindbladian()
    el = EffectiveLindbladian(c_sys, hs)
    return el


# Y180 gate on 1-qubit


def generate_vec_y180_gate_hamiltonian() -> np.array:
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


def generate_matrix_y180_gate_hamiltonian() -> np.array:
    """Return Hamiltonian for a Y180 gate.

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


def generate_matrix_y180_gate_unitary() -> np.array:
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


def generate_matrix_y180_gate_lindbladian() -> np.array:
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


def generate_matrix_y180_gate() -> np.array:
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


def generate_effective_lindbladian_y180_gate(
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
    hs = generate_hsmatrix_y180_gate_lindbladian()
    el = EffectiveLindbladian(c_sys, hs)
    return el


# Z90 gate on 1-qubit


def generate_vec_z90_gate_hamiltonian() -> np.array:
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


def generate_matrix_z90_gate_hamiltonian() -> np.array:
    """Return Hamiltonian for a Z90 gate.

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


def generate_matrix_z90_gate_unitary() -> np.array:
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


def generate_matrix_z90_gate_lindbladian() -> np.array:
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


def generate_matrix_z90_gate() -> np.array:
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


def generate_effective_lindbladian_z90_gate(
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
    hs = generate_hsmatrix_z90_gate_lindbladian()
    el = EffectiveLindbladian(c_sys, hs)
    return el


# Z180 gate on 1-qubit


def generate_vec_z180_gate_hamiltonian() -> np.array:
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


def generate_matrix_z180_gate_hamiltonian() -> np.array:
    """Return Hamiltonian for a Z180 gate.

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


def generate_matrix_z180_gate_unitary() -> np.array:
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


def generate_matrix_z180_gate_lindbladian() -> np.array:
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


def generate_matrix_z180_gate() -> np.array:
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


def generate_effective_lindbladian_z180_gate(
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
    hs = generate_hsmatrix_z180_gate_lindbladian()
    el = EffectiveLindbladian(c_sys, hs)
    return el
