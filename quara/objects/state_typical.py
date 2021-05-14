from itertools import product
from typing import List, Union

import numpy as np
from quara.objects.state import State
from quara.objects.matrix_basis import MatrixBasis, convert_vec
from quara.objects.matrix_basis import (
    get_normalized_pauli_basis,
    calc_hermitian_matrix_expansion_coefficient_hermitian_basis,
)
from quara.utils.matrix_util import calc_mat_from_vector_adjoint
from quara.objects.composite_system import CompositeSystem


def get_state_names() -> List[str]:
    """Return the list of valid state names."""
    names = []
    names += get_state_names_1qubit()
    names += get_state_names_2qubit()
    names += get_state_names_3qubit()
    names += get_state_names_1qutrit()
    names += get_state_names_2qutrit()
    return names


def get_state_names_1qubit() -> List[str]:
    """Return the list of valid gate names of 1-qubit states."""
    names = [a + b for a, b in product("xyz", "01")] + ["a"]
    return names


def _get_state_names_2qubit_typical() -> List[str]:
    return ["bell_phi_plus", "bell_phi_minus", "bell_psi_plus", "bell_psi_minus"]


def get_state_names_2qubit() -> List[str]:
    """Return the list of valid gate names of 2-qubit states."""
    names = _get_state_names_2qubit_typical()
    names_1qubit = get_state_names_1qubit()
    names += ["_".join(t) for t in product(names_1qubit, repeat=2)]
    return names


def _get_state_names_3qubit_typical() -> List[str]:
    return ["ghz", "werner"]


def get_state_names_3qubit() -> List[str]:
    """Return the list of valid gate names of 3-qubit states."""
    names = _get_state_names_3qubit_typical()

    names_1qubit = get_state_names_1qubit()
    names += ["_".join(t) for t in product(names_1qubit, repeat=3)]
    return names


def _get_state_names_1qutrit_special_typical() -> List[str]:
    return ["0_1_2_superposition"]


def _get_state_names_1qutrit_typical() -> List[str]:
    level = ["01", "12", "02"]
    axis = ["x", "y", "z"]
    d = ["0", "1"]
    names = ["".join(t) for t in product(level, axis, d)]
    return names


def get_state_names_1qutrit() -> List[str]:
    """Return the list of valid gate names of 1-qutrit states."""
    names = _get_state_names_1qutrit_special_typical()
    names += _get_state_names_1qutrit_typical()
    return names


def _get_state_names_2qutrit_typical() -> List[str]:
    return ["00_11_22_superposition"]


def get_state_names_2qutrit() -> List[str]:
    """Return the list of valid gate names of 2-qubit states."""
    names = _get_state_names_2qutrit_typical()
    names_1qutrit = _get_state_names_1qutrit_typical()
    names += ["_".join(t) for t in product(names_1qutrit, repeat=2)]
    return names


def generate_state_object_from_state_name_object_name(
    state_name: str, object_name: str, c_sys: CompositeSystem = None
) -> Union[State, np.ndarray]:
    """[summary]

    Parameters
    ----------
    state_name : str
        Name of the state.
        See the 'state_name' argument of generate_state_pure_state_vector_from_name() for available names.
    object_name : str
        Name of the format of the object to generate.
        one of ("pure_state_vector" | "density_mat" | "density_matrix_vector" | "state")
    c_sys : CompositeSystem, optional
        Specify if object_name is "state" or "density_matrix_vector", by default None.

    Returns
    -------
    Union[State, np.ndarray]
        The state specified by state_name is returned in an object of the form specified by object_name.

    Raises
    ------
    ValueError
        object_name or state_name is out of range.
    """
    expected_object_names = [
        "pure_state_vector",
        "density_mat",
        "density_matrix_vector",
        "state",
    ]

    if object_name not in expected_object_names:
        raise ValueError("object_name is out of range.")
    if object_name == "state":
        return generate_state_from_name(c_sys, state_name)
    elif object_name == "density_matrix_vector":
        return generate_state_density_matrix_vector_from_name(c_sys.basis(), state_name)
    else:
        method_name = f"generate_state_{object_name}_from_name"
        method = eval(method_name)
        return method(state_name)


def generate_state_density_mat_from_name(state_name: str) -> np.ndarray:
    """Return the density matrix ( :math:`|\\rho\\rangle` ) of state specified by name.

    Parameters
    ----------
    state_name : str
        name of the state.
        See the 'state_name' argument of generate_state_pure_state_vector_from_name() for available names.

    Returns
    -------
    np.ndarray
        density matrix ( :math:`|\\rho\\rangle` )
    """

    if state_name in get_state_names():
        pure_state_vec = generate_state_pure_state_vector_from_name(state_name)
        density_mat = calc_mat_from_vector_adjoint(pure_state_vec)
        return density_mat
    # TODO: call get_state_(state_name)_density_matrix()
    raise NotImplementedError(f"state_name={state_name}")


def generate_state_density_matrix_vector_from_name(
    basis: MatrixBasis, state_name: str
) -> np.ndarray:
    """Return the density matrix vector ( :math:`|\\rho\\rangle\\rangle` ) of state specified by name.

    Parameters
    ----------
    basis : MatrixBasis
        basis
    state_name : str
        name of the state.
        See the 'state_name' argument of generate_state_pure_state_vector_from_name() for available names.

    Returns
    -------
    np.ndarray
        density matrix vector ( :math:`|\\rho\\rangle\\rangle` )
    """
    density_mat = generate_state_density_mat_from_name(state_name)
    vec = calc_hermitian_matrix_expansion_coefficient_hermitian_basis(
        density_mat, basis
    )
    return vec


def generate_state_from_name(c_sys: CompositeSystem, state_name: str) -> State:
    """Return the state object specified by name.

    Parameters
    ----------
    c_sys : CompositeSystem
        composite system.
    state_name : str
        name of the state.
        See the 'state_name' argument of generate_state_pure_state_vector_from_name() for available names.

    Returns
    -------
    State
        State object
    """

    vec = generate_state_density_matrix_vector_from_name(c_sys.basis(), state_name)
    state = State(vec=vec, c_sys=c_sys)
    return state


def generate_state_pure_state_vector_from_name(state_name: str) -> np.ndarray:
    """Return the pure state vector for state specified by name.
    Use get_state_names() to get a list of all available names.

    Parameters
    ----------
    state_name : str
        name of the state.

        - 1 qubit: "x0", "x1", "y0", "y1", "z0", "a"
        - 2 qubit: "bell_psi_plus", "bell_psi_minus", "bell_phi_minus", "bell_phi_plus", or tensor product of 1 qubit ("z0_z0", "z0_z1", etc).
        - 3 qubit: "ghz", "werner", or tensor product of 1 qubit ("z0_z0_z0", "z0_x0_y0", etc).
        - 1 qutrit: Specify a combination of level ("01" | "12" | "02"), axis ("x" | "y" | "z"), and d ("0", "1") and "v012".

        For example, "01x0" means level is "01", axis is "x", and d is "0".
        Use get_state_names_1qutrit() to get a list of available names.

        - 2 qutrit: tensor product of 1 qutrit ("01x0_01y0", "01x0_01x1", etc) and "v001122".

    Returns
    -------
    np.ndarray
        pure state vector

    Raises
    ------
    ValueError
        'state_name' is out of range.
    """
    if state_name not in get_state_names():
        message = f"state_name is out of range."
        raise ValueError(message)

    typical_names = (
        get_state_names_1qubit()
        + _get_state_names_3qubit_typical()
        + get_state_names_1qutrit()
        + _get_state_names_2qutrit_typical()
    )
    if state_name in typical_names:
        method_name = f"get_state_{state_name}_pure_state_vector"
        method = eval(method_name)
        return method()
    elif state_name in _get_state_names_2qubit_typical():
        return get_state_bell_pure_state_vector(state_name)

    return _generate_pure_state_vec_tensor_product(state_name)


def _generate_pure_state_vec_tensor_product(state_name: str) -> np.ndarray:
    name_items = state_name.split("_")
    state_1qubit_list = []
    for i, name_item in enumerate(name_items):
        method_name = f"get_state_{name_item}_pure_state_vector"
        method = eval(method_name)
        pure_state_vec = method()
        state_1qubit_list.append(pure_state_vec)
    pure_state_vec = tensor_product_for_vecs(state_1qubit_list)
    return pure_state_vec


def tensor_product_for_vecs(state_vecs: np.ndarray) -> np.ndarray:
    state_vec = state_vecs[0]
    for vec in state_vecs[1:]:
        state_vec = np.kron(state_vec, vec)
    return state_vec


def get_state_x0_pure_state_vector() -> np.ndarray:
    """Returns the pure state vector for :math:`|+\\rangle`.

    :math:`|+\\rangle := \\frac{1}{\\sqrt{2}} (|0\\rangle + |1\\rangle)`

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = (1 / np.sqrt(2)) * np.array([1, 1], dtype=np.complex128)
    return vec


def get_state_x1_pure_state_vector() -> np.ndarray:
    """Returns the pure state vector for :math:`|-\\rangle`.

    :math:`|-\\rangle := \\frac{1}{\\sqrt{2}} (|0\\rangle - |1\\rangle)`

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = (1 / np.sqrt(2)) * np.array([1, -1], dtype=np.complex128)
    return vec


def get_state_y0_pure_state_vector() -> np.ndarray:
    """Returns the pure state vector for :math:`|i\\rangle`.

    :math:`|i\\rangle := \\frac{1}{\\sqrt{2}} (|0\\rangle + i|1\\rangle)`

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = (1 / np.sqrt(2)) * np.array([1, 1j], dtype=np.complex128)
    return vec


def get_state_y1_pure_state_vector() -> np.ndarray:
    """Returns the pure state vector for :math:`|i\\rangle`.

    :math:`|-i\\rangle := \\frac{1}{\\sqrt{2}} (|0\\rangle - i|1\\rangle)`

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = (1 / np.sqrt(2)) * np.array([1, -1j], dtype=np.complex128)
    return vec


def get_state_z0_pure_state_vector() -> np.ndarray:
    """Returns the pure state vector for :math:`|0\\rangle`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([1, 0], dtype=np.complex128)
    return vec


def get_state_z1_pure_state_vector() -> np.ndarray:
    """Returns the pure state vector for :math:`|1\\rangle`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([0, 1], dtype=np.complex128)
    return vec


def get_state_a_pure_state_vector() -> np.ndarray:
    """Return the pure state vector for A state.

    :math:`|A\\rangle := \\frac{1}{\\sqrt{2}} (|0\\rangle + \\exp(iπ/4)|1\\rangle)`

    Returns
    -------
    np.ndarray
        the pure state vector for A state.
    """
    vec = (1 / np.sqrt(2)) * np.array([1, np.exp(1j * np.pi / 4)], dtype=np.complex128)
    return vec


def get_state_bell_pure_state_vector(name: str) -> np.ndarray:
    """Return the pure state vector for bell.

    Parameters
    ----------
    name : str
        type of bell, one of ("bell_psi_plus" | "bell_psi_minus" | "bell_phi_plus" | "bell_phi_minus")

        - "bell_psi_plus": :math:`|\\Psi^+\\rangle := |0\\rangle|1\\rangle + |1\\rangle|0\\rangle`
        - "bell_psi_minus": :math:`|\\Psi^-\\rangle := |0\\rangle|1\\rangle - |1\\rangle|0\\rangle`
        - "bell_phi_plus": :math:`|\\Phi^+\\rangle := |0\\rangle|0\\rangle + |1\\rangle|1\\rangle`
        - "bell_phi_minus": :math:`|\\Phi^-\\rangle := |0\\rangle|0\\rangle - |1\\rangle|1\\rangle`

    Returns
    -------
    np.ndarray
        the pure state vector for bell state.

    Raises
    ------
    ValueError
        'name' is out of range.
    """
    if name == "bell_phi_plus":
        vec = (1 / np.sqrt(2)) * np.array([1, 0, 0, 1], dtype=np.complex128)
    elif name == "bell_phi_minus":
        vec = (1 / np.sqrt(2)) * np.array([1, 0, 0, -1], dtype=np.complex128)
    elif name == "bell_psi_plus":
        vec = (1 / np.sqrt(2)) * np.array([0, 1, 1, 0], dtype=np.complex128)
    elif name == "bell_psi_minus":
        vec = (1 / np.sqrt(2)) * np.array([0, 1, -1, 0], dtype=np.complex128)
    else:
        error_message = f"'name' is out of range."
        raise ValueError(error_message)

    return vec


def get_state_ghz_pure_state_vector() -> np.ndarray:
    """Return the pure state vector for GHZ.
    :math:`|GHZ\\rangle := \\frac{1}{\\sqrt{2}} (|0\\rangle|0\\rangle|0\\rangle + |1\\rangle|1\\rangle|1\\rangle)`

    Returns
    -------
    np.ndarray
        the pure state vector for GHZ state.
    """
    state_vec_0 = np.array([1, 0], dtype=np.complex128)  # |0>
    state_vec_1 = np.array([0, 1], dtype=np.complex128)  # |1>

    # |0>|0>|0>
    vec_0 = tensor_product_for_vecs([state_vec_0] * 3)
    # |1>|1>|1>
    vec_1 = tensor_product_for_vecs([state_vec_1] * 3)
    pure_state_vec = 1 / np.sqrt(2) * (vec_0 + vec_1)
    return pure_state_vec


def get_state_werner_pure_state_vector() -> np.ndarray:
    """Return the pure state vector for Werner.
    :math:`|W\\rangle := \\frac{1}{\\sqrt{3}} (|0\\rangle|0\\rangle|1\\rangle + |0\\rangle|1\\rangle|0\\rangle + |1\\rangle|0\\rangle|0\\rangle)`

    Returns
    -------
    np.ndarray
        the pure state vector for Werner state.
    """
    state_vec_0 = np.array([1, 0], dtype=np.complex128)  # |0>
    state_vec_1 = np.array([0, 1], dtype=np.complex128)  # |1>

    # |0>|0>|1>
    vec_0 = tensor_product_for_vecs([state_vec_0, state_vec_0, state_vec_1])
    # |0>|1>|0>
    vec_1 = tensor_product_for_vecs([state_vec_0, state_vec_1, state_vec_0])
    # |1>|0>|0>
    vec_2 = tensor_product_for_vecs([state_vec_1, state_vec_0, state_vec_0])

    pure_state_vec = 1 / np.sqrt(3) * (vec_0 + vec_1 + vec_2)
    return pure_state_vec


def get_state_x0_1q(c_sys: CompositeSystem) -> np.ndarray:
    """returns vec of state ``X_0`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.ndarray
        vec of state.
    """
    # whether dim of CompositeSystem equals 2
    if c_sys.dim != 2:
        raise ValueError(
            f"dim of CompositeSystem must equals 2.  dim of CompositeSystem is {c_sys.dim}"
        )

    # convert "vec in Pauli basis" to "vec in the basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, 1, 0, 0], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis())
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_state_x1_1q(c_sys: CompositeSystem) -> np.ndarray:
    """returns vec of state ``X_1`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.ndarray
        vec of state.
    """
    # whether dim of CompositeSystem equals 2
    if c_sys.dim != 2:
        raise ValueError(
            f"dim of CompositeSystem must equals 2.  dim of CompositeSystem is {c_sys.dim}"
        )

    # convert "vec in Pauli basis" to "vec in the basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, -1, 0, 0], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis())
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_state_y0_1q(c_sys: CompositeSystem) -> np.ndarray:
    """returns vec of state ``Y_0`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.ndarray
        vec of state.
    """
    # whether dim of CompositeSystem equals 2
    if c_sys.dim != 2:
        raise ValueError(
            f"dim of CompositeSystem must equals 2.  dim of CompositeSystem is {c_sys.dim}"
        )

    # convert "vec in Pauli basis" to "vec in the basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, 0, 1, 0], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis())
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_state_y1_1q(c_sys: CompositeSystem) -> np.ndarray:
    """returns vec of state ``Y_1`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.ndarray
        vec of state.
    """
    # whether dim of CompositeSystem equals 2
    if c_sys.dim != 2:
        raise ValueError(
            f"dim of CompositeSystem must equals 2.  dim of CompositeSystem is {c_sys.dim}"
        )

    # convert "vec in Pauli basis" to "vec in the basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, 0, -1, 0], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis())
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_state_z0_1q(c_sys: CompositeSystem) -> np.ndarray:
    """returns vec of state ``Z_0`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.ndarray
        vec of state.
    """
    # whether dim of CompositeSystem equals 2
    if c_sys.dim != 2:
        raise ValueError(
            f"dim of CompositeSystem must equals 2.  dim of CompositeSystem is {c_sys.dim}"
        )

    # convert "vec in Pauli basis" to "vec in the basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis())
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_state_z1_1q(c_sys: CompositeSystem) -> np.ndarray:
    """returns vec of state ``Z_1`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.ndarray
        vec of state.
    """
    # whether dim of CompositeSystem equals 2
    if c_sys.dim != 2:
        raise ValueError(
            f"dim of CompositeSystem must equals 2.  dim of CompositeSystem is {c_sys.dim}"
        )

    # convert "vec in Pauli basis" to "vec in the basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis())
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_state_a_1q(c_sys: CompositeSystem) -> State:
    # whether dim of CompositeSystem equals 2
    if c_sys.dim != 2:
        raise ValueError(
            f"dim of CompositeSystem must equals 2.  dim of CompositeSystem is {c_sys.dim}"
        )

    # convert "vec in Pauli basis" to "vec in the basis of CompositeSystem"
    from_vec = np.array([1 / np.sqrt(2), 1 / 2, 1 / 2, 0], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis())
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_state_bell_2q(c_sys: CompositeSystem) -> State:
    """returns vec of Bell state, :math:`\\frac{1}{2} (|00\\rangle + |11\\rangle)(\\langle00| + \\langle11|)` , with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    State
        vec of state.
    """
    # whether dim of CompositeSystem equals 4
    if c_sys.dim != 4:
        raise ValueError(
            f"dim of CompositeSystem must equals 4.  dim of CompositeSystem is {c_sys.dim}"
        )

    # \frac{1}{2}(|00>+|11>)(<00|+<11|)
    # convert "vec in comp basis" to "vec in basis of CompositeSystem"
    from_vec = (
        np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], dtype=np.float64) / 2
    )
    to_vec = convert_vec(from_vec, c_sys.comp_basis(), c_sys.basis())
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


# pure statevector of 1-qutrit, axis=01


def get_state_01x0_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`\\frac{1}{\\sqrt{2}} (|0\\rangle + |1\\rangle)`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([1, 1, 0], dtype=np.complex128) / np.sqrt(2)
    return vec


def get_state_01x1_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`\\frac{1}{\\sqrt{2}} (|0\\rangle - |1\\rangle)`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([1, -1, 0], dtype=np.complex128) / np.sqrt(2)
    return vec


def get_state_01y0_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`\\frac{1}{\\sqrt{2}} (|0\\rangle + j|1\\rangle)`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([1, 1j, 0], dtype=np.complex128) / np.sqrt(2)
    return vec


def get_state_01y1_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`\\frac{1}{\\sqrt{2}} (|0\\rangle - j|1\\rangle)`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([1, -1j, 0], dtype=np.complex128) / np.sqrt(2)
    return vec


def get_state_01z0_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`|0\\rangle`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([1, 0, 0], dtype=np.complex128)
    return vec


def get_state_01z1_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`|1\\rangle`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([0, 1, 0], dtype=np.complex128)
    return vec


# pure statevector of 1-qutrit, axis=12


def get_state_12x0_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`\\frac{1}{\\sqrt{2}} (|1\\rangle + |2\\rangle)`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([0, 1, 1], dtype=np.complex128) / np.sqrt(2)
    return vec


def get_state_12x1_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`\\frac{1}{\\sqrt{2}} (|1\\rangle - |2\\rangle)`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([0, 1, -1], dtype=np.complex128) / np.sqrt(2)
    return vec


def get_state_12y0_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`\\frac{1}{\\sqrt{2}} (|1\\rangle + j|2\\rangle)`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([0, 1, 1j], dtype=np.complex128) / np.sqrt(2)
    return vec


def get_state_12y1_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`\\frac{1}{\\sqrt{2}} (|1\\rangle - j|2\\rangle)`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([0, 1, -1j], dtype=np.complex128) / np.sqrt(2)
    return vec


def get_state_12z0_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`|1\\rangle`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([0, 1, 0], dtype=np.complex128)
    return vec


def get_state_12z1_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`|2\\rangle`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([0, 0, 1], dtype=np.complex128)
    return vec


# pure statevector of 1-qutrit, axis=02


def get_state_02x0_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`\\frac{1}{\\sqrt{2}} (|0\\rangle + |2\\rangle)`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([1, 0, 1], dtype=np.complex128) / np.sqrt(2)
    return vec


def get_state_02x1_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`\\frac{1}{\\sqrt{2}} (|0\\rangle - |2\\rangle)`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([1, 0, -1], dtype=np.complex128) / np.sqrt(2)
    return vec


def get_state_02y0_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`\\frac{1}{\\sqrt{2}} (|0\\rangle + j|2\\rangle)`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([1, 0, 1j], dtype=np.complex128) / np.sqrt(2)
    return vec


def get_state_02y1_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`\\frac{1}{\\sqrt{2}} (|0\\rangle - j|2\\rangle)`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([1, 0, -1j], dtype=np.complex128) / np.sqrt(2)
    return vec


def get_state_02z0_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`|0\\rangle`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([1, 0, 0], dtype=np.complex128)
    return vec


def get_state_02z1_pure_state_vector() -> np.ndarray:
    """returns the pure state vector of :math:`|2\\rangle`.

    Returns
    -------
    np.ndarray
        the pure state vector.
    """
    vec = np.array([0, 0, 1], dtype=np.complex128)
    return vec


def get_state_0_1_2_superposition_pure_state_vector() -> np.ndarray:
    """Return the pure state vector for v012.
    :math:`|v012\\rangle := \\frac{1}{\\sqrt{3}} (|0\\rangle + |1\\rangle + |2\\rangle)`

    Returns
    -------
    np.ndarray
        the pure state vector for v012.
    """
    pure_state_vec = 1 / np.sqrt(3) * np.array([1, 1, 1], dtype=np.complex128)
    return pure_state_vec


def get_state_00_11_22_superposition_pure_state_vector() -> np.ndarray:
    """Return the pure state vector for v001122.
    :math:`|v001122\\rangle := \\frac{1}{\\sqrt{3}} (|00\\rangle + |11\\rangle + |22\\rangle)`

    Returns
    -------
    np.ndarray
        the pure state vector for v001122.
    """
    vec_01z0 = get_state_01z0_pure_state_vector()
    vec_12z0 = get_state_12z0_pure_state_vector()
    vec_02z1 = get_state_02z1_pure_state_vector()
    pure_state_vec = (
        1
        / np.sqrt(3)
        * (
            np.kron(vec_01z0, vec_01z0)
            + np.kron(vec_12z0, vec_12z0)
            + np.kron(vec_02z1, vec_02z1)
        )
    )
    return pure_state_vec
