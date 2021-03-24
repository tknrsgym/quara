from itertools import product
from typing import List, Union

import numpy as np
from quara.objects.state import State
from quara.objects.matrix_basis import MatrixBasis, convert_vec
from quara.objects.matrix_basis import (
    get_comp_basis,
    get_pauli_basis,
    get_normalized_pauli_basis,
    calc_hermitian_matrix_expansion_coefficient_hermitian_basis,
)
from quara.utils.matrix_util import calc_mat_from_vector_adjoint
from quara.objects.operators import tensor_product
from quara.objects.composite_system import CompositeSystem


def get_state_names() -> List[str]:
    """Return the list of valid state names."""
    names = []
    names += get_state_names_1qubit()
    names += get_state_names_2qubit()
    names += get_state_names_3qubit()
    names += get_state_names_1qtrit()
    names += get_state_names_1qtrit()
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


def get_state_names_1qtrit() -> List[str]:
    """Return the list of valid gate names of 1-qtrit states."""
    level = ["01", "12", "02"]
    axis = "xyz"
    d = "01"
    names = ["".join(t) for t in product(level, axis, d)]

    return names


def get_state_names_2qtrit() -> List[str]:
    """Return the list of valid gate names of 1-qtrit states."""
    names_1qutrit = get_state_names_1qtrit()
    names = ["_".join(t) for t in product(names_1qutrit, repeat=2)]

    return names


def generate_state_object_from_state_name_object_name(
    state_name: str, object_name: str, c_sys: CompositeSystem = None
) -> Union[State, np.array]:
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


def generate_state_density_mat_from_name(state_name: str) -> np.array:
    if state_name in get_state_names():
        pure_state_vec = generate_state_pure_state_vector_from_name(state_name)
        density_mat = calc_mat_from_vector_adjoint(pure_state_vec)
        return density_mat
    # TODO: call get_state_(state_name)_density_matrix()
    raise NotImplementedError()


def generate_state_density_matrix_vector_from_name(
    basis: MatrixBasis, state_name: str
) -> np.array:
    density_mat = generate_state_density_mat_from_name(state_name)
    vec = calc_hermitian_matrix_expansion_coefficient_hermitian_basis(
        density_mat, basis
    )
    return vec


def generate_state_from_name(c_sys: CompositeSystem, state_name: str) -> State:
    vec = generate_state_density_matrix_vector_from_name(c_sys.basis(), state_name)
    state = State(vec=vec, c_sys=c_sys)
    return state


def generate_state_pure_state_vector_from_name(state_name: str) -> np.array:
    if state_name not in get_state_names():
        message = f"state_name is out of range."
        raise ValueError(message)

    if state_name in get_state_names_1qubit():
        method_name = f"get_state_{state_name}_pure_state_vec"
        method = eval(method_name)
        return method()
    elif state_name in get_state_names_1qtrit():
        raise NotImplementedError()
    elif state_name in _get_state_names_2qubit_typical():
        return get_state_bell_pure_state_vec(state_name)

    return _generate_pure_state_vec_tensor_product(state_name)


def _generate_pure_state_vec_tensor_product(state_name: str) -> np.array:
    name_items = state_name.split("_")
    state_1qubit_list = []
    for i, name_item in enumerate(name_items):
        method_name = f"get_state_{name_item}_pure_state_vec"
        method = eval(method_name)
        pure_state_vec = method()
        state_1qubit_list.append(pure_state_vec)
    pure_state_vec = tensor_product_for_vecs(state_1qubit_list)
    return pure_state_vec


def tensor_product_for_vecs(state_vecs: np.array) -> np.array:
    state_vec = state_vecs[0]
    for vec in state_vecs[1:]:
        state_vec = np.kron(state_vec, vec)
    return state_vec


def get_state_x0_pure_state_vec() -> np.array:
    vec_0 = np.array([1, 0])
    vec_1 = np.array([0, 1])
    vec = (1 / np.sqrt(2)) * (vec_0 + vec_1)
    return vec


def get_state_x1_pure_state_vec() -> np.array:
    vec_0 = np.array([1, 0])
    vec_1 = np.array([0, 1])
    vec = (1 / np.sqrt(2)) * (vec_0 - vec_1)
    return vec


def get_state_y0_pure_state_vec() -> np.array:
    vec_0 = np.array([1, 0])
    vec_1 = np.array([0, 1])
    vec = (1 / np.sqrt(2)) * (vec_0 + 1j * vec_1)
    return vec


def get_state_y1_pure_state_vec() -> np.array:
    vec_0 = np.array([1, 0])
    vec_1 = np.array([0, 1])
    vec = (1 / np.sqrt(2)) * (vec_0 - 1j * vec_1)
    return vec


def get_state_z0_pure_state_vec() -> np.array:
    vec = np.array([1, 0])
    return vec


def get_state_z1_pure_state_vec() -> np.array:
    vec = np.array([0, 1])
    return vec


def get_state_a_pure_state_vec() -> np.array:
    state_vec_0 = np.array([1, 0])
    state_vec_1 = np.array([0, 1])
    pure_state_vec = state_vec_0 + np.exp(1j * np.pi / 4) * state_vec_1
    pure_state_vec = (1 / np.sqrt(2)) * pure_state_vec
    return pure_state_vec


def get_state_bell_pure_state_vec(name: str) -> np.array:
    state_vec_0 = np.array([1, 0])
    state_vec_1 = np.array([0, 1])

    name_items = name.split("_")
    error_message = f"name is out of range."

    if name_items[1] == "phi":
        vecs_0 = [state_vec_0, state_vec_1]
        vecs_1 = [state_vec_1, state_vec_0]
    elif name_items[1] == "psi":
        vecs_0 = [state_vec_0, state_vec_0]
        vecs_1 = [state_vec_1, state_vec_1]
    else:
        raise ValueError(error_message)

    pure_state_vec = tensor_product_for_vecs(vecs_0)
    if name_items[2] == "plus":
        pure_state_vec += tensor_product_for_vecs(vecs_1)
    elif name_items[2] == "minus":
        pure_state_vec -= tensor_product_for_vecs(vecs_1)
    else:
        raise ValueError(error_message)
    pure_state_vec = 1 / np.sqrt(2) * pure_state_vec

    return pure_state_vec


def get_x0_1q(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``X_0`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.array
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


def get_x1_1q(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``X_1`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.array
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


def get_y0_1q(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``Y_0`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.array
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


def get_y1_1q(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``Y_1`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.array
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


def get_z0_1q(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``Z_0`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.array
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


def get_z1_1q(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``Z_1`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.array
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


def get_a_1q(c_sys: CompositeSystem) -> State:
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


def get_bell_2q(c_sys: CompositeSystem) -> State:
    """returns vec of Bell state, \frac{1}{2}(|00>+|11>)(<00|+<11|), with the basis of ``c_sys``.

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
