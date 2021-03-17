from itertools import product
from typing import List

import numpy as np
from quara.objects.state import State
from quara.objects.matrix_basis import MatrixBasis, convert_vec
from quara.objects.matrix_basis import (
    get_comp_basis,
    get_pauli_basis,
    get_normalized_pauli_basis,
)
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


def generate_state_from_state_name(
    state_name: str, c_sys: CompositeSystem, ids: List[int] = None
) -> "Gate":
    ids = ids if ids else []

    # TODO: validation
    # is_valid_dims_ids()
    if state_name not in get_state_names():
        message = f"state_name is out of range."
        # TODO: 指定可能な名前一覧を表示する
        raise ValueError(message)

    # 1qubit
    if state_name in get_state_names_1qubit():
        method_name = f"generate_state_{state_name}"
        method = eval(method_name)
        return method(c_sys)
    elif state_name in get_state_names_2qubit():
        if state_name in _get_state_names_2qubit_typical():
            raise NotImplementedError()
        return generate_state_nqubit(state_name, c_sys)
    elif state_name in get_state_names_3qubit():
        if state_name in _get_state_names_3qubit_typical():
            raise NotImplementedError()
        return generate_state_nqubit(state_name, c_sys)
    elif state_name in get_state_names_1qtrit():
        raise NotImplementedError()
    elif state_name in get_state_names_2qtrit():
        raise NotImplementedError()


def generate_state_nqubit(state_name: str, c_sys: CompositeSystem) -> State:
    name_items = state_name.split("_")
    c_sys_list = [CompositeSystem([e_sys]) for e_sys in c_sys._elemental_systems]
    state_1qubit_list = []
    for i, name_item in enumerate(name_items):
        method_name = f"generate_state_{name_item}"
        method = eval(method_name)
        state = method(c_sys_list[i])
        state_1qubit_list.append(state)

    state = tensor_product(state_1qubit_list)
    return state


# 1qubit
def generate_state_x0(c_sys: CompositeSystem) -> State:
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


def generate_state_x1(c_sys: CompositeSystem) -> State:
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


def generate_state_y0(c_sys: CompositeSystem) -> State:
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


def generate_state_y1(c_sys: CompositeSystem) -> State:
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


def generate_state_z0(c_sys: CompositeSystem) -> State:
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


def generate_state_z1(c_sys: CompositeSystem) -> State:
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


def generate_state_a(c_sys: CompositeSystem) -> State:
    # whether dim of CompositeSystem equals 2
    if c_sys.dim != 2:
        raise ValueError(
            f"dim of CompositeSystem must equals 2.  dim of CompositeSystem is {c_sys.dim}"
        )

    # convert "vec in Pauli basis" to "vec in the basis of CompositeSystem"
    # from_vec = 1 / np.sqrt(2) * np.array([1, 1, 0, 0], dtype=np.float64)
    # from_basis = get_normalized_pauli_basis()
    # to_vec = convert_vec(from_vec, from_basis, c_sys.basis())
    # state = State(c_sys, to_vec.real.astype(np.float64))
    # return state
    raise NotImplementedError()

