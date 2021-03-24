from itertools import product
from typing import List

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


def generate_state_from_state_name(state_name: str, c_sys: CompositeSystem) -> "State":
    if state_name not in get_state_names():
        message = f"state_name is out of range."
        raise ValueError(message)

    # 1qubit
    if state_name in get_state_names_1qubit():
        method_name = f"generate_state_{state_name}"
        method = eval(method_name)
        return method(c_sys)
    elif state_name in get_state_names_1qtrit():
        raise NotImplementedError()
    elif state_name in _get_state_names_2qubit_typical():
        raise generate_bell(state_name)

    return _generate_state_tensor_product(state_name, c_sys)


def _generate_state_tensor_product(state_name: str, c_sys: CompositeSystem) -> State:
    name_items = state_name.split("_")
    c_sys_list = [CompositeSystem([e_sys]) for e_sys in c_sys._elemental_systems]
    state_1qubit_list = []
    for i, name_item in enumerate(name_items):
        # TODO: Stateオブジェクトを返す関数ではなく、純粋状態のベクトルを返す関数を呼ぶ形に変更する
        method_name = f"generate_state_{name_item}"
        method = eval(method_name)
        state = method(c_sys_list[i])
        state_1qubit_list.append(state)
    # TODO: tensor_productではなく、tensor_product_for_vecsを呼んで純粋状態のベクトル同士のテンソル積を取る
    state = tensor_product(state_1qubit_list)
    return state


def tensor_product_for_vecs(state_vecs: np.array) -> np.array:
    state_vec = state_vecs[0]
    for vec in state_vecs[1:]:
        state_vec = np.kron(state_vec, vec)
    return state_vec


def generate_bell(name: str) -> np.array:
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


def generate_state_from_density_matrix(
    density_matrix: np.array, c_sys: CompositeSystem
) -> State:
    # 密度行列からベクトルに変換する
    vec = calc_hermitian_matrix_expansion_coefficient_hermitian_basis(
        density_matrix, c_sys.basis()
    )
    state = State(vec=vec, c_sys=c_sys)
    return state


def generate_state_from_pure_state(
    pure_state_vec: np.array, c_sys: CompositeSystem
) -> State:
    # 純粋な状態ベクトルから密度行列に変換する
    density_matrix = calc_mat_from_vector_adjoint(pure_state_vec)
    state = generate_state_from_density_matrix(density_matrix, c_sys)
    return state


def generate_state_a(c_sys) -> State:
    state_vec_0 = np.array([1, 0])
    state_vec_1 = np.array([0, 1])
    pure_state_vec = state_vec_0 + np.exp(1j * np.pi / 4) * state_vec_1
    pure_state_vec = (1 / np.sqrt(2)) * pure_state_vec
    state_a = calc_mat_from_vector_adjoint(pure_state_vec)
    return state_a


# TODO: 以下はstate.pyにあって名前を変更したもの。
# 名前を元に戻して、状態ベクトルを返す関数を別に用意する。3/24
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

