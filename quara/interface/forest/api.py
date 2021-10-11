from itertools import product
from typing import List, Tuple

import numpy as np
from numpy import pi
from qutip import Qobj, identity, sigmax, sigmay, sigmaz, tensor
from scipy.linalg import pinv
from pyquil import Program
from pyquil.gates import H, X, RX

from quara.interface.qutip.conversion import convert_povm_quara_to_qutip
from quara.objects.povm import Povm


def generate_program_for_1qubit(qubit: int, state_name: str) -> Program:
    """Generates a pyquil program that prepares the desired for 1 qubit.

    Parameters
    ----------
    qubit: int
        Index of the target qubit.
    state_name: str
        Name of the desired state which is z0, x0 or y0.

    Returns
    -------
    Program
        Program for state preparation for the specific qubit
    """
    if state_name == "z0":
        return Program()
    elif state_name == "z1":
        return Program(X(qubit))
    elif state_name == "x0":
        return Program(H(qubit))
    elif state_name == "x1":
        return Program(X(qubit), H(qubit))
    elif state_name == "y0":
        return Program(RX(-pi / 2, qubit))
    elif state_name == "y1":
        return Program(RX(pi / 2, qubit))
    else:
        raise ValueError("invalid state_name")


def generate_preprocess_program(qubits: List[int], state_name: str) -> Program:
    """Generates a pyquil program for the state preparation of the qubit system.

    Parameters
    ----------
    qubits: List[int]
        Index configuration of the qubit system.
    state_name: str
        Name of the desired state.

    Returs
    ------
    Program
        Program for the state preparation of the system
    """
    state_names_1qubit = state_name.split("_")
    assert len(qubits) == len(state_names_1qubit)
    pre_process_program = Program()
    for qubit, state_name_1qubit in zip(qubits, state_names_1qubit):
        pre_process_program += generate_program_for_1qubit(qubit, state_name_1qubit)
    return pre_process_program


def calc_empi_dist_from_observables(
    observables: List[float], num_shots: int, pauli_strings: List[str], povm: Povm
) -> Tuple[int, List[np.ndarray]]:
    """Calculates the empirical distribution of POVM from observables.

    Parameters
    ----------
    observable: List[float]
        List of observables measured by the Forest SDK.
    num_shots: int
        Number of measurement shots performed per observable.
    pauli_strings: List[str]
        List of strings that expresses a pauli operator which was used in the measurement.
    povm: Povm
        Quara's povm object which correspond to given pauli_strings.

    Returns
    -------
    Tuple[int, List[np.ndarray]]
        Empirical distribution that is compatible with Quara's tomography features.
    """
    coefficient_matrix = calc_coefficient_matrix(pauli_strings, povm)
    inv_mat, rank = pinv(coefficient_matrix, return_rank=True)
    if rank < coefficient_matrix.shape[0]:
        raise ValueError(
            "Given matrix is not full rank. Some experiments might be missing."
        )
    return (num_shots, inv_mat @ observables)


def generate_pauli_strings_from_povm_name(povm_name: str) -> List[str]:
    """Generates Pauli strings from given POVM name to construct a tomographical experiment.

    Parameters
    ----------
    povm_name: str
        Name of a POVM which is the target of the experiment.

    Returns
    -------
    List[str]
        List of Pauli strings for observation that covers sufficient information of the given POVM.
    """
    initial_string = "".join(povm_name.split("_")).upper()
    allowed_chars = set("XYZ")
    assert 0 < len(initial_string)
    assert set(initial_string) <= allowed_chars
    swap_times = 2 ** len(initial_string)
    pauli_strings = [initial_string]
    for i in range(1, swap_times):
        swap_position = f"{i:0{len(initial_string)}b}"
        pauli_str = ""
        for c, swap_flag in zip(initial_string, swap_position):
            if swap_flag == "1":
                pauli_str = pauli_str + "I"
            else:
                pauli_str = pauli_str + c
        pauli_strings.append(pauli_str)
    return pauli_strings


def calc_coefficient_matrix(pauli_strings: List[str], povm: Povm) -> np.ndarray:
    """Calculates coefficient matrix for calculatig empi dist from observables.

    Parameters
    ----------
    pauli_strings: List[str]
        List of pauli strings which are considered as operators of observables.
    povm: Povm
        POVM that is the target of generating probability distribution.

    Returns
    -------
    np.ndarray
        2d matrix that converts a list of observables to a probability distribution.
    """
    coefficient_mat = []
    for pauli_string in pauli_strings:
        coefficients = calc_coefficients(pauli_string, povm)
        coefficient_mat.append(coefficients)
    return np.array(coefficient_mat)


def calc_coefficients(pauli_string: str, povm: Povm) -> List[int]:
    """Calculates a list of coefficients that correspond observables to a probability of a POVM item.

    Parameters
    ----------
    pauli_string: str
        Pauli string which corresponds to an operator of the obtained expectation.
    povm: Povm
        POVM that is the targeto of generating probability distribution.

    Returns
    -------
    List[int]
        List of coefficients for calculating a probability from Pauli observables.
    """
    povm_items = convert_povm_quara_to_qutip(povm)
    order_candidates = [list(i) for i in product([1, -1], repeat=len(povm_items))]
    target_mat = generate_pauli_operator_from_pauli_string(pauli_string)
    dim = target_mat.dims[0][0]
    for order in order_candidates:
        observable = Qobj(dims=[[dim], [dim]])
        for sign, t in zip(order, povm_items):
            observable += sign * t
        if observable == target_mat:
            return order
    raise ValueError("Coefficient doesn't exists in given combination")


def generate_pauli_operator_from_pauli_string(pauli_string: str) -> Qobj:
    """Generates QuTip Qobj from given Pauli string in Forest SDK format.

    Parameters
    ----------
    pauli_string: str
        A Pauli string which is considered as operator of an observable.

    Returns
    -------
    Qobj
        QuTip Qobj that corresponds to the given Pauli string.
    """
    pauli_mat_list = []
    dim = 1
    for pauli_op in pauli_string:
        if pauli_op == "X":
            pauli_mat_list.append(sigmax())
        elif pauli_op == "Y":
            pauli_mat_list.append(sigmay())
        elif pauli_op == "Z":
            pauli_mat_list.append(sigmaz())
        elif pauli_op == "I":
            pauli_mat_list.append(identity(2))
        else:
            raise ValueError("Invalid character detected in pauli string.")
        dim = dim * 2
    return Qobj(tensor(pauli_mat_list), dims=[[dim], [dim]])
