from typing import List
from itertools import product

# Quara
from quara.objects.composite_system import CompositeSystem
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.operators import tensor_product
from quara.objects.state_typical import (
    get_state_names_1qubit,
    get_state_names_1qutrit,
)
from quara.objects.qoperation_typical import generate_qoperation


def generate_tester_states(c_sys: CompositeSystem, names: List[str]) -> List[State]:
    """"""
    # c_sys
    num = c_sys.num_e_sys
    dims = []
    for i in range(num):
        dims.append(c_sys.dim_e_sys(i))

    if dims[0] == 2:
        mode_sys = "qubit"
    elif dims[0] == 3:
        mode_sys = "qutrit"
    else:
        raise ValueError(f"system size is invalid!")

    e_sys = c_sys._elemental_systems[0]
    c_sys_0 = CompositeSystem([e_sys])
    method = eval("generate_states_1" + mode_sys)
    states_0 = method(c_sys_0, names)

    states = states_0
    for i in range(1, num):
        e_sys = c_sys._elemental_systems[i]
        c_sys_i = CompositeSystem([e_sys])
        states_i = method(c_sys_i, names)
        l = []
        for p in product(states, states_i):
            stateA = p[0]
            stateB = p[1]
            state = tensor_product(stateA, stateB)
            l.append(state)
        states = l

    return states


def generate_states_1qubit(c_sys: CompositeSystem, names: List[str]) -> List[State]:
    """returns a list of states on a common 1-qubit system.

    Parameters
    ----------
    c_sys: CompositeSystem
        1-qubit system

    names: List[str]
        list of 1-qubit state names

    Returns
    -------
    List[State]
    """
    assert c_sys.num_e_sys == 1
    assert c_sys.dim == 2
    names_1qubit = get_state_names_1qubit()
    for name in names:
        assert name in names_1qubit

    mode_qo = "state"
    states = []
    for name in names:
        state = generate_qoperation(mode=mode_qo, name=name, c_sys=c_sys)
        states.append(state)
    return states


def generate_states_1qutrit(c_sys: CompositeSystem, names: List[str]) -> List[State]:
    """returns a list of states on a common 1-qutrit system.

    Parameters
    ----------
    c_sys: CompositeSystem
        1-qutrit system

    names: List[str]
        list of 1-qutrit state names

    Returns
    -------
    List[State]
    """
    assert c_sys.num_e_sys == 1
    assert c_sys.dim == 3
    names_1qutrit = get_state_names_1qutrit()
    for name in names:
        assert name in names_1qutrit

    mode_qo = "state"
    states = []
    for name in names:
        state = generate_qoperation(mode=mode_qo, name=name, c_sys=c_sys)
        states.append(state)
    return states