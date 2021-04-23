import numpy as np
from typing import List, Union
from itertools import product

# Quara
from quara.objects.composite_system import CompositeSystem
from quara.objects.operators import compose_qoperations
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import (
    get_depolarizing_channel,
)
from quara.objects.operators import tensor_product
from quara.objects.state_typical import (
    get_state_names_1qubit,
    get_state_names_1qutrit,
)
from quara.objects.povm_typical import (
    get_povm_names_1qubit,
    get_povm_names_1qutrit,
)
from quara.objects.qoperation_typical import generate_qoperation

# States


def generate_tester_states_depolarized(
    c_sys: CompositeSystem,
    names: List[str],
    error_rates: Union[float, List[float]],
) -> List[State]:
    """returns a list of states corresponding to names of states on a common CompositeSystem affected by a depolarizing channel.

    Parameters
    ----------
    c_sys: CompositeSystem

    names: List[str]
        names of typical states

    error_rates: Union[float, List[float]]
        depolarizing error rate or list of error rates
        If it is float, all states are affected by a common depolarizing channel with the error rate.

    Returns
    -------
    List[State]
        list of states depolarized
    """
    if type(error_rates) is float:
        error_rate = error_rates
    elif type(error_rates) is list:
        assert len(names) == len(error_rates)
    else:
        raise ValueError(f"Type of error_rates is invalid.")

    states = generate_tester_states(c_sys=c_sys, names=names)

    states_depolarized = []
    for i, state in enumerate(states):
        if type(error_rates) is float:
            error_rate = error_rates
        elif type(error_rates) is list:
            error_rate = error_rates[i]
        dp = get_depolarizing_channel(p=error_rate, c_sys=c_sys)
        state_new = compose_qoperations(dp, state)
        states_depolarized.append(state_new)
    return states_depolarized


def generate_tester_states(c_sys: CompositeSystem, names: List[str]) -> List[State]:
    """returns a list of states corresponding to names of states on a common CompositeSystem.

    Parameters
    ----------
    c_sys: CompositeSystem

    names: List[str]
        names of typical states

    Returns
    -------
    List[State]
    """
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


# POVMs


def generate_tester_povms_depolarized(
    c_sys: CompositeSystem,
    names: List[str],
    error_rates: Union[float, List[float]],
) -> List[Povm]:
    """returns a list of POVMs corresponding to names of POVMs on a common CompositeSystem affected by a depolarizing channel.

    Parameters
    ----------
    c_sys: CompositeSystem

    names: List[str]
        names of typical povms

    error_rates: Union[float, List[float]]
        depolarizing error rate or list of error rates
        If it is float, all POVMs are affected by a common depolarizing channel with the error rate.

    Returns
    -------
    List[Povm]
        list of POVMs depolarized
    """
    if type(error_rates) is float:
        error_rate = error_rates
    elif type(error_rates) is list:
        assert len(names) == len(error_rates)
    else:
        raise ValueError(f"Type of error_rates is invalid.")

    povms = generate_tester_povms(c_sys=c_sys, names=names)

    povms_depolarized = []
    for i, povm in enumerate(povms):
        if type(error_rates) is float:
            error_rate = error_rates
        elif type(error_rates) is list:
            error_rate = error_rates[i]
        dp = get_depolarizing_channel(p=error_rate, c_sys=c_sys)
        povm_new = compose_qoperations(povm, dp)
        povms_depolarized.append(povm_new)
    return povms_depolarized


def generate_tester_povms(c_sys: CompositeSystem, names: List[str]) -> List[Povm]:
    """returns a list of POVMs corresponding to names of POVMs on a common CompositeSystem.

    Parameters
    ----------
    c_sys: CompositeSystem

    names: List[str]
        names of typical POVMs

    Returns
    -------
    List[POVM]
    """
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
    method = eval("generate_povms_1" + mode_sys)
    povms_0 = method(c_sys_0, names)

    povms = povms_0
    for i in range(1, num):
        e_sys = c_sys._elemental_systems[i]
        c_sys_i = CompositeSystem([e_sys])
        povms_i = method(c_sys_i, names)
        l = []
        for p in product(povms, povms_i):
            povmA = p[0]
            povmB = p[1]
            povm = tensor_product(povmA, povmB)
            l.append(povm)
        povms = l

    return povms


def generate_povms_1qubit(c_sys: CompositeSystem, names: List[str]) -> List[Povm]:
    """returns a list of POVMs on a common 1-qubit system.

    Parameters
    ----------
    c_sys: CompositeSystem
        1-qubit system

    names: List[str]
        list of 1-qubit POVM names

    Returns
    -------
    List[Povm]
    """
    assert c_sys.num_e_sys == 1
    assert c_sys.dim == 2
    names_1qubit = get_povm_names_1qubit()
    for name in names:
        assert name in names_1qubit

    mode_qo = "povm"
    povms = []
    for name in names:
        povm = generate_qoperation(mode=mode_qo, name=name, c_sys=c_sys)
        povms.append(povm)
    return povms


def generate_povms_1qutrit(c_sys: CompositeSystem, names: List[str]) -> List[Povm]:
    """returns a list of POVMs on a common 1-qutrit system.

    Parameters
    ----------
    c_sys: CompositeSystem
        1-qutrit system

    names: List[str]
        list of 1-qutrit POVM names

    Returns
    -------
    List[Povm]
    """
    assert c_sys.num_e_sys == 1
    assert c_sys.dim == 3
    names_1qutrit = get_povm_names_1qutrit()
    for name in names:
        assert name in names_1qutrit

    mode_qo = "povm"
    povms = []
    for name in names:
        povm = generate_qoperation(mode=mode_qo, name=name, c_sys=c_sys)
        povms.append(povm)
    return povms


# Gate