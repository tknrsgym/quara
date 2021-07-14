from typing import Tuple, List
from quara.objects.state_ensemble import StateEnsemble
import numpy as np
from quara.objects.composite_system import CompositeSystem
from quara.objects.state import State
from quara.objects.state_ensemble import StateEnsemble
from quara.objects.prob_dist import ProbDist

from quara.objects.state_typical import get_state_names_1qubit, generate_state_from_name


def get_state_ensemble_names():
    names = get_state_names_1qubit()
    return names


def generate_state_ensemble_elements_from_name(
    state_ensemble_name: str, c_sys: CompositeSystem
) -> Tuple[List[State], np.ndarray]:
    if state_ensemble_name not in get_state_ensemble_names():
        message = f"mprocess_name is out of range."
        raise ValueError(message)

    typical_names = get_state_ensemble_names()
    if state_ensemble_name in typical_names:
        method_name = f"get_state_ensemble_{state_ensemble_name}_elements"
        method = eval(method_name)
        return method(c_sys)


def get_state_ensemble_z0_elements(c_sys: CompositeSystem):
    z0 = generate_state_from_name(state_name="z0", c_sys=c_sys)
    z1 = generate_state_from_name(state_name="z0", c_sys=c_sys)
    states = [z0, z1]
    prob_dist = [1, 0]
    return states, prob_dist


def generate_state_ensemble_from_name(c_sys, state_ensemble_name):
    # TODO:
    states, prob_dist = generate_state_ensemble_elements_from_name(
        state_ensemble_name, c_sys
    )
    state_ensemble = StateEnsemble(states=states, prob_dist=prob_dist)
    return state_ensemble


def generate_state_ensemble_object_from_state_ensemble_name_object_name(
    state_ensemble_name: str, object_name: str, c_sys: CompositeSystem = None
) -> StateEnsemble:
    expected_object_names = [
        "state_ensemble",
    ]

    if object_name not in expected_object_names:
        raise ValueError("object_name is out of range.")
    if object_name == "state_ensemble":
        return generate_state_ensemble_from_name(c_sys, state_ensemble_name)
