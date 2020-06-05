from typing import List

import numpy as np
from quara.objects.state import State, convert_state_to_var
from quara.objects.gate import Gate, convert_gate_to_var
from quara.objects.povm import Povm, convert_povm_to_var


class SetListQOperation:
    def __init__(
        self,
        states: List[State],
        gates: List[Gate],
        povms: List[Povm],
        states_on_eq_const: List[bool] = None,
        gates_on_eq_const: List[bool] = None,
        povms_on_eq_const: List[bool] = None,
    ) -> None:
        # Validation
        self._validate_type(states, State)
        self._validate_type(povms, Povm)
        self._validate_type(gates, Gate)

        # TODO: add validation to check length
        if states_on_eq_const:
            self._validate_type(states_on_eq_const, bool, arg_name="states_on_eq_const")
            self._validate_length(
                states, states_on_eq_const, "states", "state_on_eq_const"
            )
        if povms_on_eq_const:
            self._validate_type(povms_on_eq_const, bool, arg_name="povms_on_eq_const")
            self._validate_length(povms, povms_on_eq_const, "povms", "povm_on_eq_const")
        if gates_on_eq_const:
            self._validate_type(gates_on_eq_const, bool, arg_name="gates_on_eq_const")
            self._validate_length(gates, gates_on_eq_const, "gates", "gate_on_eq_const")

        # Set
        self._states: List[State] = states
        self._povms: List[Povm] = povms
        self._gates: List[Gate] = gates
        # TODO: List[MProcess]
        self._mprocesses: list = []

        self._states_on_eq_const: List[bool] = states_on_eq_const
        self._povms_on_eq_const: List[bool] = povms_on_eq_const
        self._gates_on_eq_const: List[bool] = gates_on_eq_const

    def _validate_type(self, targets, expected_type, arg_name: str = None) -> None:
        for target in targets:
            if target and not isinstance(target, expected_type):
                arg_name = (
                    arg_name if arg_name else expected_type.__name__.lower() + "s"
                )
                error_message = "'{}' must be a list of {}.".format(
                    arg_name, expected_type.__name__
                )
                raise TypeError(error_message)

    def _validate_length(self, targets, compared, target_name, compared_name):
        if len(targets) != len(compared):
            error_message = "'{}' and '{}' must be the same length.".format(
                target_name, compared_name
            )
            raise ValueError(error_message)

    # Setter & Getter
    @property
    def states(self) -> List[State]:
        return self._states

    @states.setter
    def states(self, value):
        self._validate_type(value, State)
        self._states = value

    @property
    def povms(self) -> List[Povm]:
        return self._povms

    @povms.setter
    def povms(self, value):
        self._validate_type(value, Povm)
        self._povms = value

    @property
    def gates(self) -> List[Gate]:
        return self._gates

    @gates.setter
    def gates(self, value):
        self._validate_type(value, Gate)
        self._gates = value

    @property
    def states_on_eq_const(self) -> List[bool]:
        return self._states_on_eq_const

    @states_on_eq_const.setter
    def states_on_eq_const(self, value):
        self._validate_type(value, bool, "states_on_eq_const")
        self._states_on_eq_const = value

    @property
    def gates_on_eq_const(self) -> List[bool]:
        return self._gates_on_eq_const

    @gates_on_eq_const.setter
    def gates_on_eq_const(self, value):
        self._validate_type(value, bool, "gates_on_eq_const")
        self._gates_on_eq_const = value

    @property
    def povms_on_eq_const(self) -> List[bool]:
        return self._povms_on_eq_const

    @povms_on_eq_const.setter
    def povms_on_eq_const(self, value):
        self._validate_type(value, bool, "povms_on_eq_const")
        self._povms_on_eq_const = value

    def num_states(self):
        return len(self._states)

    def num_povms(self):
        return len(self._povms)

    def num_gates(self):
        return len(self._gates)

    def num_mprocesses(self):
        return len(self._mprocesses)

    def convert_state_to_var(self, index: int) -> np.array:
        on_eq_const = (
            self.states_on_eq_const if self.states_on_eq_const else [True] * len(states)
        )
        self._validate_length(self.povms, on_eq_const, "states", "states_on_eq_const")

        # TODO:
        # call function

    def convert_povm_to_var(self, index: int) -> np.array:
        pass

    def convert_povm_to_var(self, index: int) -> np.array:
        pass
