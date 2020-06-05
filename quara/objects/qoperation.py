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
        else:
            states_on_eq_const = [True] * len(states)
        if povms_on_eq_const:
            self._validate_type(povms_on_eq_const, bool, arg_name="povms_on_eq_const")
            self._validate_length(povms, povms_on_eq_const, "povms", "povm_on_eq_const")
        else:
            povms_on_eq_const = [True] * len(povms)

        if gates_on_eq_const:
            self._validate_type(gates_on_eq_const, bool, arg_name="gates_on_eq_const")
            self._validate_length(gates, gates_on_eq_const, "gates", "gate_on_eq_const")
        else:
            gates_on_eq_const = [True] * len(gates)

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

    def size_var_states(self) -> int:
        return len(self.var_states())

    def size_var_gates(self) -> int:
        return len(self.var_gates())

    def size_var_povms(self) -> int:
        return len(self.var_povms())

    def size_var_mprocesses(self) -> int:
        # TODO
        return 0

    def size_var_state(self, index: int) -> int:
        return len(self.var_state(index=index))

    def size_var_gate(self, index: int) -> int:
        return len(self.var_gate(index=index))

    def size_var_povm(self, index: int) -> int:
        return len(self.var_povm(index=index))

    def size_var_mprocess(self) -> int:
        # TODO
        pass

    def size_var_total(self, index: int) -> int:
        total = sum(
            [
                self.size_var_states(),
                self.size_var_gates(),
                self.size_var_povms(),
                self.size_var_mprocesses(),
            ]
        )
        return total

    def var_state(self, index: int) -> np.array:
        return self.states[index].to_var(self.states_on_eq_const[index])

    def var_gate(self, index: int) -> np.array:
        return self.gates[index].to_var(self.gates_on_eq_const[index])

    def var_povm(self, index: int) -> np.array:
        return self.povms[index].to_var(self.povms_on_eq_const[index])

    def var_states(self) -> List[float]:
        self._validate_length(
            self.states, self.states_on_eq_const, "state", "states_on_eq_const"
        )
        vars = [
            state.to_var(self.states_on_eq_const[i])
            for i, state in enumerate(self.states)
        ]
        vars = np.hstack(vars)
        return vars

    def var_povms(self) -> np.array:
        self._validate_length(
            self.povms, self.povms_on_eq_const, "povm", "povms_on_eq_const"
        )

        vars = [
            povm.to_var(self.povms_on_eq_const[i]) for i, povm in enumerate(self.povms)
        ]
        vars = np.hstack(vars)
        return vars

    def var_gates(self) -> np.array:
        self._validate_length(
            self.gates, self.gates_on_eq_const, "gate", "gates_on_eq_const"
        )

        vars = [
            gate.to_var(self.gates_on_eq_const[i]) for i, gate in enumerate(self.gates)
        ]
        vars = np.hstack(vars)
        return vars

    def var_total(self) -> np.array:
        vars = np.stack([self.var_states(), self.var_gates(), self.var_povms()])
        return vars
