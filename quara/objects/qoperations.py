from abc import abstractmethod
from typing import List

import numpy as np
from quara.objects.state import State, convert_state_to_var
from quara.objects.gate import Gate, convert_gate_to_var
from quara.objects.povm import Povm, convert_povm_to_var


class SetListQOperation:
    def __init__(
        self, states: List[State], gates: List[Gate], povms: List[Povm]
    ) -> None:
        # Validation
        self._validate_type(states, State)
        self._validate_type(povms, Povm)
        self._validate_type(gates, Gate)

        # Set
        self._states: List[State] = states
        self._povms: List[Povm] = povms
        self._gates: List[Gate] = gates
        # TODO: List[MProcess]
        self._mprocesses: list = []

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
        return self.states[index].to_var()

    def var_gate(self, index: int) -> np.array:
        return self.gates[index].to_var()

    def var_povm(self, index: int) -> np.array:
        return self.povms[index].to_var()

    def var_states(self) -> List[float]:
        vars = [state.to_var() for state in self.states]
        vars = np.hstack(vars)
        return vars

    def var_povms(self) -> np.array:
        vars = [povm.to_var() for povm in self.povms]
        vars = np.hstack(vars)
        return vars

    def var_gates(self) -> np.array:
        vars = [gate.to_var() for gate in self.gates]
        vars = np.hstack(vars)
        return vars

    def var_total(self) -> np.array:
        vars = np.stack([self.var_states(), self.var_gates(), self.var_povms()])
        return vars
