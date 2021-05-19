from abc import abstractmethod
from typing import List, Dict

import numpy as np
from quara.objects.qoperation import QOperation
from quara.objects.state import State, convert_var_to_state
from quara.objects.gate import Gate, convert_var_to_gate
from quara.objects.povm import Povm, convert_var_to_povm


class SetQOperations:
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

    def dim_state(self, index: int) -> int:
        # returns the dimension of the total system of the i-th state
        return self.states[index].dim

    def dim_gate(self, index: int) -> int:
        # returns the dimension of the total system of the i-th gate
        return self.gates[index].dim

    def dim_povm(self, index: int) -> int:
        # returns the dimension of the total system of the i-th povm
        return self.povms[index].dim

    def dim_mprosess(self, index: int) -> int:
        # TODO MProcess
        raise NotImplementedError()

    def size_var_states(self) -> int:
        return len(self.var_states())

    def size_var_gates(self) -> int:
        return len(self.var_gates())

    def size_var_povms(self) -> int:
        return len(self.var_povms())

    def size_var_mprocesses(self) -> int:
        # TODO MProcess
        return 0

    def size_var_state(self, index: int) -> int:
        return len(self.var_state(index=index))

    def size_var_gate(self, index: int) -> int:
        return len(self.var_gate(index=index))

    def size_var_povm(self, index: int) -> int:
        return len(self.var_povm(index=index))

    def size_var_mprocess(self) -> int:
        # TODO MProcess
        pass

    def size_var_total(self) -> int:
        total = sum(
            [
                self.size_var_states(),
                self.size_var_gates(),
                self.size_var_povms(),
                self.size_var_mprocesses(),
            ]
        )
        return total

    def var_state(self, index: int) -> np.ndarray:
        return self.states[index].to_var()

    def var_gate(self, index: int) -> np.ndarray:
        return self.gates[index].to_var()

    def var_povm(self, index: int) -> np.ndarray:
        return self.povms[index].to_var()

    def var_states(self) -> List[float]:
        vars = [state.to_var() for state in self.states]
        vars = np.hstack(vars) if vars else np.array([])
        return vars

    def var_povms(self) -> np.ndarray:
        vars = [povm.to_var() for povm in self.povms]
        vars = np.hstack(vars) if vars else np.array([])
        return vars

    def var_gates(self) -> np.ndarray:
        vars = [gate.to_var() for gate in self.gates]
        vars = np.hstack(vars) if vars else np.array([])
        return vars

    def var_total(self) -> np.ndarray:
        vars = np.hstack([self.var_states(), self.var_gates(), self.var_povms()])
        return vars

    def _get_operation_type_to_total_index_map(self) -> Dict[str, int]:
        states_first_index = 0
        gates_first_index = self.size_var_states()
        povms_first_index = gates_first_index + self.size_var_gates()
        mprocesses_first_index = povms_first_index + self.size_var_gates()
        return dict(
            state=states_first_index,
            gate=gates_first_index,
            povm=povms_first_index,
            mprocess=mprocesses_first_index,
        )

    def _get_operation_item_var_first_index(
        self, type_operation: str, index: int
    ) -> int:
        # returns the index that is the place of the 'index'-th 'type_operation' starts in the whole var
        target_operations: List[QOperation]
        if type_operation == "state":
            target_operations = self.states
            get_size_func = self.size_var_state
        elif type_operation == "gate":
            target_operations = self.gates
            get_size_func = self.size_var_gate
        elif type_operation == "povm":
            target_operations = self.povms
            get_size_func = self.size_var_povm
        else:
            raise ValueError(
                "'{}' is an unsupported operation type.".format(type_operation)
            )

        target_item_first_index = 0
        for i in range(index):
            target_item_first_index += get_size_func(i)
        return target_item_first_index

    def index_var_total_from_local_info(
        self, type_operation: str, index_operations: int, index_var_local: int
    ):
        # Returns the index in the optimization variable from local information.
        # The local information consists of type of the operation, its number in the list of operations of that type,
        # and the index in the variable that characterizes the operation.
        supported_types = ["state", "povm", "gate", "mprocess"]
        if type_operation not in supported_types:
            raise ValueError(
                "'{}' is an unsupported operation type. Supported Operations: {}.".format(
                    type_operation, ",".join(supported_types)
                )
            )
        first_index_map = self._get_operation_type_to_total_index_map()
        index_var_total = (
            first_index_map[type_operation]
            + self._get_operation_item_var_first_index(type_operation, index_operations)
            + index_var_local
        )
        return index_var_total

    def _get_type_operation_from_index_var_total(self, index_var_total: int) -> str:
        first_index_map = self._get_operation_type_to_total_index_map()
        type_operation: str
        if 0 <= index_var_total < first_index_map["gate"]:
            type_operation = "state"
        elif first_index_map["gate"] <= index_var_total < first_index_map["povm"]:
            type_operation = "gate"
        elif first_index_map["povm"] <= index_var_total < first_index_map["mprocess"]:
            type_operation = "povm"
        else:
            raise IndexError(
                f"index_var_total is out of range. index_var_total={index_var_total}"
            )
        return type_operation

    def local_info_from_index_var_total(self, index_var_total: int) -> dict:
        # Type Operation
        type_operation = self._get_type_operation_from_index_var_total(index_var_total)

        # Index Operations
        #   This function is split to make it easier to test.
        #   However, first_index_map is called twice, in this method and in _get_type_operation_from_index_var_total,
        #   so if speed is slow, it should be modified.
        first_index_map = self._get_operation_type_to_total_index_map()
        mid_level_index = index_var_total - first_index_map[type_operation]

        # Index Var Total
        target_operations: List[QOperation]
        if type_operation == "state":
            target_operations = self.states
            get_size_func = self.size_var_state
        elif type_operation == "gate":
            target_operations = self.gates
            get_size_func = self.size_var_gate
        elif type_operation == "povm":
            target_operations = self.povms
            get_size_func = self.size_var_povm

        first_index = 0
        for i, target in enumerate(target_operations):
            local_item_size = get_size_func(i)
            if first_index <= mid_level_index < first_index + local_item_size:
                index_operations = i
                index_var_local = mid_level_index - first_index
            first_index += local_item_size

        local_info = dict(
            type_operation=type_operation,
            index_operations=index_operations,
            index_var_local=index_var_local,
        )
        return local_info

    def _all_qoperations(self) -> List["QOperations"]:
        # Do NOT change the order
        return self.states + self.gates + self.povms

    def set_qoperations_from_var_total(self, var_total: np.ndarray) -> "SetQOperations":
        # returns SetQOperations corresponding to var_total
        actual, expected = len(var_total), self.size_var_total()
        if actual != expected:
            error_message = (
                "the length of var_total is wrong. expceted: {}, actual: {}".format(
                    expected, actual
                )
            )
            raise ValueError(error_message)

        q_operation2func_map = {
            State: convert_var_to_state,
            Gate: convert_var_to_gate,
            Povm: convert_var_to_povm,
        }

        new_q_operation_dict = {State: [], Gate: [], Povm: []}

        all_q_operations = self._all_qoperations()
        start_index = 0
        for q_operation in all_q_operations:
            end_index = start_index + len(q_operation.to_var())

            var = var_total[start_index:end_index]
            new_q_operation = q_operation.generate_from_var(var=var)
            new_q_operation_dict[type(q_operation)].append(new_q_operation)

            start_index = end_index

        new_set_qoperations = SetQOperations(
            states=new_q_operation_dict[State],
            gates=new_q_operation_dict[Gate],
            povms=new_q_operation_dict[Povm],
        )
        return new_set_qoperations
