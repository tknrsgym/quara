from abc import abstractmethod
from typing import List, Dict

import numpy as np
from quara.objects.qoperation import QOperation
from quara.objects.state import State, convert_state_to_var
from quara.objects.gate import Gate, convert_gate_to_var
from quara.objects.povm import Povm, convert_povm_to_var


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
        # statesのi番目に対応するtotal systemの次元を返す.
        return self.states[index].dim

    def dim_gate(self, index: int) -> int:
        # gatesのi番目に対応するtotal systemの次元を返す.
        return self.gates[index].dim

    def dim_povm(self, index: int) -> int:
        # povmsのi番目に対応するtotal systemの次元を返す.
        return self.povms[index].dim

    def dim_mprosess(self, index: int) -> int:
        # TODO
        raise NotImplementedError()

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
        vars = np.hstack([self.var_states(), self.var_gates(), self.var_povms()])
        return vars

    def _get_operation_type_to_total_index_map(self) -> Dict[str, int]:
        states_first_index = 0
        gates_first_index = self.size_var_states()
        povms_first_index = gates_first_index + self.size_var_gates()
        # TODO: MProcess
        return dict(
            state=states_first_index, gate=gates_first_index, povm=povms_first_index
        )

    def _get_operation_item_var_first_index(
        self, type_operation: str, index: int
    ) -> int:
        # TODO: メソッド名をわかりやすくする
        # statesに格納されているi番目のStateが、states全体をvarにした時に何番目のインデックスから始まるか
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
        for i in range(index - 1):
            target_item_first_index += get_size_func(i)
        return target_item_first_index

    def index_var_total_from_local_info(
        self, type_operation: str, index_operations: int, index_var_local: int
    ):
        # 演算の種類、その種類の演算リストの中での番号、その演算を特徴づける変数中のインデックス、
        # から、最適化変数中のインデックスを返す
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

    def index_var_total_from_local_info(self, index_var_total: int):
        # 最適化変数中のインデックスから
        # 演算の種類
        # その種類の演算リストの中での番号
        # その演算を特徴づける変数中のインデックス
        # を返す
        pass

    def set_qoperations_from_var_total(self, var_total: np.array) -> SetQOperations:
        # numpy array var_totalに対応するsetListQOperationを返す
        pass
