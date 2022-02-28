from typing import List, Dict, Union

import numpy as np

from quara.objects.qoperation import QOperation
from quara.objects.state import State
from quara.objects.gate import Gate
from quara.objects.povm import Povm
from quara.objects.mprocess import MProcess


class SetQOperations:
    def __init__(
        self,
        states: List[State] = None,
        gates: List[Gate] = None,
        povms: List[Povm] = None,
        mprocesses: List[MProcess] = None,
    ) -> None:

        states = [] if states is None else states
        gates = [] if gates is None else gates
        povms = [] if povms is None else povms
        mprocesses = [] if mprocesses is None else mprocesses

        # Validation
        self._validate_type(states, State)
        self._validate_type(povms, Povm)
        self._validate_type(gates, Gate)

        self._validate_type(mprocesses, MProcess)

        # Set
        self._states: List[State] = states
        self._povms: List[Povm] = povms
        self._gates: List[Gate] = gates
        self._mprocesses: List[MProcess] = mprocesses

    def _validate_type(self, targets, expected_type, arg_name: str = None) -> None:
        for target in targets:
            if target and not isinstance(target, expected_type):
                arg_name = (
                    arg_name if arg_name else expected_type.__name__.lower() + "s"
                )
                # ss -> es (ex: mprocesss -> mprocesses)
                arg_name = arg_name[:-2] + "ses" if arg_name[-2:] == "ss" else arg_name
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

    @property
    def mprocesses(self) -> List[MProcess]:
        return self._mprocesses

    @mprocesses.setter
    def mprocesses(self, value):
        self._validate_type(value, MProcess)
        self._mprocesses = value

    def qoperations(
        self, mode: str
    ) -> Union[List[State], List[Povm], List[Gate], List[MProcess]]:
        """returns qoperations with specified mode.

        Parameters
        ----------
        mode : str
            mode to get qoperations.
            mode can be "state", "povm", "gate", or "mprocess".

        Returns
        -------
        Union[List[State], List[Povm], List[Gate], List[MProcess]]
            qoperations with specified mode.

        Raises
        ------
        ValueError
            Unsupported mode is specified.
        """
        if mode == "state":
            return self.states
        elif mode == "povm":
            return self.povms
        elif mode == "gate":
            return self.gates
        elif mode == "mprocess":
            return self.mprocesses
        else:
            raise ValueError(f"Unsupported mode is specified. mode={mode}")

    def num_states(self):
        return len(self._states)

    def num_povms(self):
        return len(self._povms)

    def num_gates(self):
        return len(self._gates)

    def num_mprocesses(self):
        return len(self._mprocesses)

    def num_qoperations(self, mode: str) -> int:
        """returns number of qoperations with specified mode.

        Parameters
        ----------
        mode : str
            mode to get number of qoperations.
            mode can be "state", "povm", "gate", or "mprocess".

        Returns
        -------
        int
            number of qoperations with specified mode.

        Raises
        ------
        ValueError
            Unsupported mode is specified.
        """
        if mode == "state":
            return self.num_states()
        elif mode == "povm":
            return self.num_povms()
        elif mode == "gate":
            return self.num_gates()
        elif mode == "mprocess":
            return self.num_mprocesses()
        else:
            raise ValueError(f"An unsupported mode is specified. mode={mode}")

    def dim_state(self, index: int) -> int:
        # returns the dimension of the total system of the i-th state
        return self.states[index].dim

    def dim_gate(self, index: int) -> int:
        # returns the dimension of the total system of the i-th gate
        return self.gates[index].dim

    def dim_povm(self, index: int) -> int:
        # returns the dimension of the total system of the i-th povm
        return self.povms[index].dim

    def dim_mprocess(self, index: int) -> int:
        # returns the dimension of the total system of the i-th mprocess
        return self.mprocesses[index].dim

    def size_var_states(self) -> int:
        return len(self.var_states())

    def size_var_gates(self) -> int:
        return len(self.var_gates())

    def size_var_povms(self) -> int:
        return len(self.var_povms())

    def size_var_mprocesses(self) -> int:
        return len(self.var_mprocesses())

    def size_var_state(self, index: int) -> int:
        return len(self.var_state(index=index))

    def size_var_gate(self, index: int) -> int:
        return len(self.var_gate(index=index))

    def size_var_povm(self, index: int) -> int:
        return len(self.var_povm(index=index))

    def size_var_mprocess(self, index: int) -> int:
        return len(self.var_mprocess(index=index))

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

    def var_mprocess(self, index: int) -> np.ndarray:
        return self.mprocesses[index].to_var()

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

    def var_mprocesses(self) -> np.ndarray:
        vars = [mprocess.to_var() for mprocess in self.mprocesses]
        vars = np.hstack(vars) if vars else np.array([])
        return vars

    def var_total(self) -> np.ndarray:
        vars = np.hstack(
            [
                self.var_states(),
                self.var_gates(),
                self.var_povms(),
                self.var_mprocesses(),
            ]
        )
        return vars

    def _get_operation_mode_to_total_index_map(self) -> Dict[str, int]:
        states_first_index = 0
        gates_first_index = self.size_var_states()
        povms_first_index = gates_first_index + self.size_var_gates()
        mprocesses_first_index = povms_first_index + self.size_var_povms()
        return dict(
            state=states_first_index,
            gate=gates_first_index,
            povm=povms_first_index,
            mprocess=mprocesses_first_index,
        )

    def _get_operation_item_var_first_index(self, mode: str, index: int) -> int:
        # returns the index that is the place of the 'index'-th 'mode' starts in the whole var
        if mode == "state":
            get_size_func = self.size_var_state
        elif mode == "gate":
            get_size_func = self.size_var_gate
        elif mode == "povm":
            get_size_func = self.size_var_povm
        elif mode == "mprocess":
            get_size_func = self.size_var_mprocess
        else:
            raise ValueError("'{}' is an unsupported operation type.".format(mode))

        target_item_first_index = 0
        for i in range(index):
            target_item_first_index += get_size_func(i)
        return target_item_first_index

    def index_var_total_from_local_info(
        self, mode: str, index_operations: int, index_var_local: int
    ):
        # Returns the index in the optimization variable from local information.
        # The local information consists of type of the operation, its number in the list of operations of that type,
        # and the index in the variable that characterizes the operation.
        supported_types = ["state", "povm", "gate", "mprocess"]
        if mode not in supported_types:
            raise ValueError(
                "'{}' is an unsupported operation type. Supported Operations: {}.".format(
                    mode, ",".join(supported_types)
                )
            )
        first_index_map = self._get_operation_mode_to_total_index_map()
        index_var_total = (
            first_index_map[mode]
            + self._get_operation_item_var_first_index(mode, index_operations)
            + index_var_local
        )
        return index_var_total

    def _get_mode_from_index_var_total(self, index_var_total: int) -> str:
        first_index_map = self._get_operation_mode_to_total_index_map()
        mode: str
        if 0 <= index_var_total < first_index_map["gate"]:
            mode = "state"
        elif first_index_map["gate"] <= index_var_total < first_index_map["povm"]:
            mode = "gate"
        elif first_index_map["povm"] <= index_var_total < first_index_map["mprocess"]:
            mode = "povm"
        elif first_index_map["mprocess"] <= index_var_total < self.size_var_total():
            mode = "mprocess"
        else:
            raise IndexError(
                f"index_var_total is out of range. index_var_total={index_var_total}"
            )
        return mode

    def local_info_from_index_var_total(self, index_var_total: int) -> dict:
        # Type Operation
        mode = self._get_mode_from_index_var_total(index_var_total)

        # Index Operations
        #   This function is split to make it easier to test.
        #   However, first_index_map is called twice, in this method and in _get_mode_from_index_var_total,
        #   so if speed is slow, it should be modified.
        first_index_map = self._get_operation_mode_to_total_index_map()
        mid_level_index = index_var_total - first_index_map[mode]

        # Index Var Total
        target_operations: List[QOperation]
        if mode == "state":
            target_operations = self.states
            get_size_func = self.size_var_state
        elif mode == "gate":
            target_operations = self.gates
            get_size_func = self.size_var_gate
        elif mode == "povm":
            target_operations = self.povms
            get_size_func = self.size_var_povm
        elif mode == "mprocess":
            target_operations = self.mprocesses
            get_size_func = self.size_var_mprocess

        first_index = 0
        for i, _ in enumerate(target_operations):
            local_item_size = get_size_func(i)
            if first_index <= mid_level_index < first_index + local_item_size:
                index_operations = i
                index_var_local = mid_level_index - first_index
            first_index += local_item_size

        local_info = dict(
            mode=mode,
            index_operations=index_operations,
            index_var_local=index_var_local,
        )
        return local_info

    def _all_qoperations(self) -> List["QOperations"]:
        # Do NOT change the order
        return self.states + self.gates + self.povms + self.mprocesses

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

        new_q_operation_dict = {State: [], Gate: [], Povm: [], MProcess: []}

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
            mprocesses=new_q_operation_dict[MProcess],
        )
        return new_set_qoperations
