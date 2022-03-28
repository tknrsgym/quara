from typing import List, Union

from quara.interface.qulacs.conversion import (
    convert_gate_quara_to_qulacs,
    convert_instrument_quara_to_qulacs,
    convert_state_quara_to_qulacs,
    get_index_tuple_from_kraus_matrices_indices,

)
from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.gate import Gate

from quara.objects.gate_typical import (
    generate_gate_from_gate_name,
    get_gate_names_1qubit,
    get_gate_names_1qutrit,
    get_gate_names_2qubit,
    get_gate_names_2qutrit,
    get_gate_names_3qubit,
)
from quara.objects.mprocess import MProcess
from quara.objects.mprocess_typical import generate_mprocess_from_name
from quara.objects.multinomial_distribution import MultinomialDistribution
from quara.objects.state import State
from quara.objects.state_typical import generate_state_from_name

from qulacs import state as qulacs_state
from qulacs import DensityMatrix

import numpy as np

class CircuitResult:
    """
    Class to manage experiment settings
    This class is not limited to tomography, but deals with general quantum circuits.
    """

    def __init__(self, circuit_overview: List[dict]):
        """Constructor

        Parameters
        ----------
        circuit_overview : List[dict]
            Overview of a quantum circuit which contains dictionaries of quantum objects.
            see Circuit class for more information.
        """
        self._raw_result: List[int] = []
        self._empi_dists: Union[List[MultinomialDistribution],  None] = None
        self._circuit_overview: List[dict] = circuit_overview

    def append_raw_output(self, raw_output: List[int]):
        """Appends raw output of a quantum circuit into an internal array.

        Parameters
        ----------
        raw_output : List[int]
            List of readouts of the measurements in a circuit.

        Returns
        -------
        None
        """
        self._raw_result.append(raw_output)

    @property
    def empi_dists(self):
        return self._empi_dists

    @property
    def raw_result(self):
        return self._raw_result

    def calc_empi_dist(self) -> List[MultinomialDistribution]:
        """Calculates empirical distributions based on raw readout data for every measurement operator in a quantum circuit.

        Parameters
        ----------
        None

        Returns
        -------
        List[MultinomialDistribution]
            Number of distributions match to the number of measurement operators in a quantum circuit.
        """
        counts = [[0 for _ in qobj["KrausMatrixIndices"]] for qobj in self._circuit_overview if qobj["Type"]=="MProcess"]
        for raw_data in self._raw_result:
            for i, outcome in enumerate(raw_data):
                counts[i][outcome] += 1

        empi_dists = [np.array(count, dtype=np.float64)/sum(count) for count in counts]
        self._empi_dists = [ MultinomialDistribution(ps) for ps in empi_dists ]
        return self._empi_dists

    def __str__(self):
        desc = f"Type:\n{self.__class__.__name__}\n\n"
        desc += f"Circuit Overview:\n[\n"
        for qobject in self._circuit_overview:
            desc += f"  {qobject.__str__()},\n"
        desc += "]\n\n"
        if self._empi_dists:
            desc += "Empirical distributions:\n[\n"
            for empi_dist in self._empi_dists:
                desc += f"  {empi_dist.ps},\n"
            desc += f"]\n"
        desc = desc.rstrip("\n")
        return desc

        

class Circuit(object):
    def __init__(self, num: int, mode: str = "qubit"):
        """Constructor

        Parameters
        ----------
        num : int
            number of qubits in a quantum circuit.
        mode: 
            currently supports only "qubit".
        """
        self._num = num
        self._mode = mode
        self._qobjects: List[dict] = []

    def _generate_gate_from_name(self, gate_name: str, ids: List[int]) -> Gate:
        gate_q_num = None
        if gate_name in get_gate_names_1qubit():
            gate_q_num = 1
        elif gate_name in get_gate_names_2qubit():
            gate_q_num = 2
        elif gate_name in get_gate_names_3qubit():
            gate_q_num = 3
        elif gate_name in get_gate_names_1qutrit():
            gate_q_num = 1
        elif gate_name in get_gate_names_2qutrit():
            gate_q_num = 2
        else:
            raise ValueError(f"No such gate_name: {gate_name}")
        c_sys = generate_composite_system(self._mode, gate_q_num)
        return generate_gate_from_gate_name(gate_name, c_sys, ids)

    def _calc_qulacs_target_ids(self, ids: List[int]) -> List[int]:
        if self._mode == "qubit":
            return [self._num - 1 - i for i in ids[::-1]]
        elif self._mode == "qutrit":
            # add padding because qutrit system is embeded into qubit system.
            # double the number of target indices with dummy targets
            target_ids = []
            for i in ids[::-1]:
                target_ids.append(2*self._num - 1 - 2*i)
                target_ids.append(2*self._num - 2 - 2*i)
            return target_ids

    def add_gate(
        self,
        ids: List[int],
        gate: Union[Gate, None] = None,
        gate_name: Union[str, None] = None,
    ) -> None:
        """Adds a quantum gate to the circuit.

        Parameters
        ----------
        ids: List[int]
            List of indices of qubits which the gate will be applied on.
        gate : Gate
            optional. Either gate or gate_name must be specified.
            A gate object that will be added to a circuit.
        gate_name : str
            optional. Either gate or gate_name must be specified.
            A name of a gate object that will be added to a circuit.
            Valid gate names can be listed by calling  get_gate_names(). 

        Returns
        -------
        List[MultinomialDistribution]
            Number of distributions match to the number of measurement operators in a quantum circuit.
        """
        qulacs_target_ids = self._calc_qulacs_target_ids(ids)
        quara_ids = [i for i in range(len(ids))]
        if gate_name is None and gate is None:
            raise ValueError("Either gate_name or gate must be defined.")
        if gate_name is not None and gate is not None:
            raise ValueError(
                "You can't define both gate_name and gate. Define either one of these arguments."
            )
        if gate_name:
            gate = self._generate_gate_from_name(gate_name, quara_ids)
        if self._mode == "qutrit":
            c_sys = generate_composite_system("qubit", 2*len(ids))
            gate = Gate.embed_qoperation_from_qutrits_to_qubits(gate, list(c_sys.elemental_systems))
        qulacs_gate = convert_gate_quara_to_qulacs(gate, qulacs_target_ids)
        self._qobjects.append(
            {
                "Type": "Gate",
                "QObject": qulacs_gate,
                "TargetIds": ids,
                "Name": gate_name,
            }
        )

    def _generate_mprocess_from_name(
        self, mprocess_name: str, ids=List[int]
    ) -> MProcess:
        c_sys = generate_composite_system("qubit", len(ids))
        return generate_mprocess_from_name(c_sys, mprocess_name)

    def add_mprocess(
        self,
        ids: List[int],
        mprocess: Union[MProcess, None] = None,
        mprocess_name: Union[str, None] = None,
    ) -> None:
        """Adds a mprocess to the circuit.

        Parameters
        ----------
        ids: List[int]
            List of indices of qubits which the measurement will be applied on.
        mprocess : MProcess
            optional. Either mprocess or mprocess_name must be specified.
            A mprocess object that will be added to a circuit.
        mprocess_name : str
            optional. Either mprocess or mprocess_name must be specified.
            A name of a mprocess that will be added to a circuit.
            Valid mprocess names can be listed by calling  get_mprocess_names_type1() and get_mprocess_names_type2(). 

        Returns
        -------
        List[MultinomialDistribution]
            Number of distributions match to the number of measurement operators in a quantum circuit.
        """
        qulacs_target_ids = self._calc_qulacs_target_ids(ids)
        if mprocess_name is None and mprocess is None:
            raise ValueError("Either mprocess_name or mprocess must be defined.")
        if mprocess_name is not None and mprocess is not None:
            raise ValueError(
                "You can't define both mprocess_name and mprocess. Define either one of these arguments."
            )
        if mprocess_name:
            mprocess = self._generate_mprocess_from_name(mprocess_name, ids)
        if self._mode == "qutrit":
            c_sys = generate_composite_system("qubit", 2*len(ids))
            mprocess = MProcess.embed_qoperation_from_qutrits_to_qubits(mprocess, list(c_sys.elemental_systems))
        qulacs_instrument, qulacs_ids = convert_instrument_quara_to_qulacs(
            mprocess, qulacs_target_ids
        )
        self._qobjects.append(
            {
                "Type": "MProcess",
                "QObject": qulacs_instrument,
                "TargetIds": ids,
                "KrausMatrixIndices": qulacs_ids,
                "Name": mprocess_name,
            }
        )

    def _generate_initial_state_from_states_qubit(self, initial_states: List[State]) -> DensityMatrix:
        qulacs_density_mat = convert_state_quara_to_qulacs(initial_states[-1])
        # the order of qubits are opposite between quara and qulacs
        for state in initial_states[-2::-1]:
            tmp_mat = convert_state_quara_to_qulacs(state)
            qulacs_density_mat = qulacs_state.tensor_product(
                tmp_mat, qulacs_density_mat
            )
        return qulacs_density_mat

    def _generate_initial_state_from_states_qutrit(self, initial_states: List[State]) -> DensityMatrix:
        c_sys = generate_composite_system("qubit", 2)
        state_tmp = State.embed_qoperation_from_qutrits_to_qubits(initial_states[-1], list(c_sys.elemental_systems))
        qulacs_density_mat = convert_state_quara_to_qulacs(state_tmp)
        # the order of qubits are opposite between quara and qulacs
        for state in initial_states[-2::-1]:
            tmp_state = State.embed_qoperation_from_qutrits_to_qubits(state, list(c_sys.elemental_systems))
            tmp_mat = convert_state_quara_to_qulacs(tmp_state)
            qulacs_density_mat = qulacs_state.tensor_product(
                tmp_mat, qulacs_density_mat
            )
        return qulacs_density_mat

    def _generate_initial_state_from_states(self, initial_states: List[State]) -> DensityMatrix:
        if self._mode == "qubit":
            return self._generate_initial_state_from_states_qubit(initial_states)
        elif self._mode == "qutrit":
            return self._generate_initial_state_from_states_qutrit(initial_states)
            
    def _generate_initial_state_from_name_qutrit(self, initial_state_mode: str = "all_zero") -> DensityMatrix:
        states = []
        c_sys = generate_composite_system("qutrit", 1)
        c_sys_qubit = generate_composite_system("qubit", 2)
        if initial_state_mode == "all_zero":
            for _ in range(self._num):
                tmp_state = generate_state_from_name(c_sys, "01z0")
                states.append(State.embed_qoperation_from_qutrits_to_qubits(tmp_state, list(c_sys_qubit.elemental_systems)))
        qulacs_density_mat = convert_state_quara_to_qulacs(states[0])
        for state in states[1::]:
            tmp_mat = convert_state_quara_to_qulacs(state)
            qulacs_density_mat = qulacs_state.tensor_product(
                tmp_mat, qulacs_density_mat
            )
        return qulacs_density_mat


    def _generate_initial_state_from_name_qubit(self, initial_state_mode: str = "all_zero") -> DensityMatrix:
        states = []
        c_sys = generate_composite_system("qubit", 1)
        if initial_state_mode == "all_zero":
            for _ in range(self._num):
                states.append(generate_state_from_name(c_sys, "z0"))
        qulacs_density_mat = convert_state_quara_to_qulacs(states[0])
        for state in states[1::]:
            tmp_mat = convert_state_quara_to_qulacs(state)
            qulacs_density_mat = qulacs_state.tensor_product(
                tmp_mat, qulacs_density_mat
            )
        return qulacs_density_mat

    def _generate_initial_state_from_name(self, initial_state_mode) -> DensityMatrix:
        if self._mode == "qubit":
            return self._generate_initial_state_from_name_qubit(initial_state_mode)
        elif self._mode == "qutrit":
            return self._generate_initial_state_from_name_qutrit(initial_state_mode)

    def _run_once(
        self,
        initial_state_mode: Union[str, None] = None,
        initial_states: Union[List[State], None] = None,
    ) -> List[int]:
        # state preparation
        if initial_states:
            qulacs_density_mat = self._generate_initial_state_from_states(initial_states)
        elif initial_state_mode:
            qulacs_density_mat = self._generate_initial_state_from_name(initial_state_mode)
        else:
            raise ValueError("Define either initial_state_mode or initial_states")
        raw_result: List[int] = []
        for qobject in self._qobjects:
            qobject['QObject'].update_quantum_state(qulacs_density_mat)
            # read out value
            if qobject['Type'] == "MProcess":
                readout_value = qulacs_density_mat.get_classical_value(0)
                applied_operator_indices = get_index_tuple_from_kraus_matrices_indices(readout_value, qobject['KrausMatrixIndices'])
                raw_result.append(applied_operator_indices[0])
        return raw_result

    def run(
        self,
        num_shots: int,
        initial_state_mode: Union[str, None] = None,
        initial_states: Union[List[State], None] = None,
    ) -> CircuitResult:
        """Runs the quantum circuit with given initial state and number of shots.

        Parameters
        ----------
        num_shots: int
            number of shots to be performed.
        initial_state_mode
            optional. Either initial_state_mode or initial_states must be specified.
            only "all_zero" is available for now. "all_zero" will create a state where all qubits are "z0".
        initial_state : str
            optional. Either mprocess or mprocess_name must be specified.
            A name of a mprocess that will be added to a circuit.
            Valid mprocess names can be listed by calling  get_mprocess_names_type1() and get_mprocess_names_type2(). 

        Returns
        -------
        List[MultinomialDistribution]
            Number of distributions match to the number of measurement operators in a quantum circuit.
        """
        mprocess_found = False
        for qobject in self._qobjects[::-1]:
            if qobject["Type"] == "MProcess":
                mprocess_found = True
        if not mprocess_found:
            raise ValueError("Circuit must contain at least one measurement")

        result = CircuitResult(list(self))
        for _ in range(num_shots):
            raw_output = self._run_once(initial_state_mode, initial_states)
            result.append_raw_output(raw_output)
        result.calc_empi_dist()
        return result

    def __len__(self) -> int:
        return len(self._qobjects)

    def __getitem__(self, offset) -> dict:
        qobj = self._qobjects[offset].copy()
        del qobj["QObject"]
        return qobj

    def __str__(self) -> str:
        desc = f"Type:\n{self.__class__.__name__}\n\n"
        desc += f"QObjects:\n[\n"
        for qobject in self._qobjects:
            tmp = qobject.copy()
            del tmp["QObject"]
            desc += f"  {tmp.__str__()},\n"
        desc += "]\n\n"
        desc = desc.rstrip("\n")
        return desc