import itertools
from itertools import product
from typing import List, Tuple, Union

import numpy as np

from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate
from quara.objects.qoperation import QOperation
from quara.objects.qoperations import SetQOperations
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.qcircuit.experiment import Experiment
from quara.utils import matrix_util
from quara.utils.number_util import to_stream


class StandardQpt(StandardQTomography):
    _estimated_qoperation_type = Gate

    def __init__(
        self,
        states: List[State],
        povms: List[Povm],
        is_physicality_required: bool = False,
        is_estimation_object: bool = False,
        on_para_eq_constraint: bool = False,
        eps_proj_physical: float = None,
        seed: int = None,
        schedules: Union[str, List[List[Tuple]]] = "all",
    ):
        # Make Experment with states
        if type(schedules) == str:
            self._validate_schedules_str(schedules)
        if schedules == "all":
            schedules = []
            for i, j in product(range(len(states)), range(len(povms))):
                schedules.append([("state", i), ("gate", 0), ("povm", j)])

        experiment = Experiment(
            states=states, gates=[None], povms=povms, schedules=schedules, seed=seed
        )
        self._validate_schedules(schedules)

        # Make SetQOperation
        size = states[0].dim ** 2
        hs = np.zeros((size, size), dtype=np.float64)
        gate = Gate(
            c_sys=states[0].composite_system,
            hs=hs,
            is_physicality_required=is_physicality_required,
            is_estimation_object=is_estimation_object,
            on_para_eq_constraint=on_para_eq_constraint,
            eps_proj_physical=eps_proj_physical,
        )
        set_qoperations = SetQOperations(states=[], gates=[gate], povms=[])

        super().__init__(experiment, set_qoperations)

        # validate
        if not self.is_valid_experiment():
            raise ValueError(
                "the experiment is not valid. all CompositeSystem of testers must have same ElementalSystems."
            )

        if on_para_eq_constraint:
            self._num_variables = gate.dim ** 4 - gate.dim ** 2
        else:
            self._num_variables = gate.dim ** 4

        # create map
        self._map_experiment_to_setqoperations = {("gate", 0): ("gate", 0)}
        self._map_setqoperations_to_experiment = {("gate", 0): ("gate", 0)}

        # calc and set coeff0s, coeff1s, matA and vecB
        self._set_coeffs(experiment, on_para_eq_constraint)
        self._on_para_eq_constraint = on_para_eq_constraint

        self._template_qoperation = self._set_qoperations.gates[0]

    def _validate_schedules(self, schedules):
        for i, schedule in enumerate(schedules):
            if (
                schedule[0][0] != "state"
                or schedule[1][0] != "gate"
                or schedule[2][0] != "povm"
            ):
                message = f"schedules[{i}] is invalid. "
                message += 'Schedule of Qpt must be in format as \'[("state", state_index), ("gate", 0), ("povm", povm_index)]\', '
                message += f"not '{schedule}'."
                raise ValueError(message)
            if schedule[1][1] != 0:
                message = f"schedules[{i}] is invalid."
                message += f"Gate index of schedule in Qpt must be 0: {schedule}"
                raise ValueError(message)

    @property
    def on_para_eq_constraint(self):  # read only
        return self._on_para_eq_constraint

    def estimation_object_type(self) -> type:
        return Gate

    def _is_all_same_composite_systems(self, targets):
        if len(targets) <= 1:
            return True

        checks = [
            targets[0]._composite_system == target._composite_system
            for target in targets[1:]
        ]
        return all(checks)

    def is_valid_experiment(self) -> bool:
        is_ok_states = self._is_all_same_composite_systems(self._experiment.states)
        is_ok_povms = self._is_all_same_composite_systems(self._experiment.povms)

        return is_ok_states and is_ok_povms

    def generate_empi_dist(
        self,
        schedule_index: int,
        gate: Gate,
        num_sum: int,
        seed_or_stream: Union[int, np.random.RandomState] = None,
    ) -> Tuple[int, np.ndarray]:
        """Generate empirical distribution using the data generated from probability distribution of specified schedules.

        Parameters
        ----------
        schedule_index : int
            schedule index.
        gate: Gate
            true object.
        num_sum : int
            the number of data to use to generate the experience distributions for each schedule.
        seed_or_stream : Union[int, np.random.RandomState], optional
            If the type is int, it is assumed to be a seed used to generate random data.
            If the type is RandomState, it is used to generate random data.
            If argument is None, np.random is used to generate random data.
            Default value is None.

        Returns
        -------
        Tuple[int, np.ndarray]
            Generated empirical distribution.
        """
        tmp_experiment = self._experiment.copy()
        target_index = self._get_target_index(tmp_experiment, schedule_index)
        tmp_experiment.gates[target_index] = gate

        stream = to_stream(seed_or_stream)
        empi_dist_seq = tmp_experiment.generate_empi_dist_sequence(
            schedule_index, [num_sum], seed_or_stream=stream
        )
        return empi_dist_seq[0]

    def generate_empi_dists(
        self,
        gate: Gate,
        num_sum: int,
        seed_or_stream: Union[int, np.random.RandomState] = None,
    ) -> List[Tuple[int, np.ndarray]]:
        """Generate empirical distributions using the data generated from probability distributions of all schedules.

        see :func:`~quara.protocol.qtomography.qtomography.QTomography.generate_empi_dists`
        """
        tmp_experiment = self._experiment.copy()
        for schedule_index in range(len(tmp_experiment.schedules)):
            target_index = self._get_target_index(tmp_experiment, schedule_index)
            tmp_experiment.gates[target_index] = gate

        num_sums = [num_sum] * self._num_schedules
        stream = to_stream(seed_or_stream)
        empi_dist_seq = tmp_experiment.generate_empi_dists_sequence(
            [num_sums], seed_or_stream=stream
        )

        empi_dists = list(itertools.chain.from_iterable(empi_dist_seq))
        return empi_dists

    def generate_empi_dists_sequence(
        self,
        gate: Gate,
        num_sums: List[int],
        seed_or_stream: Union[int, np.random.RandomState] = None,
    ) -> List[List[Tuple[int, np.ndarray]]]:
        tmp_experiment = self._experiment.copy()

        list_num_sums = [num_sums] * self._num_schedules
        list_num_sums_tmp = [list(num_sums) for num_sums in zip(*list_num_sums)]

        for schedule_index in range(len(tmp_experiment.schedules)):
            # Get the index corresponding to True and replace it.
            target_index = self._get_target_index(tmp_experiment, schedule_index)
            tmp_experiment.gates[target_index] = gate

        stream = to_stream(seed_or_stream)
        empi_dists_sequence_tmp = tmp_experiment.generate_empi_dists_sequence(
            list_num_sums_tmp, seed_or_stream=stream
        )
        empi_dists_sequence = [
            list(empi_dists) for empi_dists in zip(*empi_dists_sequence_tmp)
        ]
        return empi_dists_sequence

    def _get_target_index(self, experiment: Experiment, schedule_index: int) -> int:
        schedule = experiment.schedules[schedule_index]
        # 0:state -> 1:gate -> 2:povm
        GATE_ITEM_INDEX = 1
        target_index = schedule[GATE_ITEM_INDEX][1]
        return target_index

    def _set_coeffs(self, experiment: Experiment, on_para_eq_constraint: bool):

        # coeff0s and coeff1s
        self._coeffs_0th = dict()  # b
        self._coeffs_1st = dict()  # α
        STATE_ITEM_INDEX = 0
        POVM_ITEM_INDEX = 2

        # Create C
        total_index = 0
        c_list = []
        for schedule_index, schedule in enumerate(self._experiment.schedules):
            state_index = schedule[STATE_ITEM_INDEX][1]
            state = self._experiment.states[state_index]

            povm_index = schedule[POVM_ITEM_INDEX][1]
            povm = self._experiment.povms[povm_index]

            vec_size = state.vec.shape[0]
            dim = np.sqrt(vec_size)
            for m_index, povm_vec in enumerate(povm.vecs):  # each measurement
                c = np.kron(povm_vec, state.vec.T)

                if on_para_eq_constraint:
                    a = c[int(dim * dim) :]
                    self._coeffs_1st[(schedule_index, m_index)] = a
                    self._coeffs_0th[(schedule_index, m_index)] = c[0]
                else:
                    self._coeffs_1st[(schedule_index, m_index)] = c
                    self._coeffs_0th[(schedule_index, m_index)] = 0
                total_index += 1
                c_list.append(c)
        # for debugging and test
        self._C = np.array(c_list)

    def convert_var_to_qoperation(self, var: np.ndarray) -> Gate:
        # template = self._set_qoperations.gates[0]
        template = self._template_qoperation
        gate = template.generate_from_var(var=var)
        return gate

    def generate_empty_estimation_obj_with_setting_info(self) -> QOperation:
        empty_estimation_obj = self._set_qoperations.gates[0]
        return empty_estimation_obj.copy()

    def num_outcomes(self, schedule_index: int) -> int:
        """returns the number of outcomes of probability distribution of a schedule index.

        Parameters
        ----------
        schedule_index: int
            a schedule index

        Returns
        -------
        int
            the number of outcomes
        """
        assert schedule_index >= 0
        assert schedule_index < self.num_schedules
        povm_index = self._experiment.schedules[schedule_index][2][1]
        return len(self._experiment._povms[povm_index].vecs)
