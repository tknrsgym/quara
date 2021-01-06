from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from typing import List, Tuple

import numpy as np

from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate
from quara.objects.qoperation import QOperation
from quara.objects.qoperations import SetQOperations
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.qcircuit.experiment import Experiment
from quara.utils import matrix_util


class StandardQpt(StandardQTomography):
    _estimated_qoperation_type = Gate

    def __init__(
        self,
        states: List[State],
        povms: List[Povm],
        is_physicality_required: bool = False,
        is_estimation_object: bool = False,
        on_para_eq_constraint: bool = False,
        on_algo_eq_constraint: bool = False,
        on_algo_ineq_constraint: bool = False,
        eps_proj_physical: float = None,
        seed: int = None,
    ):
        # Make Experment with states
        schedules = []
        for i, _ in enumerate(states):
            for j, _ in enumerate(povms):
                schedules.append([("state", i), ("gate", 0), ("povm", j)])

        experiment = Experiment(
            states=states, gates=[None], povms=povms, schedules=schedules, seed=seed
        )

        # Make SetQOperation
        size = states[0].dim ** 2
        hs = np.zeros((size, size), dtype=np.float64)
        gate = Gate(
            c_sys=states[0].composite_system,
            hs=hs,
            is_physicality_required=is_physicality_required,
            is_estimation_object=is_estimation_object,
            on_para_eq_constraint=on_para_eq_constraint,
            on_algo_eq_constraint=on_algo_eq_constraint,
            on_algo_ineq_constraint=on_algo_ineq_constraint,
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
            self._num_variables = gate.dim ** 2 - 1
        else:
            self._num_variables = gate.dim ** 2

        # create map
        self._map_experiment_to_setqoperations = {("gate", 0): ("gate", 0)}
        self._map_setqoperations_to_experiment = {("gate", 0): ("gate", 0)}

        # calc and set coeff0s, coeff1s, matA and vecB
        self._set_coeffs(experiment, on_para_eq_constraint)
        self._on_para_eq_constraint = on_para_eq_constraint

    @property
    def on_para_eq_constraint(self):  # read only
        return self._on_para_eq_constraint

    def _is_all_same_composite_systems(self, targets):
        if len(targets) <= 1:
            return True

        checks = [
            targets[0]._composite_system == target._composite_system
            for target in targets[1:]
        ]
        return all(checks)

    def is_valid_experiment(self) -> bool:
        is_ok_states =  self._is_all_same_composite_systems(self._experiment.states)
        is_ok_povms =  self._is_all_same_composite_systems(self._experiment.povms)

        return is_ok_states and is_ok_povms

    def generate_empi_dists_sequence(
        self, gate: Povm, num_sums: List[int]
    ) -> List[List[Tuple[int, np.array]]]:
        tmp_experiment = self._experiment.copy()

        list_num_sums = [num_sums] * self._num_schedules
        list_num_sums_tmp = [list(num_sums) for num_sums in zip(*list_num_sums)]

        for schedule_index in range(len(tmp_experiment.schedules)):
            # Trueに相当するインデックスを取得して置き換える
            target_index = self._get_target_index(tmp_experiment, schedule_index)
            tmp_experiment.gates[target_index] = gate

        empi_dists_sequence_tmp = tmp_experiment.generate_empi_dists_sequence(
            list_num_sums_tmp
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
        POVM_ITEM_INDEX = 2  # TODO:

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

    def convert_var_to_qoperation(self, var: np.array) -> Gate:
        template = self._set_qoperations.gates[0]
        gate = template.generate_from_var(var=var)
        return gate

    def generate_empty_estimation_obj_with_setting_info(self) -> QOperation:
        empty_estimation_obj = self._set_qoperations.gates[0]
        return empty_estimation_obj.copy()

