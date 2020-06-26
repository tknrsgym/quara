from typing import List, Tuple

import numpy as np

from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.qoperations import SetQOperations
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.qcircuit.experiment import Experiment


class StandardQst(StandardQTomography):
    def __init__(
        self,
        povms: List[Povm],
        is_physicality_required: bool = False,
        is_estimation_object: bool = False,
        on_para_eq_constraint: bool = False,
        on_algo_eq_constraint: bool = False,
        on_algo_ineq_constraint: bool = False,
        eps_proj_physical: float = 10 ** (-4),
    ):
        # create Experiment
        schedules = []
        for index in range(len(povms)):
            schedule = [("state", 0), ("povm", index)]
            schedules.append(schedule)
        experiment = Experiment(
            states=[None], gates=[], povms=povms, schedules=schedules
        )

        # create SetQOperations
        state = State(
            povms[0]._composite_system,
            np.zeros(povms[0].vecs[0].shape, dtype=np.float64),
            is_physicality_required=is_physicality_required,
            is_estimation_object=is_estimation_object,
            on_para_eq_constraint=on_para_eq_constraint,
            on_algo_eq_constraint=on_algo_eq_constraint,
            on_algo_ineq_constraint=on_algo_ineq_constraint,
            eps_proj_physical=eps_proj_physical,
        )
        set_qoperations = SetQOperations(states=[state], gates=[], povms=[])

        # create map
        self._map_experiment_to_setqoperations = {("state", 0): ("state", 0)}
        self._map_setqoperations_to_experiment = {("state", 0): ("state", 0)}

        # initialize super class
        super().__init__(experiment, set_qoperations)

        # validate
        if not self.is_valid_experiment():
            raise Exception

        # calc and set coeff0s, coeff1s, matA and vecB
        self._set_coeffs(experiment, on_para_eq_constraint)

    def _set_coeffs(self, experiment: Experiment, on_para_eq_constraint: bool):
        # coeff0s and coeff1s
        self._coeffs_0th = dict()
        self._coeffs_1st = dict()
        tmp_coeffs_0th = []
        tmp_coeffs_1st = []
        for schedule_index, schedule in enumerate(self._experiment.schedules):
            povm_index = schedule[-1][1]
            povm = self._experiment.povms[povm_index]
            for element_index, vec in enumerate(povm.vecs):
                if on_para_eq_constraint:
                    self._coeffs_0th[(schedule_index, element_index)] = vec[0]
                    self._coeffs_1st[(schedule_index, element_index)] = vec[1:]
                    tmp_coeffs_0th.append(vec[0])
                    tmp_coeffs_1st.append(vec[1:])
                else:
                    self._coeffs_0th[(schedule_index, element_index)] = 0
                    self._coeffs_1st[(schedule_index, element_index)] = vec
                    tmp_coeffs_0th.append(0)
                    tmp_coeffs_1st.append(vec)

    def is_valid_experiment(self) -> bool:
        povms = self._experiment.povms
        checks = [
            povms[0]._composite_system == povm._composite_system for povm in povms[1:]
        ]
        return all(checks)

    def _get_state_index(self, experiment: Experiment, schedule_index: int) -> int:
        schedule = experiment.schedules[schedule_index]
        state_index = schedule[0][1]
        return state_index

    def calc_prob_dist(self, schedule_index: int, state: State) -> List[float]:
        tmp_experiment = self._experiment.copy()
        state_index = self._get_state_index(tmp_experiment, schedule_index)
        tmp_experiment.states[state_index] = state
        return tmp_experiment.calc_prob_dist(schedule_index)

    def calc_prob_dists(self, state: State) -> List[List[float]]:
        tmp_experiment = self._experiment.copy()
        for schedule_index in range(len(tmp_experiment.schedules)):
            state_index = self._get_state_index(tmp_experiment, schedule_index)
            tmp_experiment.states[state_index] = state
        prob_dists = tmp_experiment.calc_prob_dists()
        return prob_dists

    def generate_empi_dist(
        self, schedule_index: int, state: State, num_sum: int, seed: int = None
    ) -> Tuple[int, np.array]:
        tmp_experiment = self._experiment.copy()
        state_index = self._get_state_index(tmp_experiment, schedule_index)
        tmp_experiment.states[state_index] = state
        return tmp_experiment.generate_empi_dist(schedule_index, [num_sum], seed)

    def generate_empi_dists(
        self, state: State, num_sum: int, seed: int = None
    ) -> List[Tuple[int, np.array]]:
        tmp_experiment = self._experiment.copy()
        for schedule_index in range(len(tmp_experiment.schedules)):
            state_index = self._get_state_index(tmp_experiment, schedule_index)
            tmp_experiment.states[state_index] = state

        empi_dists = tmp_experiment.generate_empi_dists(num_sum, seed)
        return empi_dists

    def generate_empi_dists_sequence(
        self, state: State, num_sums: List[int], seed: int = None
    ) -> List[List[Tuple[int, np.array]]]:
        tmp_experiment = self._experiment.copy()

        empi_dists_sequence = []
        for schedule_index in range(len(tmp_experiment.schedules)):
            state_index = self._get_state_index(tmp_experiment, schedule_index)
            tmp_experiment.states[state_index] = state

        empi_dists_sequence = tmp_experiment.generate_empi_dists_sequence(
            num_sums, seed
        )
        return empi_dists_sequence
