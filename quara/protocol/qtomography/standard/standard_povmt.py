from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
import itertools
from typing import List, Tuple

import numpy as np

from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.qoperations import SetQOperations
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.qcircuit.experiment import Experiment


class StandardPovmt(StandardQTomography):
    def __init__(
        self,
        states: List[State],
        is_physicality_required: bool = False,
        is_estimation_object: bool = False,
        on_para_eq_constraint: bool = False,
        on_algo_eq_constraint: bool = False,
        on_algo_ineq_constraint: bool = False,
        eps_proj_physical: float = 10 ** (-4),
    ):

        schedules = []
        for index in range(len(states)):
            schedule = [("state", index), ("povm", 0)]
            schedules.append(schedule)
        experiment = Experiment(
            states=states, gates=[], povms=[None], schedules=schedules
        )

        vec_n = states[0].vec.shape[0]  # TODO
        vecs = [np.zeros(states[0].vec.shape, dtype=np.float64) for _ in range(vec_n)]
        povm = Povm(
            c_sys=states[0]._composite_system,
            vecs=vecs,
            is_physicality_required=is_physicality_required,
            is_estimation_object=is_estimation_object,
            on_para_eq_constraint=on_para_eq_constraint,
            on_algo_eq_constraint=on_algo_eq_constraint,
            on_algo_ineq_constraint=on_algo_ineq_constraint,
            eps_proj_physical=eps_proj_physical,
        )

        set_qoperations = SetQOperations(states=[], gates=[], povms=[povm])

        super().__init__(experiment, set_qoperations)

        # validate
        if not self.is_valid_experiment():
            raise ValueError(
                "the experiment is not valid. all povms must have same CompositeSystem."
            )

        if on_para_eq_constraint:
            self._num_variables = povm.dim ** 2 - 1
        else:
            self._num_variables = povm.dim ** 2

        # create map
        self._map_experiment_to_setqoperations = {("povm", 0): ("povm", 0)}
        self._map_setqoperations_to_experiment = {("povm", 0): ("povm", 0)}

        # calc and set coeff0s, coeff1s, matA and vecB
        # self._set_coeffs(experiment, on_para_eq_constraint)

        self.debug_set_qoperations = set_qoperations
        self.debug_experiment = experiment

    def is_valid_experiment(self) -> bool:
        states = self._experiment.states
        if len(states) <= 1:
            return True

        checks = [
            states[0]._composite_system is state._composite_system
            for state in states[1:]
        ]
        return all(checks)

    def generate_empi_dists_sequence(
        self, povm: Povm, num_sums: List[int], seeds: List[int] = None
    ) -> List[List[Tuple[int, np.array]]]:
        # TODO:
        tmp_experiment = self._experiment.copy()

        list_num_sums = [num_sums] * self._num_schedules
        list_num_sums_tmp = [list(num_sums) for num_sums in zip(*list_num_sums)]
        list_seeds = [seeds] * self._num_schedules
        list_seeds_tmp = [list(seeds) for seeds in zip(*list_seeds)]

        for schedule_index in range(len(tmp_experiment.schedules)):
            # Trueに相当するインデックスを取得して置き換える
            target_index = self._get_target_index(tmp_experiment, schedule_index)
            print(target_index)
            print(f"len:{len(tmp_experiment.povms)}")
            tmp_experiment.povms[target_index] = povm

        empi_dists_sequence_tmp = tmp_experiment.generate_empi_dists_sequence(
            list_num_sums_tmp, list_seeds_tmp,
        )
        empi_dists_sequence = [
            list(empi_dists) for empi_dists in zip(*empi_dists_sequence_tmp)
        ]
        return empi_dists_sequence

    def _get_target_index(self, experiment: Experiment, schedule_index: int) -> int:
        schedule = experiment.schedules[schedule_index]
        ITEM_INDEX = 1
        target_index = schedule[ITEM_INDEX][1]
        return target_index
