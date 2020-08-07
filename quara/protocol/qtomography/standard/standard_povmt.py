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
        # Make Experment with states
        schedules = [[("state", i), ("povm", 0)] for i in range(len(states))]
        experiment = Experiment(
            states=states, gates=[], povms=[None], schedules=schedules
        )

        # Make SetQOperation
        # povmsはPovmを一つだけ持つ。
        # そのPovmはStateと同じcomposite systemを持ち、vec以外の値は引数の設定を代入する。
        # gates, states, mprocessesの長さは0.
        self._vec_n = int(np.sqrt(states[0].vec.shape[0]))  # TODO
        vecs = [
            np.zeros(states[0].vec.shape, dtype=np.float64) for _ in range(self._vec_n)
        ]
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
        # 引数となるStateオブジェクトのリストに対する妥当性チェック
        # リスト内のすべてのStateのcomposite systemが一致している。
        if not self.is_valid_experiment():
            raise ValueError(
                "the experiment is not valid. all povms must have same CompositeSystem."
            )

        # TODO:
        if on_para_eq_constraint:
            self._num_variables = povm.dim ** 2 - 1
        else:
            self._num_variables = povm.dim ** 2

        # create map
        self._map_experiment_to_setqoperations = {("povm", 0): ("povm", 0)}
        self._map_setqoperations_to_experiment = {("povm", 0): ("povm", 0)}

        # calc and set coeff0s, coeff1s, matA and vecB
        self._set_coeffs(experiment, on_para_eq_constraint)

        # For debug
        # TODO: remove
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
        POVM_ITEM_INDEX = 1
        target_index = schedule[POVM_ITEM_INDEX][1]
        return target_index

    def _set_coeffs(self, experiment: Experiment, on_para_eq_constraint: bool):
        # coeff0s and coeff1s
        self._coeffs_0th = dict()  # b
        self._coeffs_1st = dict()  # α
        tmp_coeffs_0th = []
        tmp_coeffs_1st = []
        STATE_ITEM_INDEX = 0
        c = []
        x = self._vec_n

        for schedule_index, schedule in enumerate(self._experiment.schedules):
            # 当該のスケジュールで指定されているStateが何番目のStateなのか、states内におけるindexを取得する
            state_index = schedule[STATE_ITEM_INDEX][1]
            # スケジュールで指定されているStateを取得する
            state = self._experiment.states[state_index]
            s = np.conjugate(state.vec.T) @ np.conjugate(state.vec.T)
            w = np.diag(np.array([s for _ in range(x)]))
            if on_para_eq_constraint:
                raise NotImplementedError()
            else:
                for x_index, c in enumerate(w):
                    self._coeffs_1st[(schedule_index, x_index)] = c
                    self._coeffs_0th[(schedule_index, x_index)] = 0
