from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from typing import List, Tuple, Union

import numpy as np

from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.qoperation import QOperation
from quara.objects.qoperations import SetQOperations
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.qcircuit.experiment import Experiment
from quara.utils import matrix_util


class StandardPovmt(StandardQTomography):
    _estimated_qoperation_type = Povm

    def __init__(
        self,
        states: List[State],
        measurement_n: int,
        is_physicality_required: bool = False,
        is_estimation_object: bool = False,
        on_para_eq_constraint: bool = False,
        on_algo_eq_constraint: bool = False,
        on_algo_ineq_constraint: bool = False,
        eps_proj_physical: float = None,
        seed: int = None,
    ):
        # Make Experment with states
        schedules = [[("state", i), ("povm", 0)] for i in range(len(states))]
        experiment = Experiment(
            states=states, gates=[], povms=[None], schedules=schedules, seed=seed
        )

        # Make SetQOperation
        # povmsはPovmを一つだけ持つ。
        # そのPovmはStateと同じcomposite systemを持ち、vec以外の値は引数の設定を代入する。
        # gates, states, mprocessesの長さは0.
        self._measurement_n = measurement_n
        vecs = [
            np.zeros(states[0].vec.shape, dtype=np.float64)
            for _ in range(self._measurement_n)
        ]
        povm = Povm(
            c_sys=states[0].composite_system,
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
                "the experiment is not valid. all CompositeSystem of testers must have same ElementalSystems."
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
        self._on_para_eq_constraint = on_para_eq_constraint

    @property
    def on_para_eq_constraint(self):  # read only
        return self._on_para_eq_constraint

    def is_valid_experiment(self) -> bool:
        states = self._experiment.states
        if len(states) <= 1:
            return True

        checks = [
            states[0]._composite_system == state._composite_system
            for state in states[1:]
        ]
        return all(checks)

    def _generate_matS(self):
        STATE_ITEM_INDEX = 0
        schedule = self._experiment.schedules[0]
        state_index = schedule[STATE_ITEM_INDEX][1]
        state = self._experiment.states[state_index]
        squared_dim = state.vec.shape[0]
        I = np.eye(squared_dim, dtype=np.float64)
        I_list = [I for _ in range(self._measurement_n - 1)]
        matS = np.hstack(I_list)

        return matS

    def _calc_mse_linear_analytical_mode_qoperation(
        self, qope: "QOperation", data_num_list: List[int]
    ) -> np.float64:
        if qope.on_para_eq_constraint:
            val_1st_term = self._calc_mse_linear_analytical_mode_var(
                qope, data_num_list
            )

            # generate matS
            matS = self._generate_matS()

            # calcurates val_2nd_term = Tr[S V(v^{L}) S^T]
            ScovST = matrix_util.calc_conjugate(
                matS, self.calc_covariance_linear_mat_total(qope, data_num_list)
            )
            val_2nd_term = np.trace(ScovST)
            val = val_1st_term + val_2nd_term

        else:
            val = self._calc_mse_linear_analytical_mode_var(qope, data_num_list)
        return val

    def calc_cramer_rao_bound(
        self, var: Union[QOperation, np.array], N: int, list_N: List[int]
    ) -> np.array:
        if self.on_para_eq_constraint:
            val_1st_term = self._calc_cramer_rao_bound(var, N, list_N)

            # generate matS
            matS = self._generate_matS()

            # calcurates val_2nd_term = Tr[S F^{-1} S^T]/N
            weights = [tmp_N / N for tmp_N in list_N]
            fisher = self.calc_fisher_matrix_total(var, weights)
            ScovST = matrix_util.calc_conjugate(matS, np.linalg.inv(fisher))
            val_2nd_term = np.trace(ScovST) / N
            val = val_1st_term + val_2nd_term
        else:
            val = self._calc_cramer_rao_bound(var, N, list_N)
        return val

    def generate_empi_dists_sequence(
        self, povm: Povm, num_sums: List[int]
    ) -> List[List[Tuple[int, np.array]]]:
        tmp_experiment = self._experiment.copy()

        list_num_sums = [num_sums] * self._num_schedules
        list_num_sums_tmp = [list(num_sums) for num_sums in zip(*list_num_sums)]

        for schedule_index in range(len(tmp_experiment.schedules)):
            # Trueに相当するインデックスを取得して置き換える
            target_index = self._get_target_index(tmp_experiment, schedule_index)
            tmp_experiment.povms[target_index] = povm

        empi_dists_sequence_tmp = tmp_experiment.generate_empi_dists_sequence(
            list_num_sums_tmp
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
        STATE_ITEM_INDEX = 0
        m = self._measurement_n

        # Create C
        c_list = []
        a_prime_list = []
        c_prime_list = []
        c_prime_tile_list = []
        for schedule_index, schedule in enumerate(self._experiment.schedules):
            state_index = schedule[STATE_ITEM_INDEX][1]
            state = self._experiment.states[state_index]
            vec_size = state.vec.shape[0]
            dim = np.sqrt(vec_size)
            print("==============================")
            print(f"{schedule_index=}")
            for m_index in range(m):
                print("------------------------------------")
                print(f"{m_index=}")
                pre_zeros = np.zeros((1, m_index * vec_size)).flatten()
                post_zeros = np.zeros((1, ((m - 1) - m_index) * vec_size)).flatten()

                stack_list = []
                if pre_zeros.size != 0:
                    stack_list.append(pre_zeros)
                stack_list.append(state.vec)
                if post_zeros.size != 0:
                    stack_list.append(post_zeros)
                c = np.hstack(stack_list)
                c_list.append(c)

                if on_para_eq_constraint:
                    a_prime, c_prime = np.split(c, [vec_size * (m - 1)])
                    a = a_prime - np.tile(c_prime, m - 1)

                    # TODO: remove
                    a_prime_list.append(a_prime)  # for debug
                    c_prime_list.append(c_prime)  # for debug
                    c_prime_tile_list.append(np.tile(c_prime, m - 1))  # for debug

                    self._coeffs_1st[(schedule_index, m_index)] = a
                    self._coeffs_0th[(schedule_index, m_index)] = (
                        np.sqrt(dim) * c_prime[0]
                    )
                else:
                    self._coeffs_1st[(schedule_index, m_index)] = c
                    self._coeffs_0th[(schedule_index, m_index)] = 0

        # TODO: remove
        # self._debug_c = np.vstack(c_list)
        # self._debug_a_prime = np.vstack(a_prime_list)
        # self._debug_c_prime = np.vstack(c_prime_list)
        # self._debug_c_prime_tile = np.vstack(c_prime_tile_list)

    def convert_var_to_qoperation(self, var: np.array) -> Povm:
        template = self._set_qoperations.povms[0]
        povm = template.generate_from_var(var=var)
        return povm

    def generate_empty_estimation_obj_with_setting_info(self) -> QOperation:
        """generates the empty estimation object with setting information.

        Returns
        -------
        QOperation
            the empty estimation object(Povm) with setting information.
        """
        empty_estimation_obj = self._set_qoperations.povms[0]
        return empty_estimation_obj.copy()
