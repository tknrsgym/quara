import itertools
from itertools import product
from typing import List, Tuple, Union

import numpy as np
from scipy.linalg import block_diag

from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.mprocess import MProcess
from quara.objects.qoperation import QOperation
from quara.objects.qoperations import SetQOperations
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.protocol.qtomography.standard.standard_qpt import calc_c_qpt
from quara.qcircuit.experiment import Experiment
from quara.utils.number_util import to_stream


class StandardQmpt(StandardQTomography):
    _estimated_qoperation_type = MProcess

    def __init__(
        self,
        states: List[State],
        povms: List[Povm],
        num_outcomes: int,
        is_physicality_required: bool = False,
        is_estimation_object: bool = False,
        on_para_eq_constraint: bool = False,
        eps_proj_physical: float = None,
        eps_truncate_imaginary_part: float = None,
        seed_data: int = None,
        schedules: Union[str, List[List[Tuple]]] = "all",
    ):
        # Make Experment with states
        if type(schedules) == str:
            self._validate_schedules_str(schedules)
        if schedules == "all":
            schedules = []
            for i, j in product(range(len(states)), range(len(povms))):
                schedules.append([("state", i), ("mprocess", 0), ("povm", j)])

        experiment = Experiment(
            states=states,
            mprocesses=[None],
            povms=povms,
            schedules=schedules,
            seed_data=seed_data,
        )
        self._validate_schedules(schedules)
        self._num_outcomes = num_outcomes
        # Make SetQOperation
        size = states[0].dim ** 2
        hss = [np.zeros((size, size), dtype=np.float64) for _ in range(num_outcomes)]
        mprocess = MProcess(
            c_sys=states[0].composite_system,
            hss=hss,
            is_physicality_required=is_physicality_required,
            is_estimation_object=is_estimation_object,
            on_para_eq_constraint=on_para_eq_constraint,
            eps_proj_physical=eps_proj_physical,
            eps_truncate_imaginary_part=eps_truncate_imaginary_part,
        )
        set_qoperations = SetQOperations(states=[], mprocesses=[mprocess], povms=[])

        super().__init__(experiment, set_qoperations)

        # validate
        if not self.is_valid_experiment():
            raise ValueError(
                "the experiment is not valid. all CompositeSystem of testers must have same ElementalSystems."
            )

        if on_para_eq_constraint:
            self._num_variables = num_outcomes * mprocess.dim ** 4 - mprocess.dim ** 2
        else:
            self._num_variables = num_outcomes * mprocess.dim ** 4

        # create map
        self._map_experiment_to_setqoperations = {("mprocess", 0): ("mprocess", 0)}
        self._map_setqoperations_to_experiment = {("mprocess", 0): ("mprocess", 0)}

        # calc and set coeff0s, coeff1s, matA and vecB
        self._set_coeffs(experiment, on_para_eq_constraint)
        self._on_para_eq_constraint = on_para_eq_constraint

        self._template_qoperation = self._set_qoperations.mprocesses[0]

    def _validate_schedules(self, schedules):
        for i, schedule in enumerate(schedules):
            if (
                schedule[0][0] != "state"
                or schedule[1][0] != "mprocess"
                or schedule[2][0] != "povm"
            ):
                message = f"schedules[{i}] is invalid. "
                message += 'Schedule of Qmpt must be in format as \'[("state", state_index), ("mprocess", 0), ("povm", povm_index)]\', '
                message += f"not '{schedule}'."
                raise ValueError(message)
            if schedule[1][1] != 0:
                message = f"schedules[{i}] is invalid."
                message += f"MProcess index of schedule in Qmpt must be 0: {schedule}"
                raise ValueError(message)

    @property
    def on_para_eq_constraint(self):  # read only
        return self._on_para_eq_constraint

    @property
    def num_outcomes_estimate(self):
        return self._num_outcomes

    def num_outcomes(self, schedule_index: int) -> int:
        """returns the number of outcomes of probability distribution of a schedule index.

        Parameters
        ----------
        schedule_index: int

        Returns
        -------
        int
            the number of outcomes
        """
        assert schedule_index >= 0
        assert schedule_index < self.num_schedules
        povm_index = self._experiment.schedules[schedule_index][2][1]
        num_outcomes_povm = len(self._experiment._povms[povm_index].vecs)
        num_outcomes_mprocess = self._num_outcomes
        return num_outcomes_povm * num_outcomes_mprocess

    def estimation_object_type(self) -> type:
        return MProcess

    def is_valid_experiment(self) -> bool:
        is_ok_states = self.is_all_same_composite_systems(self._experiment.states)
        is_ok_povms = self.is_all_same_composite_systems(self._experiment.povms)

        return is_ok_states and is_ok_povms

    def generate_empi_dist(
        self,
        schedule_index: int,
        mprocess: MProcess,
        num_sum: int,
        seed_or_generator: Union[int, np.random.Generator] = None,
    ) -> Tuple[int, np.ndarray]:
        """Generate empirical distribution using the data generated from probability distribution of specified schedules.

        Parameters
        ----------
        schedule_index : int
            schedule index.
        mprocess: MProcess
            true object.
        num_sum : int
            the number of data to use to generate the experience distributions for each schedule.
        seed_or_generator : Union[int, np.random.Generator], optional
            If the type is int, it is assumed to be a seed used to generate random data.
            If the type is Generator, it is used to generate random data.
            If argument is None, np.random is used to generate random data.
            Default value is None.

        Returns
        -------
        Tuple[int, np.ndarray]
            Generated empirical distribution.
        """
        tmp_experiment = self._experiment.copy()
        target_index = self._get_target_index(tmp_experiment, schedule_index)
        tmp_experiment.mprocesses[target_index] = mprocess

        stream = to_stream(seed_or_generator)
        empi_dist_seq = tmp_experiment.generate_empi_dist_sequence(
            schedule_index, [num_sum], seed_or_generator=stream
        )
        return empi_dist_seq[0]

    def generate_empi_dists_sequence(
        self,
        mprocess: MProcess,
        num_sums: List[int],
        seed_or_generator: Union[int, np.random.Generator] = None,
    ) -> List[List[Tuple[int, np.ndarray]]]:
        tmp_experiment = self._experiment.copy()

        list_num_sums = [num_sums] * self._num_schedules
        list_num_sums_tmp = [list(num_sums) for num_sums in zip(*list_num_sums)]

        for schedule_index in range(len(tmp_experiment.schedules)):
            # Get the index corresponding to True and replace it.
            target_index = self._get_target_index(tmp_experiment, schedule_index)
            tmp_experiment.mprocesses[target_index] = mprocess

        stream = to_stream(seed_or_generator)
        empi_dists_sequence_tmp = tmp_experiment.generate_empi_dists_sequence(
            list_num_sums_tmp, seed_or_generator=stream
        )
        empi_dists_sequence = [
            list(empi_dists) for empi_dists in zip(*empi_dists_sequence_tmp)
        ]
        return empi_dists_sequence

    def _testers(self) -> List[Union[State, Povm]]:
        return self.experiment.states + self.experiment.povms

    def _get_target_index(self, experiment: Experiment, schedule_index: int) -> int:
        schedule = experiment.schedules[schedule_index]
        # 0:state -> 1:mprocess -> 2:povm
        MPROCESS_ITEM_INDEX = 1
        target_index = schedule[MPROCESS_ITEM_INDEX][1]
        return target_index

    def _set_coeffs(self, experiment: Experiment, on_para_eq_constraint: bool) -> None:
        # coeff0s and coeff1s
        self._coeffs_0th = dict()  # b
        self._coeffs_1st = dict()  # α

        _, _, c_qpt_dict = calc_c_qpt(
            states=self._experiment.states,
            povms=self._experiment.povms,
            schedules=self._experiment.schedules,
            on_para_eq_constraint=on_para_eq_constraint,
        )

        self._c_qpt_dict = c_qpt_dict

        dim = self._experiment.states[0].dim
        schedule_n = len(self._experiment.schedules)
        for schedule_index in range(schedule_n):
            c_qpt = c_qpt_dict[schedule_index]
            a_qmpt, b_qmpt = cqpt_to_cqmpt(
                c_qpt,
                m_mprocess=self.num_outcomes_estimate,
                dim=dim,
                on_para_eq_constraint=on_para_eq_constraint,
            )

            for element_index, a in enumerate(a_qmpt):
                self._coeffs_1st[(schedule_index, element_index)] = a
                self._coeffs_0th[(schedule_index, element_index)] = b_qmpt[
                    element_index
                ]

    def generate_empi_dists(
        self,
        mprocess: MProcess,
        num_sum: int,
        seed_or_generator: Union[int, np.random.Generator] = None,
    ) -> List[Tuple[int, np.ndarray]]:
        """Generate empirical distributions using the data generated from probability distributions of all schedules.

        see :func:`~quara.protocol.qtomography.qtomography.QTomography.generate_empi_dists`
        """
        tmp_experiment = self._experiment.copy()
        for schedule_index in range(len(tmp_experiment.schedules)):
            target_index = self._get_target_index(tmp_experiment, schedule_index)
            tmp_experiment.mprocesses[target_index] = mprocess

        num_sums = [num_sum] * self._num_schedules
        stream = to_stream(seed_or_generator)
        empi_dist_seq = tmp_experiment.generate_empi_dists_sequence(
            [num_sums], seed_or_generator=stream
        )

        empi_dists = list(itertools.chain.from_iterable(empi_dist_seq))
        return empi_dists

    def convert_var_to_qoperation(self, var: np.ndarray) -> MProcess:
        template = self._template_qoperation
        mprocess = template.generate_from_var(var=var)
        return mprocess

    def generate_empty_estimation_obj_with_setting_info(self) -> QOperation:
        empty_estimation_obj = self._set_qoperations.mprocesses[0]
        return empty_estimation_obj.copy()


def cqpt_to_cqmpt(
    c_qpt: np.ndarray, m_mprocess: int, dim: int, on_para_eq_constraint: bool
) -> List[np.ndarray]:
    c_list = [c_qpt] * m_mprocess
    if on_para_eq_constraint:
        if len(c_qpt.shape) < 2:
            c_qpt = c_qpt.reshape((1, c_qpt.shape[0]))
        d_qpt = c_qpt[:, : dim ** 2]
        e_qpt = c_qpt[:, dim ** 2 :]

        c_list = [c_qpt] * (m_mprocess - 1)
        a_0_left = block_diag(*c_list)

        a_0_right = np.zeros((a_0_left.shape[0], e_qpt.shape[1]))
        a_0 = np.hstack([a_0_left, a_0_right])

        d_dash_right_size = (d_qpt.shape[0], c_qpt.shape[1] - d_qpt.shape[1])
        d_dash = np.hstack([-d_qpt, np.zeros(d_dash_right_size)])

        a_1 = np.hstack([d_dash] * (m_mprocess - 1) + [e_qpt])
        a_qmpt = np.vstack([a_0, a_1])

        b_0 = np.zeros(d_qpt.shape[0] * (m_mprocess - 1))
        b_1 = d_qpt.T[0]
        b_qmpt = np.hstack([b_0, b_1])
    else:
        c_list = [c_qpt] * m_mprocess
        c_qmpt = block_diag(*c_list)

        a_qmpt = c_qmpt
        b_qmpt = np.zeros(c_qmpt.shape[0])

    return a_qmpt, b_qmpt
