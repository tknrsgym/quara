import itertools
from itertools import product
from typing import List, Tuple, Union

import numpy as np
from quara.objects import mprocess

from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.mprocess import MProcess
from quara.objects.qoperation import QOperation
from quara.objects.qoperations import SetQOperations
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.qcircuit.experiment import Experiment
from quara.utils import matrix_util
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
        seed: int = None,
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
            states=states, gates=[None], povms=povms, schedules=schedules, seed=seed
        )
        self._validate_schedules(schedules)

        # Make SetQOperation
        size = states[0].dim ** 2
        hss = [np.zeros((size, size), dtype=np.float64) for _ in num_outcomes]
        mprocess = MProcess(
            c_sys=states[0].composite_system,
            hss=hss,
            is_physicality_required=is_physicality_required,
            is_estimation_object=is_estimation_object,
            on_para_eq_constraint=on_para_eq_constraint,
            eps_proj_physical=eps_proj_physical,
        )
        set_qoperations = SetQOperations(states=[], mprocesses=[mprocess], povms=[])

        super().__init__(experiment, set_qoperations)

        # validate
        if not self.is_valid_experiment():
            raise ValueError(
                "the experiment is not valid. all CompositeSystem of testers must have same ElementalSystems."
            )

        # TODO: modify
        if on_para_eq_constraint:
            self._num_variables = mprocess.dim ** 4 - mprocess.dim ** 2
        else:
            self._num_variables = mprocess.dim ** 4

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

    def estimation_object_type(self) -> type:
        return MProcess

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
        mprocess: MProcess,
        num_sum: int,
        seed_or_stream: Union[int, np.random.RandomState] = None,
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
        tmp_experiment.mprocesses[target_index] = mprocess

        stream = to_stream(seed_or_stream)
        empi_dist_seq = tmp_experiment.generate_empi_dist_sequence(
            schedule_index, [num_sum], seed_or_stream=stream
        )
        return empi_dist_seq[0]

    def generate_empi_dists_sequence(
        self,
        mprocess: MProcess,
        num_sums: List[int],
        seed_or_stream: Union[int, np.random.RandomState] = None,
    ) -> List[List[Tuple[int, np.ndarray]]]:
        tmp_experiment = self._experiment.copy()

        list_num_sums = [num_sums] * self._num_schedules
        list_num_sums_tmp = [list(num_sums) for num_sums in zip(*list_num_sums)]

        for schedule_index in range(len(tmp_experiment.schedules)):
            # Get the index corresponding to True and replace it.
            target_index = self._get_target_index(tmp_experiment, schedule_index)
            tmp_experiment.gates[target_index] = mprocess

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
        # 0:state -> 1:mprocess -> 2:povm
        MPROCESS_ITEM_INDEX = 1
        target_index = schedule[MPROCESS_ITEM_INDEX][1]
        return target_index

    def _set_coeffs(self, experiment: Experiment, on_para_eq_constraint: bool):
        raise NotImplementedError()
