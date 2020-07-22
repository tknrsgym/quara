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

        state = State(
            states[0]._composite_system,
            np.zeros(states[0].vec.shape, dtype=np.float64),
            is_physicality_required=is_physicality_required,
            is_estimation_object=is_estimation_object,
            on_para_eq_constraint=on_para_eq_constraint,
            on_algo_eq_constraint=on_algo_eq_constraint,
            on_algo_ineq_constraint=on_algo_ineq_constraint,
            eps_proj_physical=eps_proj_physical,
        )

        set_qoperations = SetQOperations(states=[], gates=[], povms=povms)

        super().__init__(experiment, set_qoperations)

        # validate
        if not self.is_valid_experiment():
            raise ValueError(
                "the experiment is not valid. all povms must have same CompositeSystem."
            )

        # TODO:

    def is_valid_experiment(self) -> bool:
        raise NotImplementedError()

