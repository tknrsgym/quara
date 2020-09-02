import itertools
from typing import List, Tuple

import numpy as np

from quara.objects.state import State, convert_var_to_state
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
        seed: int = None,
    ):
        """Constructor

        Parameters
        ----------
        povms : List[Povm]
            testers of QST.
        is_physicality_required : bool, optional
            whether the QOperation is physically required, by default False
        is_estimation_object : bool, optional
            whether the QOperation is estimation object, by default False
        on_para_eq_constraint : bool, optional
            whether the parameters of QOperation are on equal constraint, by default False
        on_algo_eq_constraint : bool, optional
            whether the algorithm of estimate is on equal constraint, by default False
        on_algo_ineq_constraint : bool, optional
            whether the algorithm of estimate is on inequal constraint, by default False
        eps_proj_physical : float, optional
            threshold epsilon where the algorithm repeats the projection in order to make estimate object is physical, by default 10**(-4)
        seed : int, optional
            a seed used to generate random data, by default None.

        Raises
        ------
        ValueError
            the experiment is not valid.
        """
        # create Experiment
        schedules = []
        for index in range(len(povms)):
            schedule = [("state", 0), ("povm", index)]
            schedules.append(schedule)
        experiment = Experiment(
            states=[None], gates=[], povms=povms, schedules=schedules, seed=seed
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

        # initialize super class
        super().__init__(experiment, set_qoperations)

        # validate
        if not self.is_valid_experiment():
            raise ValueError(
                "the experiment is not valid. all povms must have same CompositeSystem."
            )

        # calc num_variables
        if on_para_eq_constraint:
            self._num_variables = state.dim ** 2 - 1
        else:
            self._num_variables = state.dim ** 2

        # create map
        self._map_experiment_to_setqoperations = {("state", 0): ("state", 0)}
        self._map_setqoperations_to_experiment = {("state", 0): ("state", 0)}

        # calc and set coeff0s, coeff1s, matA and vecB
        self._set_coeffs(experiment, on_para_eq_constraint)

        self.debug_set_qoperations = set_qoperations
        self.debug_experiment = experiment

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
        """returns whether the experiment is valid.

        all of the following conditions are ``True``, the state is physically correct:

        - all povms have same CompositeSystem.

        Returns
        -------
        bool
            whether the experiment is valid.
        """
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
        """calculates a probability distribution.
        
        see :func:`~quara.protocol.qtomography.qtomography.QTomography.calc_prob_dist`
        """
        tmp_experiment = self._experiment.copy()
        state_index = self._get_state_index(tmp_experiment, schedule_index)
        tmp_experiment.states[state_index] = state

        return tmp_experiment.calc_prob_dist(schedule_index)

    def calc_prob_dists(self, state: State) -> List[List[float]]:
        """calculates probability distributions.
        
        see :func:`~quara.protocol.qtomography.qtomography.QTomography.calc_prob_dists`
        """
        tmp_experiment = self._experiment.copy()
        for schedule_index in range(len(tmp_experiment.schedules)):
            state_index = self._get_state_index(tmp_experiment, schedule_index)
            tmp_experiment.states[state_index] = state

        prob_dists = tmp_experiment.calc_prob_dists()
        return prob_dists

    def generate_dataset(self, data_nums: List[int]) -> List[List[np.array]]:
        """calculates a probability distribution.
        
        see :func:`~quara.protocol.qtomography.qtomography.QTomography.generate_dataset`
        """
        # TODO
        pass

    def generate_empi_dist(
        self, schedule_index: int, state: State, num_sum: int
    ) -> Tuple[int, np.array]:
        """Generate empirical distribution using the data generated from probability distribution of specified schedules.

        Parameters
        ----------
        schedule_index : int
            schedule index.
        state : State
            true object.
        num_sum : int
            the number of data to use to generate the experience distributions for each schedule.

        Returns
        -------
        Tuple[int, np.array]
            Generated empirical distribution.
        """
        tmp_experiment = self._experiment.copy()
        state_index = self._get_state_index(tmp_experiment, schedule_index)
        tmp_experiment.states[state_index] = state

        empi_dist_seq = tmp_experiment.generate_empi_dist_sequence(
            schedule_index, [num_sum]
        )
        return empi_dist_seq[0]

    def generate_empi_dists(
        self, state: State, num_sum: int
    ) -> List[Tuple[int, np.array]]:
        """Generate empirical distributions using the data generated from probability distributions of all schedules.

        see :func:`~quara.protocol.qtomography.qtomography.QTomography.generate_empi_dists`
        """
        tmp_experiment = self._experiment.copy()
        for schedule_index in range(len(tmp_experiment.schedules)):
            state_index = self._get_state_index(tmp_experiment, schedule_index)
            tmp_experiment.states[state_index] = state

        num_sums = [num_sum] * self._num_schedules

        empi_dist_seq = tmp_experiment.generate_empi_dists_sequence([num_sums])

        empi_dists = list(itertools.chain.from_iterable(empi_dist_seq))
        return empi_dists

    def generate_empi_dists_sequence(
        self, state: State, num_sums: List[int]
    ) -> List[List[Tuple[int, np.array]]]:
        """Generate sequence of empirical distributions using the data generated from probability distributions of all schedules.

        Parameters
        ----------
        state : State
            true object.
        num_sums : List[int]
            list of the number of data to use to generate the experience distributions for each schedule.

        Returns
        -------
        List[List[Tuple[int, np.array]]]
            sequence of list of tuples for the number of data and experience distributions for each schedules.
        """
        tmp_experiment = self._experiment.copy()

        list_num_sums = [num_sums] * self._num_schedules
        list_num_sums_tmp = [list(num_sums) for num_sums in zip(*list_num_sums)]

        for schedule_index in range(len(tmp_experiment.schedules)):
            state_index = self._get_state_index(tmp_experiment, schedule_index)
            tmp_experiment.states[state_index] = state

        empi_dists_sequence_tmp = tmp_experiment.generate_empi_dists_sequence(
            list_num_sums_tmp
        )
        empi_dists_sequence = [
            list(empi_dists) for empi_dists in zip(*empi_dists_sequence_tmp)
        ]
        return empi_dists_sequence

    def convert_var_to_qoperation(self, var: np.array) -> State:
        """converts variable to QOperation.

        see :func:`~quara.protocol.qtomography.standard.standard_qtomography.StandardQTomography.convert_var_to_qoperation`
        """
        template = self._set_qoperations.states[0]
        state = template.generate_from_var(var=var)
        return state
