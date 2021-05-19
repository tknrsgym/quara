import collections
import copy
from typing import List, Tuple

import numpy as np

from quara.objects.gate import Gate
from quara.objects.povm import Povm
from quara.objects.state import State
import quara.objects.operators as op
from quara.qcircuit import data_generator


class QuaraScheduleItemError(Exception):
    """Raised when an element of the schedule is incorrect.

    Parameters
    ----------
    Exception : [type]
        [description]
    """

    pass


class QuaraScheduleOrderError(Exception):
    """Raised when the order of the schedule is incorrect.

    Parameters
    ----------
    Exception : [type]
        [description]
    """

    pass


class Experiment:
    """
    Class to manage experiment settings
    This class is not limited to tomography, but deals with general quantum circuits.
    """

    def __init__(
        self,
        states: List[State],
        povms: List[Povm],
        gates: List[Gate],
        schedules: List[List[Tuple[str, int]]],
        seed: int = None,
    ) -> None:

        # Validation
        self._validate_type(states, State)
        self._validate_type(povms, Povm)
        self._validate_type(gates, Gate)

        # Set
        self._states: List[State] = states
        self._povms: List[Povm] = povms
        self._gates: List[Gate] = gates
        # TODO: MProcess functions are not yet implemented. Only attributes are provided for future use.
        self._mprocesses: list = []

        # Validate
        self._validate_schedules(schedules)
        # Set
        self._schedules: List[List[Tuple[str, int]]] = schedules
        self._seed: int = seed

        # Set seed
        self.reset_seed(self._seed)

    @property
    def states(self) -> List[State]:
        return self._states

    @states.setter
    def states(self, value):
        self._validate_type(value, State)
        objdict = dict(
            state=value,
            povm=self._povms,
            gate=self._gates,
            mprocess=self._mprocesses,
        )

        try:
            self._validate_schedules(self._schedules, objdict=objdict)
        except QuaraScheduleItemError as e:
            raise QuaraScheduleItemError(
                e.args[0] + "\nNew 'states' does not match schedules."
            )
        else:
            self._states = value

    @property
    def povms(self) -> List[Povm]:
        return self._povms

    @povms.setter
    def povms(self, value):
        self._validate_type(value, Povm)
        objdict = dict(
            state=self._states,
            povm=value,
            gate=self._gates,
            mprocess=self._mprocesses,
        )

        try:
            self._validate_schedules(self._schedules, objdict=objdict)
        except QuaraScheduleItemError as e:
            raise QuaraScheduleItemError(
                e.args[0] + "\nNew 'povms' does not match schedules."
            )
        else:
            self._povms = value

    @property
    def gates(self) -> List[Gate]:
        return self._gates

    @gates.setter
    def gates(self, value):
        self._validate_type(value, Gate)

        objdict = dict(
            state=self._states,
            povm=self._povms,
            gate=value,
            mprocess=self._mprocesses,
        )
        try:
            self._validate_schedules(self._schedules, objdict=objdict)
        except QuaraScheduleItemError as e:
            raise QuaraScheduleItemError(
                e.args[0] + "\nNew 'gates' does not match schedules."
            )
        else:
            self._gates = value

    @property
    def schedules(self) -> List[List[Tuple[str, int]]]:
        return self._schedules

    @schedules.setter
    def schedules(self, value):
        self._validate_schedules(value)
        self._schedules = value

    @property
    def seed(self) -> int:
        return self._seed

    def reset_seed(self, seed: int) -> None:
        """reset new seed.

        Parameters
        ----------
        seed : int
            new seed.
        """
        self._seed = seed
        if self._seed is not None:
            np.random.seed(self._seed)

    def _validate_type(self, targets, expected_type) -> None:
        for target in targets:
            if target and not isinstance(target, expected_type):
                arg_name = expected_type.__name__.lower() + "s"
                error_message = "'{}' must be a list of {}.".format(
                    arg_name, expected_type.__name__
                )
                if type(target) == list:
                    type_names = set([type(t) for t in target])
                    type_names = ", ".join(type_names)
                    error_message += f"Type of parameter passed: List of {type_names}"
                else:
                    error_message += f"Type of parameter passed: {type(target)}"
                    error_message += f"\n{target.__str__()}"

                raise TypeError(error_message)

    def _validate_schedules(
        self, schedules: List[List[Tuple[str, int]]], objdict: dict = None
    ) -> None:
        """
        - The schedule always starts with the state.
        - There must be one state.
        - The gate and mprocess are 0~N
        - POVM is 0 or 1.
        - The last one is povm or mprocess

        Parameters
        ----------
        schedules : List[List[Tuple[str, int]]]
            [description]

        Returns
        -------
        bool
            [description]
        """

        for i, schedule in enumerate(schedules):
            try:
                for j, item in enumerate(schedule):
                    self._validate_schedule_item(item, objdict=objdict)
            except (ValueError, IndexError, TypeError) as e:
                message = "The item in the schedules[{}] is invalid.\n".format(i)
                message += "Invalid Schedule: [{}] {}\n".format(i, str(schedule))
                message += "{}: {}\n".format(j, item)
                message += "\nDetail: {}".format(e.args[0])
                raise QuaraScheduleItemError(message)

            try:
                self._validate_schedule_order(schedule)
            except ValueError as e:
                message = "There is a schedule with an invalid order.\n"
                message += "Invalid Schedule: [{}] {}\n".format(i, str(schedule))
                message += "Detail: {}".format(e.args[0])
                raise QuaraScheduleOrderError(message)

    def _validate_schedule_order(self, schedule: List[Tuple[str, int]]) -> None:
        """Validate that the order of the schedule is correct.
        For example, check to see if the schedule starts with 'state' and ends with 'povm'.

        Parameters
        ----------
        schedule : List[Tuple[str, int]]
            Schedule to be validated
        """

        if len(schedule) < 2:
            raise ValueError(
                "The schedule is too short. The schedule should start with state and end with povm or mprocess."
            )

        TYPE_INDEX = 0
        INDEX_INDEX = 1

        if schedule[0][TYPE_INDEX] != "state":
            raise ValueError("The first element of the schedule must be a 'state'.")
        if schedule[-1][TYPE_INDEX] not in ["povm", "mprocess"]:
            raise ValueError(
                "The last element of the schedule must be either 'povm' or 'mprocess'."
            )

        counter = collections.Counter([s[TYPE_INDEX] for s in schedule])
        if counter["state"] >= 2:
            raise ValueError(
                "There are too many States; one schedule can only contain one State."
            )
        if counter["povm"] >= 2:
            raise ValueError(
                "There are too many POVMs; one schedule can only contain one POVM."
            )

    def _validate_schedule_item(self, item: Tuple[str, int], objdict=None) -> None:
        """Validate that the item in the schedule is correct

        Parameters
        ----------
        item : Tuple[str, int]
            Schedule item to be validated.

        Raises
        ------
        TypeError
            [description]
        ValueError
            [description]
        TypeError
            [description]
        TypeError
            [description]
        ValueError
            [description]
        IndexError
            [description]
        IndexError
            [description]
        IndexError
            [description]
        """
        if type(item) != tuple:
            raise TypeError("A schedule item must be a tuple of str and int.")

        if len(item) != 2:
            raise ValueError("A schedule item must be a tuple of str and int.")

        item_name, item_index = item[0], item[1]
        if type(item_name) != str:
            raise TypeError("A schedule item must be a tuple of str and int.")

        if type(item_index) != int:
            raise TypeError("A schedule item must be a tuple of str and int.")

        # Currently, only lowercase is assumed.
        if item_name not in ["state", "povm", "gate", "mprocess"]:
            raise ValueError(
                "The item of schedule can be specified as either 'state', 'povm', 'gate', or 'mprocess'."
            )

        now_povms = objdict["povm"] if objdict else self._povms
        if item_name == "povm" and not now_povms:
            raise IndexError(
                "'povm' is used in the schedule, but no povm is given. Give a list of Povm to parameter 'povms' in the constructor."
            )
        now_mprocesses = objdict["mprocess"] if objdict else self._mprocesses
        if item_name == "mprocess" and not now_mprocesses:
            raise IndexError(
                "'mprocess' is used in the schedule, but no mprocess is given. Give a list of Mprocess to parameter 'mprocesses' in the constructor."
            )

        if not objdict:
            objdict = dict(
                state=self._states,
                povm=self._povms,
                gate=self._gates,
                mprocess=self._mprocesses,
            )
        if not (0 <= item_index < len(objdict[item_name])):
            error_message = "The index out of range."
            error_message += "'{}s' is {} in length, but an index out of range was referenced in the schedule.".format(
                item_name, item_index
            )
            raise IndexError(error_message)

    def _validate_schedule_index(self, schedule_index: int) -> None:
        if type(schedule_index) != int:
            raise TypeError("The type of 'schedule_index' must be int.")

        if not (0 <= schedule_index < len(self.schedules)):
            error_message = "The value of 'schedule_index' must be an integer between 0 and {}.".format(
                len(self.schedules) - 1
            )
            raise IndexError(error_message)

    def _validate_eq_schedule_len(self, target: list, var_name: str) -> None:
        if type(target) != list:
            error_message = "The type of '{}' must be list.".format(var_name)
            raise TypeError(error_message)

        if len(target) != len(self.schedules):
            error_message = "The number of elements in '{}' must be the same as the number of 'schedules';\n".format(
                var_name
            )
            error_message += "The length of '{}': {}\n".format(var_name, len(target))
            error_message += "The length of 'schedules': {}\n".format(
                len(self.schedules)
            )
            raise ValueError(error_message)

    def copy(self):
        """returns copied Experiment.

        Returns
        -------
        Experiment
            copied Experiment.
        """
        states = copy.copy(self.states)
        gates = copy.copy(self.gates)
        povms = copy.copy(self.povms)
        schedules = copy.copy(self.schedules)

        experiment = Experiment(
            states=states, gates=gates, povms=povms, schedules=schedules
        )
        return experiment

    def calc_prob_dist(self, schedule_index: int) -> np.ndarray:
        """Calculate the probability distributionthe by running the specified schedule.

        Parameters
        ----------
        schedule_index : int
            Index of the schedule

        Returns
        -------
        np.ndarray
            Probability distribution

        Raises
        ------
        ValueError
            If the object referenced in the schedule, such as State, POVM, Gate, or Mprocess, is None.
        """
        self._validate_schedule_index(schedule_index)
        schedule = self.schedules[schedule_index]
        key_map = dict(state=self._states, gate=self._gates, povm=self._povms)
        targets = collections.deque()
        for item in schedule:
            k, i = item
            target = key_map[k][i]
            if not target:
                raise ValueError("{}s[{}] is None.".format(k, i))
            targets.appendleft(target)

        prob_dist = op.compose_qoperations(*targets)
        return prob_dist

    def calc_prob_dists(self) -> List[np.ndarray]:
        """Caluclate probability distributions for all schedules.

        Returns
        -------
        List[np.ndarray]
            Probability distributions for all schedules
        """
        prob_dists = []
        for i in range(len(self.schedules)):
            r = self.calc_prob_dist(i)
            prob_dists.append(r)
        return prob_dists

    def generate_data(self, schedule_index: int, data_num: int) -> List[int]:
        """Runs the specified schedule to caluclate the probability distribution and generate random data.

        Parameters
        ----------
        schedule_index : int
            Index of the schedule.
        data_num : int
            Length of the data.

        Returns
        -------
        List[int]
            Generated data.

        Raises
        ------
        TypeError
            [description]
        ValueError
            [description]
        IndexError
            [description]
        """
        if type(data_num) != int:
            raise TypeError("The type of 'data_num' must be int.")

        if data_num < 0:
            raise ValueError("The value of 'data_num' must be a non-negative integer.")

        self._validate_schedule_index(schedule_index)

        prob_dist = self.calc_prob_dist(schedule_index)
        data = data_generator.generate_data_from_prob_dist(prob_dist, data_num)
        return data

    def generate_dataset(self, data_nums: List[int]) -> List[List[np.ndarray]]:
        """Run all the schedules to caluclate the probability distribution and generate random data.

        Parameters
        ----------
        data_nums : List[int]
            A list of the number of data to be generated in each schedule. This parameter should be a list of non-negative integers.

        Returns
        -------
        List[List[np.ndarray]]
            Generated dataset.
        """

        self._validate_eq_schedule_len(data_nums, "data_nums")

        prob_dists = self.calc_prob_dists()

        dataset = data_generator.generate_dataset_from_prob_dists(
            prob_dists=prob_dists, data_nums=data_nums
        )
        return dataset

    def generate_empi_dist_sequence(
        self, schedule_index: int, num_sums: List[int]
    ) -> List[Tuple[int, np.ndarray]]:
        """Generate an empirical distribution using the data generated from the probability distribution of a specified schedule.

        Uses generated data from 0-th to ``num_sums[index]``-th to calculate empirical distributions.

        Parameters
        ----------
        schedule_index : int
            Index of schedule.
        num_sums : List[int]
            List of the number of data to caluclate the experience distribution

        Returns
        -------
        List[Tuple[int, np.ndarray]]
            A list of the numbers of data and empirical distribution.
        """
        data_n = max(num_sums)
        # Calculating to get the measurement_num.
        # The probability distribution is calculated twice, in this function and in the generate_data method,
        # so it should be improved.
        prob_dist = self.calc_prob_dist(schedule_index)
        measurement_num = len(prob_dist)

        data = self.generate_data(schedule_index=schedule_index, data_num=data_n)
        empi_dist = data_generator.calc_empi_dist_sequence(
            measurement_num=measurement_num, data=data, num_sums=num_sums
        )
        return empi_dist

    def generate_empi_dists_sequence(
        self, list_num_sums: List[List[int]]
    ) -> List[List[Tuple[int, np.ndarray]]]:
        """Generate empirical distributions using the data generated from probability distributions of all specified schedules.

        Parameters
        ----------
        list_num_sums : List[List[int]]
            A list of the number of data to use to calculate the experience distribution for each schedule.

        Returns
        -------
        List[List[Tuple[int, np.ndarray]]]
            A list of tuples for the number of data and experience distribution for each schedules.
        """
        for num_sums in list_num_sums:
            self._validate_eq_schedule_len(num_sums, "list_num_sums")

        measurement_nums = [len(prob_dist) for prob_dist in self.calc_prob_dists()]
        datasets = self.generate_dataset(data_nums=list_num_sums[-1])

        list_num_sums_tmp = [list(num_sums) for num_sums in zip(*list_num_sums)]

        empi_dists_sequence = data_generator.calc_empi_dists_sequence(
            measurement_nums=measurement_nums,
            dataset=datasets,
            list_num_sums=list_num_sums_tmp,
        )

        return empi_dists_sequence
