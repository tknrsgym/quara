import collections
from typing import List, Tuple

from quara.objects.gate import Gate
from quara.objects.povm import Povm
from quara.objects.state import State


class QuaraScheduleItemError(Exception):
    """スケジュールの要素が不正だった時に送出する例外

    Parameters
    ----------
    Exception : [type]
        [description]
    """

    pass


class QuaraScheduleOrderError(Exception):
    """スケジュールの並び順が不正だった時に送出する例外

    Parameters
    ----------
    Exception : [type]
        [description]
    """

    pass


class Experiment:
    def __init__(
        self,
        states: List[State],
        povms: List[Povm],
        gates: List[Gate],
        schedules: List[List[Tuple[str, int]]],
    ) -> None:

        # Validation
        if not self._is_valid_type(states, State):
            raise TypeError("'states' must be a list of State.")

        if not self._is_valid_type(povms, Povm):
            raise TypeError("'povms' must be a list of Povm.")

        if not self._is_valid_type(gates, Gate):
            raise TypeError("'gates' must be a list of Gate.")

        # Set
        self._states: List[State] = states
        self._povms: List[Povm] = povms
        self._gates: List[Gate] = gates
        # TODO: List[MProcess]
        self._mprocesses: list = []

        # Validate
        self._validate_schedules(schedules)
        # Set
        self._schedules: List[List[Tuple[str, int]]] = schedules

    @property
    def states(self) -> List[State]:
        return self._states

    @property
    def povms(self) -> List[Povm]:
        return self._povms

    @property
    def gates(self) -> List[Gate]:
        return self._gates

    @property
    def schedules(self) -> List[List[Tuple[str, int]]]:
        return self._schedules

    # TODO: setter

    def _is_valid_type(self, targets, expected_type) -> bool:
        for target in targets:
            if target and not isinstance(target, expected_type):
                return False
        return True

    def _validate_schedules(self, schedules: List[List[Tuple[str, int]]]) -> None:
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
            # 何番目のscheduleで何のエラーが発生したのかわかるようにする
            try:
                for j, item in enumerate(schedule):
                    self._validate_schedule_item(item)
            except (ValueError, IndexError, TypeError) as e:
                # TODO: error message
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

    def _validate_schedule_order(self, schedule: List[Tuple[str, int]]):
        """
        - schedule単体の並び順に問題がないか検証する
        - scheduleの最初はstate, 最後はpovmで終わっている。

        Parameters
        ----------
        schedule : List[Tuple[str, int]]
            [description]
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
            raise ValueError("scheduleの最後の要素はpovmかmprocessである必要があります")

        counter = collections.Counter([s[TYPE_INDEX] for s in schedule])
        if counter["state"] >= 2:
            raise ValueError("1つのスケジュールでStateは1つである必要があります")
        if counter["povm"] >= 2:
            raise ValueError("1つのスケジュールでPovmは1つである必要があります")

    def _validate_schedule_item(self, item: Tuple[str, int]) -> None:
        # scheduleのtuple単体の中身に問題がないか検証する
        if len(item) != 2:
            raise ValueError("Scheduleのitemは、strとintのタプルで表現してください")

        item_name = item[0]
        item_index = item[1]
        if type(item_name) != str:
            raise TypeError("Scheduleのitemは、strとintのタプルで表現してください")

        if type(item_index) != int:
            raise TypeError("Scheduleのitemは、strとintのタプルで表現してください")

        # TODO: 大文字・小文字の考慮。現状は小文字だけを想定する
        if item_name not in ["state", "povm", "gate", "mprocess"]:
            # TODO: error message
            raise ValueError(
                "Scheduleのitemではstate, povm, gate, mprocessのいずれかで指定してください。"
            )

        if item_name == "povm" and not self._povms:
            raise IndexError(
                "'povm' is used in the schedule, but no povm is given. Give a list of Povm to parameter 'povms' in the constructor."
            )

        if item_name == "mprocess" and not self._mprocesses:
            raise IndexError(
                "'mprocess' is used in the schedule, but no mprocess is given. Give a list of Mprocess to parameter 'mprocesses' in the constructor."
            )

        name2list = dict(
            state=self._states,
            povm=self._povms,
            gate=self._gates,
            mprocess=self._mprocesses,
        )
        if not (0 <= item_index < len(name2list[item_name])):
            # TODO: error message
            raise IndexError()

    def calc_prob_dist(self):
        pass

    def calc_prob_dists(self):
        pass
