import collections
from typing import List, Tuple

import numpy as np

from quara.objects.gate import Gate
from quara.objects.povm import Povm
from quara.objects.state import State

import quara.objects.operators as op
from quara.qcircuit import data_generator


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
    """
    実験設定のクラス
    トモグラフィーに限らず一般の量子回路を扱う
    """

    def __init__(
        self,
        states: List[State],
        povms: List[Povm],
        gates: List[Gate],
        schedules: List[List[Tuple[str, int]]],
        trial_nums: List[List[int]],
    ) -> None:

        # Validation
        self._validate_type(states, State)
        self._validate_type(povms, Povm)
        self._validate_type(gates, Gate)

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

        # Validate
        self._validate_trial_nums(trial_nums, self._schedules)

        # Set
        # 各スケジュールを試行する回数
        self._trial_nums: List[int] = trial_nums

    def _validate_trial_nums(self, trial_nums, schedules):
        if type(trial_nums) != list:
            raise TypeError("'trial_nums' must be a list with int elements.")
        for n in trial_nums:
            if type(n) != int:
                raise TypeError("'trial_nums' must be a list with int elements.")

        if len(trial_nums) != len(schedules):
            raise ValueError("'trial_nums' and 'schedules' must be equal in length.")

    @property
    def states(self) -> List[State]:
        return self._states

    @states.setter
    def states(self, value):
        # TODO: povm, gateとあわせて実装が冗長なので、あとで共通化する
        self._validate_type(value, State)
        old_value = self._states

        try:
            self._states = value
            self._validate_schedules(self._schedules)
        except QuaraScheduleItemError as e:
            self._states = old_value
            raise QuaraScheduleItemError(
                e.args[0] + "\nNew 'states' does not match schedules."
            )
        except Exception as e:
            self._states = old_value
            raise e

    @property
    def povms(self) -> List[Povm]:
        return self._povms

    @povms.setter
    def povms(self, value):
        self._validate_type(value, Povm)
        old_value = self._povms

        try:
            self._povms = value
            self._validate_schedules(self._schedules)
        except QuaraScheduleItemError as e:
            self._povms = old_value
            raise QuaraScheduleItemError(
                e.args[0] + "\nNew 'povms' does not match schedules."
            )
        except Exception as e:
            self._povms = old_value
            raise e

    @property
    def gates(self) -> List[Gate]:
        return self._gates

    @gates.setter
    def gates(self, value):
        self._validate_type(value, Gate)
        old_value = self._gates

        try:
            self._gates = value
            self._validate_schedules(self._schedules)
        except QuaraScheduleItemError as e:
            self._gates = old_value
            raise QuaraScheduleItemError(
                e.args[0] + "\nNew 'gates' does not match schedules."
            )
        except Exception as e:
            self._gates = old_value
            raise e

    @property
    def schedules(self) -> List[List[Tuple[str, int]]]:
        return self._schedules

    @schedules.setter
    def schedules(self, value):
        self._validate_schedules(value)
        # TODO: 変更後のスケジュールがtrial_numsと整合性が取れているかチェック
        self._validate_trial_nums(self._trial_nums, value)
        self._schedules = value

    @property
    def trial_nums(self) -> List[int]:
        return self._trial_nums

    @trial_nums.setter
    def trial_nums(self, value):
        self._validate_trial_nums(value, self._schedules)
        self._trial_nums = value

    def _validate_type(self, targets, expected_type) -> None:
        for target in targets:
            if target and not isinstance(target, expected_type):
                arg_name = expected_type.__name__.lower() + "s"
                error_message = "'{}' must be a list of {}.".format(
                    arg_name, expected_type.__name__
                )
                raise TypeError(error_message)

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

    def _validate_schedule_item(self, item: Tuple[str, int]) -> None:
        # scheduleのtuple単体の中身に問題がないか検証する
        if type(item) != tuple:
            raise TypeError("A schedule item must be a tuple of str and int.")

        if len(item) != 2:
            raise ValueError("A schedule item must be a tuple of str and int.")

        item_name, item_index = item[0], item[1]
        if type(item_name) != str:
            raise TypeError("A schedule item must be a tuple of str and int.")

        if type(item_index) != int:
            raise TypeError("A schedule item must be a tuple of str and int.")

        # TODO: 大文字・小文字の考慮。現状は小文字だけを想定する
        if item_name not in ["state", "povm", "gate", "mprocess"]:
            # TODO: error message
            raise ValueError(
                "The item of schedule can be specified as either 'state', 'povm', 'gate', or 'mprocess'."
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
            error_message = "The index out of range."
            error_message += "'{}s' is {} in length, but an index out of range was referenced in the schedule.".format(
                item_name, item_index
            )
            raise IndexError(error_message)

    def calc_probdist(self, index: int):
        # probDist
        # 確率分布を計算する
        # - 入力：scheduleのインデックス
        # - 出力：対応する確率分布

        schedule = self.schedules[index]
        key_map = dict(state=self._states, gate=self._gates, povm=self._povms)
        target_list = collections.deque()
        for item in schedule:
            k, i = item
            target = key_map[k][i]
            if not target:
                # TODO: error message
                raise ValueError("{}s[{}] is None.".format(k, i))
            target_list.appendleft(target)

        result = op.composite(*target_list)
        return result

    def calc_probdists(self) -> List[np.array]:
        # - list_probDist
        # - 入力：なし
        # - 出力：全scheduleに対する確率分布のリスト
        probdist_list = []
        for i in range(len(self.schedules)):
            r = self.calc_probdist(i)
            probdist_list.append(r)
        return probdist_list

    def generate_data(self, index: int, data_num: int, seed: int = None) -> List[int]:
        """
        - 入力
        - index_schedule (list_schedule内のscheduleを指定する整数)
        - データ数 $N$. 非負の整数（0は許すことにする）
        - 擬似乱数シードの値. 整数
        - 出力
        - $N$個の測定値（0 ~ $M$-1の間の整数）のリスト
        - 備考
        - メンバ関数probDistを使って確率分布を計算し、その確率分布と関数generate_data_from_probDistを使って擬似データを生成する.
        """
        # TODO: バリデーションが過剰なら共通化するなり簡潔な方法を考える
        # data_generatorの方でチェックをしているなら、ここではしなくてよい
        if type(data_num) != int:
            # TODO: error message
            raise TypeError

        if data_num < 0:
            # TODO: error message
            raise ValueError

        if type(index) != int:
            # TODO: error message
            raise TypeError

        if not (0 <= index < len(self.schedules)):
            # TODO: error message
            raise IndexError

        probdist = self.calc_probdist(index)
        data = data_generator.generate_data_from_probdist(probdist, data_num)
        return data

    def generate_dataset(
        self, data_num_list: List[int], seed: int = None,
    ) -> List[List[np.array]]:
        """
        - 入力
        - データ数のリスト $\{ N_j \}_{j=0}^{N_p -1}$. 非負の整数のリスト.
        - 擬似乱数シードのリスト. 整数のリスト.
        - 出力
        - 「$N_j$個の測定値(0 ~ $M_j$ -1の間の整数)のリスト」のリスト
        - 備考
        - メンバ関数list_probDistを使って確率分布のリストを計算し、その確率分布のリストと関数generate_dataSet_from_list_probDistを使って擬似データセットを計算する.
        """
        if len(data_num_list) != len(self.schedules):
            # TODO: error message
            raise ValueError
        dataset = []
        for i, data_num in enumerate(data_num_list):
            data = self.generate_data(index=i, data_num=data_num, seed=seed)
            dataset.append(data)
        return dataset

    def generate_empidist(self, index: int, list_num_sum: List[int], seed: int = None) -> List[Tuple[int, np.array]]:
        """
        - 入力
        - index_schedule (list_schedule内のscheduleを指定する整数)
        - 経験分布を計算するために和を取る数のリスト
        - 擬似乱数シードの値. 整数
        - 出力
        - データ数と経験分布のペアのリスト
        - 備考
        - メンバ関数generate_dataと関数calc_empiDistを組み合わせる. generate_dataに渡すデータ数は「和を取る数のリスト」中の最大値。
        """
        data_n = max(list_num_sum)
        data = self.generate_data(index=index, data_num=data_n, seed=seed)
        empidist = data_generator.calc_empidist(
            measurement_num=len(data), data=data, list_num_sum=list_num_sum
        )
        return empidist

    def generate_empidists(
        self, list_num_sums: List[List[int]], seeds: List[int] = None
    ) -> List[List[Tuple[int, np.array]]]:
        """
        - 入力：
        - 和を取る数のリストのリスト.
        - 擬似乱数シードのリスト. 整数のリスト.
        - 出力：
        - 「データ数と経験分布のペアのリスト」のリスト.
        - 備考
        - メンバ関数generate_dataSetと関数calc_list_empiDistを組み合わせる. generate_dataSetに渡す「データ数のリスト」は「「和を取る数のリスト」中の最大値のリスト」。
        """
        if len(list_num_sums) != len(self.schedules):
            # TODO: error message
            raise ValueError
        if seeds:
            if len(seeds) != len(self.schedules):
                # TODO: error message
                raise ValueError
        empidists = []
        # TODO: list_num_sumsの名前
        for i, list_num_sum in enumerate(list_num_sums):
            seed = seeds[i] if seeds else None
            empidist = self.generate_empidist(
                index=i, list_num_sum=list_num_sum, seed=seed
            )
            empidists.append(empidist)

        return empidists
