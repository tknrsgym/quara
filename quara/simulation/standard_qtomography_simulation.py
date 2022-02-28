from typing import List, Optional, Union, Tuple
import copy
from collections import Counter
import dataclasses
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
import joblib

from quara.objects.qoperation import QOperation
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate
from quara.objects.mprocess import MProcess

from quara.minimization_algorithm.minimization_algorithm import (
    MinimizationAlgorithm,
    MinimizationAlgorithmOption,
)
from quara.loss_function.probability_based_loss_function import (
    ProbabilityBasedLossFunction,
    ProbabilityBasedLossFunctionOption,
)
from quara.protocol.qtomography.estimator import EstimationResult
from quara.protocol.qtomography.qtomography import QTomography
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
)
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.standard_qpt import StandardQpt
from quara.protocol.qtomography.standard.standard_povmt import StandardPovmt
from quara.protocol.qtomography.standard.standard_qmpt import StandardQmpt
from quara.protocol.qtomography.standard.standard_qtomography_estimator import (
    StandardQTomographyEstimator,
    StandardQTomographyEstimationResult,
)
from quara.simulation.generation_setting import (
    QOperationGenerationSettings,
    QOperationGenerationSetting,
)
from quara.simulation.depolarized_qoperation_generation_setting import (
    DepolarizedQOperationGenerationSetting,
)
from quara.simulation.random_effective_lindbladian_generation_setting import (
    RandomEffectiveLindbladianGenerationSetting,
)


class StandardQTomographySimulationSetting:
    def __init__(
        self,
        name: str,
        true_object: "QOperation",
        tester_objects: List["QOperation"],
        estimator: "Estimator",
        seed_data: int,
        n_rep: int,
        num_data: List[int],
        schedules: Union[str, List[List[int]]],
        eps_proj_physical: float,
        eps_truncate_imaginary_part: float,
        loss=None,
        loss_option=None,
        algo=None,
        algo_option=None,
    ) -> None:
        self.name = name
        self.true_object = copy.copy(true_object)
        self.tester_objects = copy.copy(tester_objects)
        self.estimator = copy.copy(estimator)
        self.loss = copy.copy(loss)
        self.loss_option = loss_option
        self.algo = copy.copy(algo)
        self.algo_option = algo_option

        self.seed_data = seed_data
        self.n_rep = n_rep
        self.num_data = num_data
        self.eps_proj_physical = eps_proj_physical
        self.eps_truncate_imaginary_part = eps_truncate_imaginary_part

        self.schedules = schedules

    def __str__(self):
        desc = f"Name: {self.name}"
        desc += f"\nTrue Object: {self.true_object}"

        counter = Counter([t.__class__.__name__ for t in self.tester_objects])
        desc += "\nTester Objects" + ", ".join(
            [f"{k} x {v}" for k, v in counter.items()]
        )

        desc += f"\nn_rep: {self.n_rep}"
        desc += f"\nnum_data: {self.num_data}"
        desc += f"\nseed_data: {self.seed_data}"
        desc += f"\nEstimator: {self.estimator.__class__.__name__}"
        desc += f"\neps_proj_physical: {self.eps_proj_physical}"
        desc += f"\neps_truncate_imaginary_part: {self.eps_truncate_imaginary_part}"
        loss = None if self.loss is None else self.loss.__class__.__name__
        desc += f"\nLoss: {loss}"
        algo = None if self.algo is None else self.algo.__class__.__name__
        desc += f"\nAlgo: {algo}"
        return desc

    def copy(self):
        return StandardQTomographySimulationSetting(
            name=self.name,
            true_object=self.true_object,
            tester_objects=self.tester_objects,
            estimator=self.estimator,
            loss=copy.deepcopy(self.loss),
            loss_option=self.loss_option,
            algo=copy.deepcopy(self.algo),
            algo_option=self.algo_option,
            seed_data=self.seed_data,
            n_rep=self.n_rep,
            num_data=self.num_data,
            schedules=self.schedules,
            eps_proj_physical=self.eps_proj_physical,
            eps_truncate_imaginary_part=self.eps_truncate_imaginary_part,
        )


@dataclasses.dataclass
class NoiseSetting:
    qoperation_base: Union[QOperation, str]
    method: str
    para: dict
    ids: List[int] = None

    def to_generation_setting(
        self, c_sys: "CompositeSystem", is_physicality_required: bool = True
    ) -> "QOperationGenerationSetting":
        if self.method is None:
            return QOperationGenerationSetting(
                qoperation_base=self.qoperation_base,
                c_sys=c_sys,
                ids=self.ids,
                is_physicality_required=is_physicality_required,
            )  # dummy

        name2class_map = {
            "depolarized": DepolarizedQOperationGenerationSetting,
            "random_effective_lindbladian": RandomEffectiveLindbladianGenerationSetting,
        }

        if self.method in name2class_map:
            target_class = name2class_map[self.method]
        else:
            message = f"noise_setting.method='{self.method}' is not implemented."
            raise NotImplementedError(message)

        return target_class(
            qoperation_base=self.qoperation_base,
            c_sys=c_sys,
            **self.para,
            ids=self.ids,
            is_physicality_required=is_physicality_required,
        )


@dataclasses.dataclass
class EstimatorTestSetting:
    true_object: NoiseSetting
    tester_objects: List[NoiseSetting]
    seed_data: int
    seed_qoperation: int
    n_rep: int
    num_data: List[int]
    n_sample: int
    schedules: Union[str, List[List[int]]]
    case_names: List[str]
    estimators: List["Estimator"]
    eps_proj_physical_list: List[float]
    eps_truncate_imaginary_part_list: List[float]
    algo_list: List[tuple]
    loss_list: List[tuple]
    parametrizations: List[bool]
    c_sys: "CompositeSystem"
    generation_setting_is_physicality_required: bool = True

    def to_pickle(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def to_generation_settings(self) -> QOperationGenerationSettings:
        true_setting = self.true_object.to_generation_setting(
            self.c_sys,
            is_physicality_required=self.generation_setting_is_physicality_required,
        )
        tester_settings = [
            setting.to_generation_setting(
                self.c_sys,
                is_physicality_required=self.generation_setting_is_physicality_required,
            )
            for setting in self.tester_objects
        ]
        generation_settings = QOperationGenerationSettings(
            true_setting=true_setting, tester_settings=tester_settings
        )
        return generation_settings

    def to_simulation_setting(
        self,
        true_object: "QOperation",
        tester_objects: List["QOperation"],
        case_index: int,
    ) -> StandardQTomographySimulationSetting:
        return StandardQTomographySimulationSetting(
            name=self.case_names[case_index],
            estimator=self.estimators[case_index],
            loss=self.loss_list[case_index][0],
            loss_option=self.loss_list[case_index][1],
            algo=self.algo_list[case_index][0],
            algo_option=self.algo_list[case_index][1],
            true_object=true_object,
            tester_objects=tester_objects,
            n_rep=self.n_rep,
            seed_data=self.seed_data,
            num_data=self.num_data,
            schedules=self.schedules,
            eps_proj_physical=self.eps_proj_physical_list[case_index],
            eps_truncate_imaginary_part=self.eps_truncate_imaginary_part_list[
                case_index
            ],
        )

    def show_description(self) -> None:
        """show the description of schema of EstimatorTestSetting."""
        print(f"=== decsription of {type(self)} ===")
        self._show_description(self)

    @staticmethod
    def _show_description(obj, indent: int = 0) -> None:
        # show attributes of "obj"
        for attr in dir(obj):
            # don't show private functions and utility functions
            if attr.startswith("_") or attr.startswith("to_"):
                continue

            # show attribute
            attr_obj = getattr(obj, attr)
            print(" " * indent, attr, type(attr_obj))

            # do recursive call
            if type(attr_obj) is dict or type(attr_obj).__name__ == "method":
                # if "obj" is dict or method or QTomography, it doesn't do recursive call
                continue
            elif type(attr_obj) is list:
                if attr in [
                    "algo_list",
                    "case_names",
                    "eps_proj_physical_list",
                    "eps_truncate_imaginary_part_list",
                    "estimators",
                    "loss_list",
                    "parametrizations",
                ]:
                    print(" " * (indent + 2), "[case_index]")
                elif attr == "num_data":
                    print(" " * (indent + 2), "[num_data_index]")
                elif attr == "tester_objects":
                    if len(attr_obj) == 0:
                        print(" " * (indent + 2), "[tester_index]")
                    else:
                        print(" " * (indent + 2), f"(list of {type(attr_obj[0])})")
                        print(" " * (indent + 2), "[tester_index]")
            elif type(attr_obj) is NoiseSetting:
                EstimatorTestSetting._show_description(attr_obj, indent + 2)


@dataclasses.dataclass
class SimulationResult:
    """Parameters

    Parameters
    -------
    estimation_results : List[EstimationResult]
        the index of list is the index of "n_rep".
    empi_dists_sequences: List[List[List[Tuple[int, np.ndarray]]]]
        the order of indexes is as follows: "n_rep", "num_data", and "schecule_index".
    qtomography: StandardQTomography
        StandardQTomography object.
    simulation_setting: StandardQTomographySimulationSetting
        StandardQTomographySimulationSetting object.
    result_index: dict
        dictionary of the check result.
        it contains the following keys: "name", "total_result", and "results".
    check_result: dict
        dictionary of the check result.
        it contains the following keys: "test_setting_index", "sample_index", and "case_index".
    """

    estimation_results: List["EstimationResult"]
    empi_dists_sequences: List[List[List[Tuple[int, np.ndarray]]]]
    qtomography: StandardQTomography
    simulation_setting: StandardQTomographySimulationSetting = None
    result_index: dict = None
    check_result: dict = None

    def get_variable_estimate(
        self, n_rep: int, num_data_index: int = None, num_data_value: int = None
    ) -> np.ndarray:
        """get variable estimate with specific parameter.
        Either num_data_index or num_data_value must be specified.

        Parameters
        ----------
        n_rep : int
            value of n_rep.
        num_data_index : int, optional
            index of num_data, by default None
        num_data_value : int, optional
            value of num_data, by default None

        Returns
        -------
        np.ndarray
            variable estimate.

        Raises
        ------
        ValueError
            Either num_data_index or num_data_value must be specified. But both are None.
        ValueError
            Either num_data_index or num_data_value must be specified. But both are specified.
        ValueError
            num_data_value is not found in SimulationResult.
        """
        if num_data_index is None and num_data_value is None:
            raise ValueError(
                f"Either num_data_index or num_data_value must be specified. But both are None."
            )
        elif num_data_index is not None and num_data_value is not None:
            raise ValueError(
                f"Either num_data_index or num_data_value must be specified. But both are specified."
            )
        elif num_data_index is not None:
            return self.estimation_results[n_rep].estimated_var_sequence[num_data_index]
        else:
            for num_data_index, num in enumerate(self.simulation_setting.num_data):
                if num == num_data_value:
                    return self.estimation_results[n_rep].estimated_var_sequence[
                        num_data_index
                    ]
            raise ValueError(
                f"num_data_value({num_data_value}) is not found in SimulationResult."
            )

    def get_empi_dists(
        self, n_rep: int, num_data_index: int = None, num_data_value: int = None
    ) -> List[Tuple[int, np.ndarray]]:
        """get empi dists with specific parameter.
        Either num_data_index or num_data_value must be specified.

        Parameters
        ----------
        n_rep : int
            value of n_rep.
        num_data_index : int, optional
            index of num_data, by default None
        num_data_value : int, optional
            value of num_data, by default None

        Returns
        -------
        List[Tuple[int, np.ndarray]]
            empi dists.
            the index of list is "schecule_index".
            the tuple consists of (num_data, empi dist).

        Raises
        ------
        ValueError
            Either num_data_index or num_data_value must be specified. But both are None.
        ValueError
            Either num_data_index or num_data_value must be specified. But both are specified.
        ValueError
            num_data_value is not found in SimulationResult.
        """
        if num_data_index is None and num_data_value is None:
            raise ValueError(
                f"Either num_data_index or num_data_value must be specified. But both are None."
            )
        elif num_data_index is not None and num_data_value is not None:
            raise ValueError(
                f"Either num_data_index or num_data_value must be specified. But both are specified."
            )
        elif num_data_index is not None:
            return self.empi_dists_sequences[n_rep][num_data_index]
        else:
            for num_data_index, num in enumerate(self.simulation_setting.num_data):
                if num == num_data_value:
                    return self.empi_dists_sequences[n_rep][num_data_index]
            raise ValueError(
                f"num_data_value({num_data_value}) is not found in SimulationResult."
            )

    def to_pickle(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def to_dict(self) -> dict:
        result_dict = dict(
            test_setting_index=self.result_index["test_setting_index"],
            sample_index=self.result_index["sample_index"],
            case_index=self.result_index["case_index"],
            name=self.simulation_setting.name,
            total_result=self.check_result["total_result"],
        )

        def _make_warning_text(r):
            possibly_ok = r["detail"]["possibly_ok"]
            to_be_checked = r["detail"]["to_be_checked"]
            warning_text = ""
            if not possibly_ok:
                warning_text = f"Consistency: possibly_ok={possibly_ok}, to_be_checked={to_be_checked}"
            return warning_text

        check_result = {}
        warning_text = ""

        for r in self.check_result["results"]:
            check_result[r["name"]] = r["result"]
            if r["name"] == "Consistency":
                warning_text += _make_warning_text(r)

        result_dict.update(check_result)
        result_dict["warning"] = warning_text

        return result_dict

    def show_description(self) -> None:
        """show the description of schema of SimulationResult."""
        print(f"=== decsription of {type(self)} ===")
        self._show_description(self)

    @staticmethod
    def _show_description(obj, indent: int = 0) -> None:
        # if "obj" is StandardQTomographySimulationSetting or EstimationResult, it doesn't do recursive call
        if isinstance(obj, StandardQTomographySimulationSetting) or isinstance(
            obj, EstimationResult
        ):
            recursive_call = False
        else:
            recursive_call = True

        # show attributes of "obj"
        for attr in dir(obj):
            # don't show private functions and utility functions
            if attr.startswith("_") or attr.startswith("to_"):
                continue

            # show attribute
            attr_obj = getattr(obj, attr)
            print(" " * indent, attr, type(attr_obj))

            # do recursive call
            if (
                type(attr_obj) is dict
                or type(attr_obj).__name__ == "method"
                or isinstance(attr_obj, QTomography)
            ):
                # if "obj" is dict or method or QTomography, it doesn't do recursive call
                continue
            elif type(attr_obj) is list:
                if attr == "empi_dists_sequences":
                    print(" " * (indent + 2), "(multi-dimensional lists)")
                    print(" " * (indent + 2), "[n_rep]")
                    print(" " * (indent + 4), "[num_data]")
                    print(" " * (indent + 6), "[schecule_index]")
                elif attr == "estimation_results":
                    if len(attr_obj) == 0:
                        print(" " * (indent + 2), f"(list)")
                        print(" " * (indent + 2), "[n_rep]")
                    else:
                        print(" " * (indent + 2), f"(list of {type(attr_obj[0])})")
                        print(" " * (indent + 2), "[n_rep]")
                        SimulationResult._show_description(attr_obj[0], indent + 4)
            else:
                if recursive_call:
                    SimulationResult._show_description(attr_obj, indent + 2)


# common
def execute_simulation(
    qtomography: "StandardQTomography",
    simulation_setting: StandardQTomographySimulationSetting,
    seed_or_generator: Union[int, np.random.Generator] = None,
    is_computation_time_required: bool = True,
    is_detailed_results_required: bool = False,
) -> SimulationResult:
    org_sim_setting = simulation_setting.copy()
    if seed_or_generator is None:
        seed_or_generator = simulation_setting.seed_data

    simulation_result = generate_empi_dists_and_calc_estimate(
        qtomography=qtomography,
        true_object=simulation_setting.true_object,
        num_data=simulation_setting.num_data,
        estimator=simulation_setting.estimator,
        loss=simulation_setting.loss,
        loss_option=simulation_setting.loss_option,
        algo=simulation_setting.algo,
        algo_option=simulation_setting.algo_option,
        iteration=simulation_setting.n_rep,
        seed_or_generator=seed_or_generator,
        is_computation_time_required=is_computation_time_required,
        is_detailed_results_required=is_detailed_results_required,
    )
    simulation_result.simulation_setting = org_sim_setting
    return simulation_result


def _load_and_execute_estimation(
    empi_dists_path: str,
    qtomography: "StandardQTomography",
    empi_dists_seq: List[List[Tuple[int, np.ndarray]]],
    estimator=StandardQTomographyEstimator,
    loss: ProbabilityBasedLossFunction = None,
    loss_option: ProbabilityBasedLossFunctionOption = None,
    algo: MinimizationAlgorithm = None,
    algo_option: MinimizationAlgorithmOption = None,
    is_computation_time_required: bool = True,
    is_detailed_results_required: bool = False,
) -> EstimationResult:
    # Load
    with open(empi_dists_path, "rb") as f:
        empi_dists_path = pickle.load(f)
    estimation_result = _execute_estimation(
        qtomography=qtomography,
        empi_dists_seq=empi_dists_seq,
        estimator=estimator,
        loss=loss,
        loss_option=loss_option,
        algo=algo,
        algo_option=algo_option,
        is_computation_time_required=is_computation_time_required,
        is_detailed_results_required=is_detailed_results_required,
    )
    return estimation_result


def load_simulation_results(
    root_dir: str,
    test_setting_index: int,
    sample_index: int,
    case_index: int = None,
) -> list:
    print("Loading SimulationResult pickles ...")
    print(
        f"(test_setting_index, sample_index, case_index) = ({test_setting_index}, {sample_index}, {case_index})"
    )
    result_dir_path = Path(root_dir) / str(test_setting_index) / str(sample_index)
    simulation_results = []
    if case_index is not None:
        # load specific pickle file
        file_name = f"case_{case_index}_result.pickle"
        file_path = result_dir_path / file_name
        print(file_path)
        with open(file_path, "rb") as f:
            simulation_result = pickle.load(f)
        simulation_results.append(simulation_result)
    else:
        # load some pickle files
        file_paths = sorted(result_dir_path.glob("case_*_result.pickle"))
        for file_path in file_paths:
            print(file_path)
            with open(file_path, "rb") as f:
                simulation_result = pickle.load(f)
            simulation_results.append(simulation_result)
    print(
        f"Completed to load SimulationResult pickles. ({len(simulation_results)} files)"
    )
    return simulation_results


def load_test_setting(
    root_dir: str,
    test_setting_index: int,
) -> list:
    result_dir_path = Path(root_dir) / str(test_setting_index)
    # load test setting pickle file
    file_name = "test_setting.pickle"
    file_path = result_dir_path / file_name
    with open(file_path, "rb") as f:
        test_setting = pickle.load(f)
        return test_setting


def execute_estimation_with_saved_empi_dists_sequences(
    qtomography: "StandardQTomography",
    simulation_setting: StandardQTomographySimulationSetting,
    dir_path_empi_dists_sequences: Union[str, Path],
    n_jobs: int = 1,
    is_computation_time_required: bool = True,
    is_detailed_results_required: bool = False,
) -> SimulationResult:
    org_sim_setting = simulation_setting.copy()

    estimation_results = joblib.Parallel(n_jobs=n_jobs, verbose=2)(
        [
            joblib.delayed(_load_and_execute_estimation)(
                Path(dir_path_empi_dists_sequences)
                / f"empi_dists_{empi_dists_index}.pickle",
                qtomography,
                None,
                simulation_setting.estimator,
                simulation_setting.loss,
                simulation_setting.loss_option,
                simulation_setting.algo,
                simulation_setting.algo_option,
                is_computation_time_required=is_computation_time_required,
                is_detailed_results_required=is_detailed_results_required,
            )
            for empi_dists_index in range(simulation_setting.n_rep)
        ]
    )

    simulation_result = SimulationResult(
        qtomography=qtomography,
        empi_dists_sequences=dir_path_empi_dists_sequences,  # TODO
        estimation_results=estimation_results,
    )

    simulation_result.simulation_setting = org_sim_setting
    return simulation_result


def execute_estimation(
    qtomography: "StandardQTomography",
    simulation_setting: StandardQTomographySimulationSetting,
    empi_dists_sequences: List[List[Tuple[int, np.ndarray]]],
    n_jobs: int = 1,
    is_computation_time_required: bool = True,
    is_detailed_results_required: bool = False,
) -> SimulationResult:
    org_sim_setting = simulation_setting.copy()

    estimation_results = joblib.Parallel(n_jobs=n_jobs, verbose=2)(
        [
            joblib.delayed(_execute_estimation)(
                qtomography,
                empi_dists_seq,
                simulation_setting.estimator,
                simulation_setting.loss,
                simulation_setting.loss_option,
                simulation_setting.algo,
                simulation_setting.algo_option,
                is_computation_time_required,
                is_detailed_results_required,
            )
            for empi_dists_seq in empi_dists_sequences
        ]
    )

    simulation_result = SimulationResult(
        qtomography=qtomography,
        empi_dists_sequences=empi_dists_sequences,
        estimation_results=estimation_results,
    )

    simulation_result.simulation_setting = org_sim_setting
    return simulation_result


# common
def _generate_empi_dists_and_calc_estimate(
    qtomography: "StandardQTomography",
    true_object: QOperation,
    num_data: List[int],
    estimator=StandardQTomographyEstimator,
    loss: ProbabilityBasedLossFunction = None,
    loss_option: ProbabilityBasedLossFunctionOption = None,
    algo: MinimizationAlgorithm = None,
    algo_option: MinimizationAlgorithmOption = None,
    seed_or_generator: Union[int, np.random.Generator] = None,
    is_computation_time_required: bool = True,
    is_detailed_results_required: bool = False,
) -> Tuple[StandardQTomographyEstimationResult, List[List[Tuple[int, np.ndarray]]]]:
    empi_dists_seq = qtomography.generate_empi_dists_sequence(
        true_object, num_data, seed_or_generator=seed_or_generator
    )
    estimation_result = _execute_estimation(
        qtomography=qtomography,
        empi_dists_seq=empi_dists_seq,
        estimator=estimator,
        loss=loss,
        loss_option=loss_option,
        algo=algo,
        algo_option=algo_option,
        is_computation_time_required=is_computation_time_required,
        is_detailed_results_required=is_detailed_results_required,
    )
    return estimation_result, empi_dists_seq


def _execute_estimation(
    qtomography: "StandardQTomography",
    empi_dists_seq: List[List[Tuple[int, np.ndarray]]],
    estimator=StandardQTomographyEstimator,
    loss: ProbabilityBasedLossFunction = None,
    loss_option: ProbabilityBasedLossFunctionOption = None,
    algo: MinimizationAlgorithm = None,
    algo_option: MinimizationAlgorithmOption = None,
    is_computation_time_required: bool = True,
    is_detailed_results_required: bool = False,
) -> StandardQTomographyEstimationResult:
    if isinstance(estimator, LossMinimizationEstimator):
        estimation_result = estimator.calc_estimate_sequence(
            qtomography,
            empi_dists_seq,
            loss=loss,
            loss_option=loss_option,
            algo=algo,
            algo_option=algo_option,
            is_computation_time_required=is_computation_time_required,
            is_detailed_results_required=is_detailed_results_required,
        )
    else:
        estimation_result = estimator.calc_estimate_sequence(
            qtomography,
            empi_dists_seq,
            is_computation_time_required=is_computation_time_required,
        )
    return estimation_result


def re_estimate(
    test_setting: EstimatorTestSetting,
    simulation_result: SimulationResult,
    n_rep_index: int,
) -> StandardQTomographyEstimationResult:
    case_index = simulation_result.result_index["case_index"]
    empi_dists_seq = simulation_result.empi_dists_sequences[n_rep_index]

    sim_setting = simulation_result.simulation_setting
    qtomography = generate_qtomography(
        sim_setting,
        para=test_setting.parametrizations[case_index],
    )

    estimator = copy.deepcopy(simulation_result.simulation_setting.estimator)
    if isinstance(estimator, LossMinimizationEstimator):
        estimation_result = estimator.calc_estimate_sequence(
            qtomography,
            empi_dists_seq,
            loss=sim_setting.loss,
            loss_option=sim_setting.loss_option,
            algo=sim_setting.algo,
            algo_option=sim_setting.algo_option,
            is_computation_time_required=True,
        )
    else:
        estimation_result = estimator.calc_estimate_sequence(
            qtomography,
            empi_dists_seq,
            is_computation_time_required=True,
        )

    return estimation_result


def re_estimate_sequence(
    test_setting: EstimatorTestSetting, result: SimulationResult
) -> List[StandardQTomographyEstimationResult]:
    sim_setting = result.simulation_setting
    estimation_results = []
    for n_rep_index in range(sim_setting.n_rep):
        estimation_result = re_estimate(test_setting, result, n_rep_index)
        estimation_results.append(estimation_result)
    return estimation_results


def re_estimate_sequence_from_path(
    test_setting_path: Union[str, Path], result_path: Union[str, Path]
) -> List[StandardQTomographyEstimationResult]:
    with open(result_path, "rb") as f:
        result = pickle.load(f)

    with open(test_setting_path, "rb") as f:
        test_setting = pickle.load(f)
    estimation_results = re_estimate_sequence(test_setting, result)
    return estimation_results


def re_estimate_sequence_from_index(
    root_dir: str, test_setting_index: int, sample_index: int, case_index: int
) -> List[StandardQTomographyEstimationResult]:
    result_path = (
        Path(root_dir)
        / str(test_setting_index)
        / str(sample_index)
        / f"case_{case_index}_result.pickle"
    )
    test_setting_path = Path(root_dir) / str(test_setting_index) / "test_setting.pickle"
    estimation_results = re_estimate_sequence_from_path(test_setting_path, result_path)
    return estimation_results


# common
def generate_empi_dists_and_calc_estimate(
    qtomography: "StandardQTomography",
    true_object: QOperation,
    num_data: List[int],
    estimator: StandardQTomographyEstimator,
    loss: ProbabilityBasedLossFunction = None,
    loss_option: ProbabilityBasedLossFunctionOption = None,
    algo: MinimizationAlgorithm = None,
    algo_option: MinimizationAlgorithmOption = None,
    iteration: Optional[int] = None,
    seed_or_generator: Union[int, np.random.Generator] = None,
    is_computation_time_required: bool = True,
    is_detailed_results_required: bool = False,
) -> Union[Tuple[StandardQTomographyEstimationResult, list], SimulationResult,]:

    if iteration is None:
        estimation_result, empi_dists_seq = _generate_empi_dists_and_calc_estimate(
            qtomography,
            true_object,
            num_data,
            estimator,
            loss=loss,
            loss_option=loss_option,
            algo=algo,
            algo_option=algo_option,
            seed_or_generator=seed_or_generator,
            is_computation_time_required=is_computation_time_required,
            is_detailed_results_required=is_detailed_results_required,
        )
        return estimation_result, empi_dists_seq
    else:
        estimation_results = []
        empi_dists_sequences = []
        for _ in tqdm(range(iteration)):
            estimation_result, empi_dists_seq = _generate_empi_dists_and_calc_estimate(
                qtomography,
                true_object,
                num_data,
                estimator,
                loss=loss,
                loss_option=loss_option,
                algo=algo,
                algo_option=algo_option,
                seed_or_generator=seed_or_generator,
                is_computation_time_required=is_computation_time_required,
                is_detailed_results_required=is_detailed_results_required,
            )
            estimation_results.append(estimation_result)
            empi_dists_sequences.append(empi_dists_seq)

        simulation_result = SimulationResult(
            qtomography=qtomography,
            empi_dists_sequences=empi_dists_sequences,
            estimation_results=estimation_results,
        )

        return simulation_result


# Data Convert
def generate_qtomography(
    sim_setting: StandardQTomographySimulationSetting,
    para: bool,
    init_with_seed: bool = True,
) -> "StandardQTomography":
    true_object = sim_setting.true_object
    tester_objects = sim_setting.tester_objects
    eps_proj_physical = sim_setting.eps_proj_physical
    eps_truncate_imaginary_part = sim_setting.eps_truncate_imaginary_part
    if init_with_seed:
        seed_data = sim_setting.seed_data
    else:
        seed_data = None

    if type(true_object) == State:
        return StandardQst(
            tester_objects,
            on_para_eq_constraint=para,
            seed_data=seed_data,
            eps_proj_physical=eps_proj_physical,
            eps_truncate_imaginary_part=eps_truncate_imaginary_part,
        )
    if type(true_object) == Povm:
        return StandardPovmt(
            tester_objects,
            on_para_eq_constraint=para,
            seed_data=seed_data,
            eps_proj_physical=eps_proj_physical,
            eps_truncate_imaginary_part=eps_truncate_imaginary_part,
            num_outcomes=len(true_object.vecs),
        )
    if type(true_object) == Gate:
        states = [t for t in tester_objects if type(t) == State]
        povms = [t for t in tester_objects if type(t) == Povm]

        return StandardQpt(
            states=states,
            povms=povms,
            on_para_eq_constraint=para,
            seed_data=seed_data,
            eps_proj_physical=eps_proj_physical,
            eps_truncate_imaginary_part=eps_truncate_imaginary_part,
        )
    if type(true_object) == MProcess:
        states = [t for t in tester_objects if type(t) == State]
        povms = [t for t in tester_objects if type(t) == Povm]

        return StandardQmpt(
            states=states,
            povms=povms,
            num_outcomes=true_object.num_outcomes,
            on_para_eq_constraint=para,
            seed_data=seed_data,
            eps_proj_physical=eps_proj_physical,
            eps_truncate_imaginary_part=eps_truncate_imaginary_part,
        )

    message = f"type of sim_setting.true_object must be State, Povm, or Gate, not {type(true_object)}"
    print(f"{sim_setting.true_object}")
    raise TypeError(message)
