from typing import List, Optional, Union
import copy
from collections import Counter
import dataclasses
import pickle
import time
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from quara.objects.qoperation import QOperation
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate

from quara.minimization_algorithm.minimization_algorithm import (
    MinimizationAlgorithm,
    MinimizationAlgorithmOption,
)
from quara.loss_function.probability_based_loss_function import (
    ProbabilityBasedLossFunction,
    ProbabilityBasedLossFunctionOption,
)
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
)
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.standard_qpt import StandardQpt
from quara.protocol.qtomography.standard.standard_povmt import StandardPovmt
from quara.protocol.qtomography.standard.standard_qtomography_estimator import (
    StandardQTomographyEstimator,
    StandardQTomographyEstimationResult,
)
from quara.simulation.generation_setting import QOperationGenerationSettings
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
        seed: int,
        n_rep: int,
        num_data: List[int],
        schedules: Union[str, List[List[int]]],
        eps_proj_physical: float,
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

        self.seed = seed
        self.n_rep = n_rep
        self.num_data = num_data
        self.eps_proj_physical = eps_proj_physical

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
        desc += f"\nEstimator: {self.estimator.__class__.__name__}"
        desc += f"\neps_proj_physical: {self.eps_proj_physical}"
        loss = None if self.loss is None else self.loss.__class__.__name__
        desc += f"\nLoss: {loss}"
        algo = None if self.algo is None else self.algo.__class__.__name__
        desc += f"\nAlgo: {algo}"
        return desc


@dataclasses.dataclass
class NoiseSetting:
    qoperation_base: Union[QOperation, str]
    method: str
    para: bool
    ids: List[int] = None

    def to_generation_setting(
        self, c_sys: "CompositeSystem"
    ) -> "QOperationGenerationSetting":
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
        )


@dataclasses.dataclass
class EstimatorTestSetting:
    true_object: NoiseSetting
    tester_objects: List[NoiseSetting]
    seed: int
    n_rep: int
    num_data: List[int]
    n_sample: int
    schedules: Union[str, List[List[int]]]
    case_names: List[str]
    estimators: List["Estimator"]
    eps_proj_physical_list: List[float]
    algo_list: List[tuple]
    loss_list: List[tuple]
    parametrizations: List[bool]
    c_sys: "CompositeSystem"

    def to_pickle(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def to_generation_settings(self) -> QOperationGenerationSettings:
        true_setting = self.true_object.to_generation_setting(self.c_sys)
        tester_settings = [
            setting.to_generation_setting(self.c_sys) for setting in self.tester_objects
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
            seed=self.seed,
            num_data=self.num_data,
            schedules=self.schedules,
            eps_proj_physical=self.eps_proj_physical_list[case_index],
        )


@dataclasses.dataclass
class SimulationResult:
    result_index: dict
    simulation_setting: StandardQTomographySimulationSetting
    estimation_results: List["EstimationResult"]
    check_result: dict

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


# common
def execute_simulation(
    qtomography: "StandardQTomography",
    simulation_setting: StandardQTomographySimulationSetting,
) -> List[StandardQTomographyEstimationResult]:
    estimation_results = generate_empi_dists_and_calc_estimate(
        qtomography=qtomography,
        true_object=simulation_setting.true_object,
        num_data=simulation_setting.num_data,
        estimator=simulation_setting.estimator,
        loss=simulation_setting.loss,
        loss_option=simulation_setting.loss_option,
        algo=simulation_setting.algo,
        algo_option=simulation_setting.algo_option,
        iteration=simulation_setting.n_rep,
    )
    return estimation_results


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
) -> StandardQTomographyEstimationResult:
    empi_dists_seq = qtomography.generate_empi_dists_sequence(true_object, num_data)

    if isinstance(estimator, LossMinimizationEstimator):
        result = estimator.calc_estimate_sequence(
            qtomography,
            empi_dists_seq,
            loss=loss,
            loss_option=loss_option,
            algo=algo,
            algo_option=algo_option,
            is_computation_time_required=True,
        )
    else:
        result = estimator.calc_estimate_sequence(
            qtomography,
            empi_dists_seq,
            is_computation_time_required=True,
        )
    return result


def re_estimate(
    test_setting: EstimatorTestSetting, result: SimulationResult, n_rep_index: int
) -> StandardQTomographyEstimationResult:
    case_index = result.result_index["case_index"]
    empi_dists_seq = result.estimation_results[n_rep_index].data

    sim_setting = result.simulation_setting
    qtomography = generate_qtomography(
        sim_setting,
        para=test_setting.parametrizations[case_index],
    )

    estimator = copy.deepcopy(result.simulation_setting.estimator)
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
) -> Union[
    StandardQTomographyEstimationResult,
    List[StandardQTomographyEstimationResult],
]:

    if iteration is None:
        result = _generate_empi_dists_and_calc_estimate(
            qtomography,
            true_object,
            num_data,
            estimator,
            loss=loss,
            loss_option=loss_option,
            algo=algo,
            algo_option=algo_option,
        )
        return result
    else:
        results = []
        for _ in tqdm(range(iteration)):
            result = _generate_empi_dists_and_calc_estimate(
                qtomography,
                true_object,
                num_data,
                estimator,
                loss=loss,
                loss_option=loss_option,
                algo=algo,
                algo_option=algo_option,
            )
            results.append(result)
        return results


# Data Convert
def generate_qtomography(
    sim_setting: StandardQTomographySimulationSetting, para: bool
) -> "StandardQTomography":
    true_object = sim_setting.true_object
    tester_objects = sim_setting.tester_objects
    eps_proj_physical = sim_setting.eps_proj_physical
    seed = sim_setting.seed

    if type(true_object) == State:
        return StandardQst(
            tester_objects,
            on_para_eq_constraint=para,
            seed=seed,
            eps_proj_physical=eps_proj_physical,
        )
    if type(true_object) == Povm:
        return StandardPovmt(
            tester_objects,
            on_para_eq_constraint=para,
            seed=seed,
            eps_proj_physical=eps_proj_physical,
            num_outcomes=len(true_object.vecs),
        )
    if type(true_object) == Gate:
        states = [t for t in tester_objects if type(t) == State]
        povms = [t for t in tester_objects if type(t) == Povm]

        return StandardQpt(
            states=states,
            povms=povms,
            on_para_eq_constraint=para,
            seed=seed,
            eps_proj_physical=eps_proj_physical,
        )
    message = f"type of sim_setting.true_object must be State, Povm, or Gate, not {type(true_object)}"
    print(f"{sim_setting.true_object}")
    raise TypeError(message)
