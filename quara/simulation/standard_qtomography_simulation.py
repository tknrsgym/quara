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

# from quara.simulation import standard_qtomography_simulation_report as report

from quara.objects.povm import Povm
from quara.objects.gate import Gate
from quara.objects.state import State


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


@dataclasses.dataclass
class TestSetting:
    true_object: NoiseSetting
    tester_objects: List[NoiseSetting]
    seed: int
    n_rep: int
    num_data: List[int]
    n_sample: int
    schedules: Union[str, List[List[int]]]
    case_names: List[str]
    estimators: List["Estimator"]
    algo_list: List[tuple]
    loss_list: List[tuple]
    parametrizations: List[bool]
    c_sys: "CompositeSystem"


@dataclasses.dataclass
class Result:
    result_index: int
    simulation_setting: StandardQTomographySimulationSetting
    estimation_results: List["EstimationResult"]
    check_result: dict


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
            qtomography, empi_dists_seq, is_computation_time_required=True,
        )
    return result


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
    StandardQTomographyEstimationResult, List[StandardQTomographyEstimationResult],
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
def to_generation_setting(
    noise_setting: NoiseSetting, c_sys: "CompositeSystem"
) -> "QOperationGenerationSetting":
    # TODO: 他に良い方法がないか検討
    name2class_map = {
        "depolarized": DepolarizedQOperationGenerationSetting,
        "random_effective_lindbladian": RandomEffectiveLindbladianGenerationSetting,
    }

    if noise_setting.method in name2class_map:
        target_class = name2class_map[noise_setting.method]
    else:
        message = f"noise_setting.method='{noise_setting.method}' is not implemented."
        raise NotImplementedError(message)
    return target_class(
        qoperation_base=noise_setting.qoperation_base,
        c_sys=c_sys,
        **noise_setting.para,
    )


def to_generation_settings(test_setting: TestSetting) -> QOperationGenerationSettings:
    true_setting = to_generation_setting(test_setting.true_object, test_setting.c_sys)
    tester_settings = [
        to_generation_setting(setting, test_setting.c_sys)
        for setting in test_setting.tester_objects
    ]
    generation_settings = QOperationGenerationSettings(
        true_setting=true_setting, tester_settings=tester_settings
    )
    return generation_settings


def to_simulation_setting(
    test_setting: TestSetting,
    true_object: "QOperation",
    tester_objects: List["QOperation"],
    case_index: int,
) -> StandardQTomographySimulationSetting:
    return StandardQTomographySimulationSetting(
        name=test_setting.case_names[case_index],
        estimator=test_setting.estimators[case_index],
        loss=test_setting.loss_list[case_index][0],
        loss_option=test_setting.loss_list[case_index][1],
        algo=test_setting.algo_list[case_index][0],
        algo_option=test_setting.algo_list[case_index][1],
        true_object=true_object,
        tester_objects=tester_objects,
        n_rep=test_setting.n_rep,
        seed=test_setting.seed,
        num_data=test_setting.num_data,
        schedules=test_setting.schedules,
    )


def generate_qtomography(
    sim_setting: StandardQTomographySimulationSetting,
    para: bool,
    eps_proj_physical: float,
) -> "StandardQTomography":
    # TrueObjectに応じて、適切なQTomographyを生成する
    true_object = sim_setting.true_object
    tester_objects = sim_setting.tester_objects
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


def result2dict(result: Result) -> dict:
    result_dict = dict(
        test_setting_index=result.result_index["test_setting_index"],
        sample_index=result.result_index["sample_index"],
        case_index=result.result_index["case_index"],
        name=result.simulation_setting.name,
        total_result=result.check_result["total_result"],
    )

    def _make_warning_text(r):
        possibly_ok = r["detail"]["possibly_ok"]
        to_be_checked = r["detail"]["to_be_checked"]
        warning_text = ""
        if not possibly_ok:
            warning_text = (
                f"Consistency: possibly_ok={possibly_ok}, to_be_checked={to_be_checked}"
            )
        return warning_text

    check_result = {}
    warning_text = ""

    for r in result.check_result["results"]:
        check_result[r["name"]] = r["result"]
        if r["name"] == "Consistency":
            warning_text += _make_warning_text(r)

    result_dict.update(check_result)
    result_dict["warning"] = warning_text

    return result_dict


def write_setting(
    test_setting: TestSetting, root_dir: str, test_setting_index: int
) -> None:
    dir_path = Path(f"{root_dir}/{test_setting_index}/")
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"test_setting.pickle"
    with open(path, "wb") as f:
        pickle.dump(test_setting, f)

    print(f"Completed to write test_setting. {path}")


def write_results(results: List[Result], dir_path: str) -> None:
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "check_result.csv"

    result_dict_list = [result2dict(r) for r in results]
    sample_result_df = pd.DataFrame(result_dict_list)
    sample_result_df.to_csv(path, index=None)

    print(f"Completed to write csv. {path}")


# writer
def write_result_case_unit(result: Result, root_dir: str) -> None:
    test_setting_index = result.result_index["test_setting_index"]
    sample_index = result.result_index["sample_index"]
    case_index = result.result_index["case_index"]

    # Save all
    dir_path = Path(f"{root_dir}/{test_setting_index}/{sample_index}")
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"case_{case_index}_result.pickle"
    with open(path, "wb") as f:
        pickle.dump(result, f)

    check_result = result.check_result
    path = dir_path / f"case_{case_index}_check_result.json"
    with open(path, "w") as f:
        json.dump(check_result, f, ensure_ascii=False, indent=4, separators=(",", ": "))


def write_result_sample_unit(results: List[Result], root_dir: str) -> None:
    test_setting_index = results[0].result_index["test_setting_index"]
    sample_index = results[0].result_index["sample_index"]
    dir_path = Path(root_dir) / str(test_setting_index) / str(sample_index)

    write_results(results, dir_path)


def write_result_test_setting_unit(results: List[Result], root_dir: str) -> None:
    test_setting_index = results[0].result_index["test_setting_index"]
    dir_path = Path(root_dir) / str(test_setting_index)

    write_results(results, dir_path)

