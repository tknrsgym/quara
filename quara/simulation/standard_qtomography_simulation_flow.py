from itertools import starmap
from typing import List, Union
import copy
import time
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from quara.simulation import standard_qtomography_simulation_report as report

from quara.simulation.standard_qtomography_simulation_check import (
    StandardQTomographySimulationCheck,
)
from quara.simulation import standard_qtomography_simulation as sim
from quara.simulation.standard_qtomography_simulation import (
    EstimatorTestSetting,
    SimulationResult,
    StandardQTomographySimulationSetting,
)
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
)


def execute_simulation_case_unit(
    test_setting,
    true_object,
    tester_objects,
    case_index: int,
    sample_index: int,
    test_setting_index: int,
    root_dir: str,
    exec_sim_check: dict = None,
) -> SimulationResult:
    # Generate QTomographySimulationSetting
    sim_setting = test_setting.to_simulation_setting(
        true_object, tester_objects, case_index
    )
    print(f"Case {case_index}: {sim_setting.name}")

    org_sim_setting = sim_setting.copy()

    # Generate QTomography
    # Do not set the random number seed when initializing qtomography.
    # Use the random number stream later when generating the empirical distribution.
    qtomography = sim.generate_qtomography(
        sim_setting,
        para=test_setting.parametrizations[case_index],
        init_with_seed=False,
    )

    # Generate a random number stream to generate the empirical distribution.
    stream_data = np.random.RandomState(sim_setting.seed_data)

    # Execute
    sim_result = sim.execute_simulation(
        qtomography=qtomography,
        simulation_setting=sim_setting,
        seed_or_stream=stream_data,
    )

    # Simulation Check
    sim_check = StandardQTomographySimulationCheck(sim_result)
    check_result = sim_check.execute_all(
        show_detail=False, with_detail=True, exec_check=exec_sim_check
    )

    # Show result
    if not check_result["total_result"]:
        start_red = "\033[31m"
        end_color = "\033[0m"
        print(f"Total Result: {start_red}NG{end_color}")

    result_index = dict(
        test_setting_index=test_setting_index,
        sample_index=sample_index,
        case_index=case_index,
    )

    # Add to SimulationResult
    sim_result.simulation_setting = org_sim_setting
    sim_result.result_index = result_index
    sim_result.check_result = check_result

    # Save
    write_result_case_unit(sim_result, root_dir=root_dir)
    return sim_result


def execute_simulation_sample_unit(
    test_setting,
    generation_settings,
    test_setting_index,
    sample_index,
    root_dir,
    pdf_mode: str = "only_ng",
    stream_qoperation: Union[int, np.random.RandomState] = None,
    exec_sim_check: dict = None,
) -> List[SimulationResult]:
    # Generate sample
    _f = generation_settings.true_setting.generate
    if "seed_or_stream" in _f.__code__.co_varnames[: _f.__code__.co_argcount]:
        true_object = generation_settings.true_setting.generate(
            seed_or_stream=stream_qoperation
        )
        tester_objects = [
            tester_setting.generate(stream_qoperation)
            for tester_setting in generation_settings.tester_settings
        ]
    else:
        true_object = generation_settings.true_setting.generate()
        tester_objects = [
            tester_setting.generate()
            for tester_setting in generation_settings.tester_settings
        ]

    true_object = true_object[0] if type(true_object) == tuple else true_object
    tester_objects = [
        tester[0] if type(tester) == tuple else tester for tester in tester_objects
    ]
    results = []
    case_n = len(test_setting.case_names)

    for case_index in range(case_n):
        result = execute_simulation_case_unit(
            test_setting,
            true_object=true_object,
            tester_objects=tester_objects,
            case_index=case_index,
            sample_index=sample_index,
            test_setting_index=test_setting_index,
            root_dir=root_dir,
            exec_sim_check=exec_sim_check,
        )
        results.append(result)

    # Save
    write_result_sample_unit(results, root_dir=root_dir)

    # Save PDF
    if pdf_mode == "all":
        write_pdf_report(results, root_dir, display_items=exec_sim_check)
    elif pdf_mode == "only_ng":
        total_results = [r.check_result["total_result"] for r in results]
        print(f"total_result={np.all(total_results)}")
        if not np.all(total_results):
            write_pdf_report(results, root_dir, display_items=exec_sim_check)
    elif pdf_mode == "none":
        pass
    else:
        message = "`pdf_mode` must be 'all', 'only_ng', or 'none'."
        raise ValueError(message)

    return results


def execute_simulation_test_setting_unit(
    test_setting,
    test_setting_index,
    root_dir,
    exec_sim_check: dict = None,
    pdf_mode: str = "only_ng",
) -> List[SimulationResult]:
    generation_settings = test_setting.to_generation_settings()
    n_sample = test_setting.n_sample
    results = []

    stream_qoperation = np.random.RandomState(test_setting.seed_qoperation)

    for sample_index in range(n_sample):
        sample_results = execute_simulation_sample_unit(
            test_setting,
            generation_settings,
            test_setting_index,
            sample_index,
            root_dir,
            pdf_mode=pdf_mode,
            stream_qoperation=stream_qoperation,
            exec_sim_check=exec_sim_check,
        )
        results += sample_results

    # Save
    write_result_test_setting_unit(results, root_dir)
    return results


def execute_simulation_test_settings(
    test_settings: List[EstimatorTestSetting],
    root_dir: str,
    pdf_mode: str = "only_ng",
    exec_sim_check: dict = None,
) -> List[SimulationResult]:
    all_results = []
    start = time.time()

    for test_setting_index, test_setting in enumerate(test_settings):
        path = Path(root_dir) / str(test_setting_index) / "test_setting.pickle"
        test_setting.to_pickle(path)
        print(f"Completed to write test_setting. {path}")
        test_results = execute_simulation_test_setting_unit(
            test_setting,
            test_setting_index,
            root_dir,
            exec_sim_check=exec_sim_check,
            pdf_mode=pdf_mode,
        )
        all_results += test_results

    # Save
    write_results(all_results, root_dir)

    elapsed_time = time.time() - start
    _print_summary(all_results, elapsed_time)

    return all_results


def _print_summary(results: List[SimulationResult], elapsed_time: float) -> None:
    def _to_dict(result: SimulationResult) -> dict:
        check_result = {}
        for r in result.check_result["results"]:
            if r["name"] == "Consistency":
                check_result["Consistency_possibly_ok"] = r["detail"]["possibly_ok"]
                check_result["Consistency_to_be_checked"] = r["detail"]["to_be_checked"]
            else:
                check_result[r["name"]] = r["result"]
        return check_result

    result_dict_list = [_to_dict(result) for result in results]
    df = pd.DataFrame(result_dict_list)

    start_red = "\033[31m"
    start_green = "\033[32m"
    start_yellow = "\033[33m"
    end_color = "\033[0m"

    result_lines = []

    for col in df.columns:
        if col == "Consistency_to_be_checked":
            continue
        ok_n = df[df[col]].shape[0]
        ng_n = df[~df[col]].shape[0]

        if col == "Consistency_possibly_ok":
            result_line = f"Consistency:\n"
            result_line += f"{start_green}OK: {ok_n} cases{end_color}, {start_red}NG: {ng_n} cases{end_color}\n"
            to_be_checked_n = df[df["Consistency_to_be_checked"]].shape[0]
            result_line += f"You need to check report: {to_be_checked_n} cases\n"
        else:
            result_line = f"{col}:\n"
            result_line += f"{start_green}OK: {ok_n} cases{end_color}, {start_red}NG: {ng_n} cases{end_color}\n"

        result_lines.append(result_line)

    def _to_h_m_s(sec) -> tuple:
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        return h, m, s

    h, m, s = _to_h_m_s(elapsed_time)
    time_text = "{:.1f}s ".format(elapsed_time)
    time_text += f"({h}:{str(m).zfill(2)}:{str(s).zfill(2)})"

    summary_text = (
        f"{start_yellow}=============== Summary ================={end_color}\n"
    )
    summary_text += "\n".join(result_lines)
    summary_text += f"\nTime:\n{time_text}\n"
    summary_text += (
        f"{start_yellow}========================================={end_color}"
    )

    print(summary_text)


# re-estimate
def re_estimate_case_unit(
    input_root_dir: str,
    case_index: int,
    sample_index: int,
    test_setting_index: int,
    output_root_dir: str,
    test_setting=None,
    exec_sim_check: dict = None,
) -> SimulationResult:

    if test_setting is None:
        test_setting_pickle_path = (
            f"{input_root_dir}/{test_setting_index}/test_setting.pickle"
        )
        with open(test_setting_pickle_path, "rb") as f:
            test_setting = pickle.load(f)

    # Load pickle
    simulation_result_path = (
        Path(input_root_dir)
        / str(test_setting_index)
        / str(sample_index)
        / f"case_{case_index}_result.pickle"
    )
    with open(simulation_result_path, "rb") as f:
        source_sim_result = pickle.load(f)

    sim_setting = source_sim_result.simulation_setting
    empi_dists_seqences = source_sim_result.empi_dists_sequences
    print(f"Case {case_index}: {sim_setting.name}")
    org_sim_setting = sim_setting.copy()

    # Generate QTomography
    qtomography = sim.generate_qtomography(
        sim_setting,
        para=test_setting.parametrizations[case_index],
    )

    # Re-estimate
    estimation_results = []
    for n_rep_index in range(sim_setting.n_rep):
        empi_dists_seq = source_sim_result.empi_dists_sequences[n_rep_index]
        estimator = copy.deepcopy(source_sim_result.simulation_setting.estimator)

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

        estimation_results.append(estimation_result)

    result_index = dict(
        test_setting_index=test_setting_index,
        sample_index=sample_index,
        case_index=case_index,
    )

    re_estimated_sim_result = SimulationResult(
        estimation_results=estimation_results,
        empi_dists_sequences=empi_dists_seqences,
        qtomography=qtomography,
        simulation_setting=sim_setting,
        result_index=result_index,
    )

    # Simulation Check
    sim_check = StandardQTomographySimulationCheck(re_estimated_sim_result)
    check_result = sim_check.execute_all(
        show_detail=False, with_detail=True, exec_check=exec_sim_check
    )

    # Show result
    if not check_result["total_result"]:
        start_red = "\033[31m"
        end_color = "\033[0m"
        print(f"Total Result: {start_red}NG{end_color}")

    re_estimated_sim_result.simulation_setting = org_sim_setting
    re_estimated_sim_result.check_result = check_result
    # Save
    write_result_case_unit(re_estimated_sim_result, root_dir=output_root_dir)
    return re_estimated_sim_result


def re_estimate_sample_unit(
    test_setting_index,
    sample_index,
    output_root_dir,
    input_root_dir,
    exec_sim_check: dict = None,
    pdf_mode: str = "only_ng",
) -> List[SimulationResult]:

    # Load test setting pickle
    test_setting_pickle_path = (
        f"{input_root_dir}/{test_setting_index}/test_setting.pickle"
    )
    with open(test_setting_pickle_path, "rb") as f:
        test_setting = pickle.load(f)

    case_n = len(test_setting.case_names)
    results = []
    for case_index in range(case_n):
        result = re_estimate_case_unit(
            test_setting=test_setting,
            case_index=case_index,
            sample_index=sample_index,
            test_setting_index=test_setting_index,
            input_root_dir=input_root_dir,
            output_root_dir=output_root_dir,
            exec_sim_check=exec_sim_check,
        )
        results.append(result)

    # Save
    write_result_sample_unit(results, root_dir=output_root_dir)

    # Save PDF
    if pdf_mode == "all":
        write_pdf_report(results, output_root_dir, display_items=exec_sim_check)
    elif pdf_mode == "only_ng":
        total_results = [r.check_result["total_result"] for r in results]
        print(f"total_result={np.all(total_results)}")
        if not np.all(total_results):
            write_pdf_report(results, output_root_dir, display_items=exec_sim_check)
    elif pdf_mode == "none":
        pass
    else:
        message = "`pdf_mode` must be 'all', 'only_ng', or 'none'."
        raise ValueError(message)

    return results


def re_estimate_test_setting_unit(
    test_setting_index,
    output_root_dir,
    input_root_dir,
    exec_sim_check: dict = None,
    pdf_mode: str = "only_ng",
) -> List[SimulationResult]:
    # Load test setting from pickle
    test_setting_pickle_path = (
        f"{input_root_dir}/{test_setting_index}/test_setting.pickle"
    )
    with open(test_setting_pickle_path, "rb") as f:
        test_setting = pickle.load(f)

    n_sample = test_setting.n_sample

    results = []

    for sample_index in range(n_sample):
        sample_results = re_estimate_sample_unit(
            test_setting_index=test_setting_index,
            sample_index=sample_index,
            input_root_dir=input_root_dir,
            output_root_dir=output_root_dir,
            exec_sim_check=exec_sim_check,
            pdf_mode=pdf_mode,
        )
        results += sample_results

    # Save
    write_result_test_setting_unit(results, output_root_dir)
    return results


def re_estimate_test_settings(
    input_root_dir: str,
    output_root_dir: str,
    pdf_mode: str,
    exec_sim_check: dict = None,
) -> List[SimulationResult]:
    # Load All Test Setting
    test_setting_pickle_paths = sorted(
        Path(input_root_dir).glob("*/test_setting.pickle")
    )
    all_results = []
    for path in test_setting_pickle_paths:
        test_setting_index = path.parent.name  # directory name is test_setting_index
        test_results = re_estimate_test_setting_unit(
            test_setting_index,
            input_root_dir=input_root_dir,
            output_root_dir=output_root_dir,
            exec_sim_check=exec_sim_check,
            pdf_mode=pdf_mode,
        )
        all_results += test_results
    return all_results


# writer
def write_results(results: List[SimulationResult], dir_path: str) -> None:
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "check_result.csv"

    result_dict_list = [result.to_dict() for result in results]
    sample_result_df = pd.DataFrame(result_dict_list)
    sample_result_df.to_csv(path, index=None)

    print(f"Completed to write csv. {path}")


def write_result_sample_unit(results: List[SimulationResult], root_dir: str) -> None:
    test_setting_index = results[0].result_index["test_setting_index"]
    sample_index = results[0].result_index["sample_index"]
    dir_path = Path(root_dir) / str(test_setting_index) / str(sample_index)

    write_results(results, dir_path)


def write_result_test_setting_unit(
    results: List[SimulationResult], root_dir: str
) -> None:
    test_setting_index = results[0].result_index["test_setting_index"]
    dir_path = Path(root_dir) / str(test_setting_index)

    write_results(results, dir_path)


def write_pdf_report(
    results: List[SimulationResult], root_dir: str, display_items: dict = None
) -> None:
    test_setting_index = results[0].result_index["test_setting_index"]
    sample_index = results[0].result_index["sample_index"]

    dir_path = Path(root_dir) / str(test_setting_index) / str(sample_index)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{test_setting_index}_{sample_index}_quara_report.pdf"

    report.export_report_from_index(
        input_root_dir=root_dir,
        test_setting_index=test_setting_index,
        sample_index=sample_index,
        output_path=path,
        display_items=display_items,
    )


def write_result_case_unit(sim_result: SimulationResult, root_dir: str) -> None:
    test_setting_index = sim_result.result_index["test_setting_index"]
    sample_index = sim_result.result_index["sample_index"]
    case_index = sim_result.result_index["case_index"]

    # Save pickle
    dir_path = Path(root_dir) / str(test_setting_index) / str(sample_index)
    path = dir_path / f"case_{case_index}_result.pickle"
    sim_result.to_pickle(path)

    # Save JSON
    # EstimationResult cannot be converted to JSON.
    # Therefore, alternative text is used.
    alternative_results = []
    for r in sim_result.check_result["results"]:
        if r["name"] == "Consistency":
            alternative_text = "EstimationResult generated in the process of ConsistencyCheck is not dumped to json, check the pickle."
            new_r = copy.deepcopy(r)
            new_r["detail"]["estimation_result"] = alternative_text
            alternative_results.append(new_r)
        else:
            alternative_results.append(r)

    alternative_check_result = copy.deepcopy(sim_result.check_result)
    alternative_check_result["results"] = alternative_results

    path = dir_path / f"case_{case_index}_check_result.json"
    with open(path, "w") as f:
        json.dump(
            alternative_check_result,
            f,
            ensure_ascii=False,
            indent=4,
            separators=(",", ": "),
        )
