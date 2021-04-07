from typing import List
import copy
import time
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from quara.simulation import standard_qtomography_simulation_report as report

from quara.simulation.standard_qtomography_simulation_check import (
    StandardQTomographySimulationCheck,
)
from quara.simulation import standard_qtomography_simulation as sim
from quara.simulation.standard_qtomography_simulation import (
    TestSetting,
    Result,
    StandardQTomographySimulationSetting,
)


def execute_simulation_case_unit(
    test_setting,
    true_object,
    tester_objects,
    case_index: int,
    sample_index: int,
    test_setting_index: int,
    root_dir: str,
) -> Result:
    # Generate QTomographySimulationSetting
    sim_setting = test_setting.to_simulation_setting(true_object, tester_objects, case_index
    )
    print(f"Case {case_index}: {sim_setting.name}")

    def _copy_sim_setting(source):
        return StandardQTomographySimulationSetting(
            name=source.name,
            true_object=source.true_object,
            tester_objects=source.tester_objects,
            estimator=source.estimator,
            loss=copy.deepcopy(source.loss),
            loss_option=source.loss_option,
            algo=copy.deepcopy(source.algo),
            algo_option=source.algo_option,
            seed=source.seed,
            n_rep=source.n_rep,
            num_data=source.num_data,
            schedules=source.schedules,
        )

    org_sim_setting = _copy_sim_setting(sim_setting)

    # Generate QTomography
    qtomography = sim.generate_qtomography(
        sim_setting,
        para=test_setting.parametrizations[case_index],
        eps_proj_physical=1e-13,
    )

    # Execute
    estimation_results = sim.execute_simulation(
        qtomography=qtomography, simulation_setting=sim_setting
    )

    # Simulation Check
    sim_check = StandardQTomographySimulationCheck(sim_setting, estimation_results)
    check_result = sim_check.execute_all(show_detail=False, with_detail=True)

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

    # Store result
    result = Result(
        result_index=result_index,
        simulation_setting=org_sim_setting,
        estimation_results=estimation_results,
        check_result=check_result,
    )
    # Save
    write_result_case_unit(result, root_dir=root_dir)
    return result


def execute_simulation_sample_unit(
    test_setting,
    generation_settings,
    test_setting_index,
    sample_index,
    root_dir,
    pdf_mode: str = "only_ng",
) -> List[Result]:
    # Generate sample
    true_object = generation_settings.true_setting.generate()
    true_object = true_object[0] if type(true_object) == tuple else true_object
    tester_objects = [
        tester_setting.generate()
        for tester_setting in generation_settings.tester_settings
    ]
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
        )
        results.append(result)

    # Save
    write_result_sample_unit(results, root_dir=root_dir)

    # Save PDF
    if pdf_mode == "all":
        write_pdf_report(results, root_dir)
    elif pdf_mode == "only_ng":
        total_results = [r.check_result["total_result"] for r in results]
        print(f"total_result={np.all(total_results)}")
        if not np.all(total_results):
            write_pdf_report(results, root_dir)
    elif pdf_mode == "none":
        pass
    else:
        message = "`pdf_mode` must be 'all', 'only_ng', or 'none'."
        raise ValueError(message)

    return results


def execute_simulation_test_setting_unit(
    test_setting, test_setting_index, root_dir, pdf_mode: str = "only_ng"
) -> List[Result]:
    generation_settings = test_setting.to_generation_settings()
    n_sample = test_setting.n_sample
    results = []

    for sample_index in range(n_sample):
        sample_results = execute_simulation_sample_unit(
            test_setting,
            generation_settings,
            test_setting_index,
            sample_index,
            root_dir,
            pdf_mode=pdf_mode,
        )
        results += sample_results

    # Save
    write_result_test_setting_unit(results, root_dir)
    return results


def execute_simulation_test_settings(
    test_settings: List[TestSetting], root_dir: str, pdf_mode: str = "only_ng"
) -> List[Result]:
    all_results = []
    start = time.time()

    for test_setting_index, test_setting in enumerate(test_settings):
        path = Path(root_dir) / str(test_setting_index) / "test_setting.pickle"
        test_setting.to_pickle(path)
        print(f"Completed to write test_setting. {path}")
        test_results = execute_simulation_test_setting_unit(
            test_setting, test_setting_index, root_dir, pdf_mode="only_ng"
        )
        all_results += test_results

    # Save
    write_results(all_results, root_dir)

    elapsed_time = time.time() - start
    _print_summary(all_results, elapsed_time)

    return all_results


def _print_summary(results: List[Result], elapsed_time: float) -> None:
    result_dict_list = [result.to_dict() for result in results]
    result_df = pd.DataFrame(result_dict_list)
    ok_n = result_df[result_df["total_result"]].shape[0]
    ng_n = result_df[~result_df["total_result"]].shape[0]
    warning_n = result_df[
        result_df["total_result"] & result_df["warning"].isnull()
    ].shape[0]

    def _to_h_m_s(sec) -> tuple:
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        return h, m, s

    h, m, s = _to_h_m_s(elapsed_time)
    time_text = "{:.1f}s ".format(elapsed_time)
    time_text += f"({h}:{str(m).zfill(2)}:{str(s).zfill(2)})"

    start_red = "\033[31m"
    start_green = "\033[32m"
    start_yellow = "\033[33m"
    end_color = "\033[0m"

    summary_text = f"{start_yellow}=============={end_color} "
    summary_text += f"{start_green}OK: {ok_n} cases{end_color} ({start_yellow}{warning_n} warnings{end_color}), "
    summary_text += f"{start_red}NG: {ng_n} cases{end_color} "
    summary_text += f"{start_yellow} in {time_text}s=============={end_color}"

    print(summary_text)

# writer
def write_results(results: List[Result], dir_path: str) -> None:
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "check_result.csv"

    result_dict_list = [result.to_dict() for result in results]
    sample_result_df = pd.DataFrame(result_dict_list)
    sample_result_df.to_csv(path, index=None)

    print(f"Completed to write csv. {path}")

def write_result_sample_unit(results: List[Result], root_dir: str) -> None:
    test_setting_index = results[0].result_index["test_setting_index"]
    sample_index = results[0].result_index["sample_index"]
    dir_path = Path(root_dir) / str(test_setting_index) / str(sample_index)

    write_results(results, dir_path)


def write_result_test_setting_unit(results: List[Result], root_dir: str) -> None:
    test_setting_index = results[0].result_index["test_setting_index"]
    dir_path = Path(root_dir) / str(test_setting_index)

    write_results(results, dir_path)

def write_pdf_report(results: List[Result], root_dir: str) -> None:
    test_setting_index = results[0].result_index["test_setting_index"]
    sample_index = results[0].result_index["sample_index"]

    dir_path = Path(root_dir) / str(test_setting_index) / str(sample_index)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{test_setting_index}_{sample_index}_quara_report.pdf"

    estimation_results_list = [r.estimation_results for r in results]
    sim_settings = [r.simulation_setting for r in results]

    report.export_report(path, estimation_results_list, sim_settings)

def write_result_case_unit(result: Result, root_dir: str) -> None:
    test_setting_index = result.result_index["test_setting_index"]
    sample_index = result.result_index["sample_index"]
    case_index = result.result_index["case_index"]

    # Save all
    dir_path = Path(root_dir) / str(test_setting_index) / str(sample_index)
    path = dir_path / f"case_{case_index}_result.pickle"
    result.to_pickle(path)

    check_result = result.check_result
    path = dir_path / f"case_{case_index}_check_result.json"
    with open(path, "w") as f:
        json.dump(check_result, f, ensure_ascii=False, indent=4, separators=(",", ": "))
