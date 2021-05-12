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
    EstimatorTestSetting,
    SimulationResult,
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
) -> SimulationResult:
    # Generate QTomographySimulationSetting
    sim_setting = test_setting.to_simulation_setting(
        true_object, tester_objects, case_index
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
            eps_proj_physical=source.eps_proj_physical,
        )

    org_sim_setting = _copy_sim_setting(sim_setting)

    # Generate QTomography
    qtomography = sim.generate_qtomography(
        sim_setting,
        para=test_setting.parametrizations[case_index],
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
    result = SimulationResult(
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
) -> List[SimulationResult]:
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
) -> List[SimulationResult]:
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
    test_settings: List[EstimatorTestSetting], root_dir: str, pdf_mode: str = "only_ng"
) -> List[SimulationResult]:
    all_results = []
    start = time.time()

    for test_setting_index, test_setting in enumerate(test_settings):
        path = Path(root_dir) / str(test_setting_index) / "test_setting.pickle"
        test_setting.to_pickle(path)
        print(f"Completed to write test_setting. {path}")
        test_results = execute_simulation_test_setting_unit(
            test_setting, test_setting_index, root_dir, pdf_mode=pdf_mode
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


def write_pdf_report(results: List[SimulationResult], root_dir: str) -> None:
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
    )


def write_result_case_unit(result: SimulationResult, root_dir: str) -> None:
    test_setting_index = result.result_index["test_setting_index"]
    sample_index = result.result_index["sample_index"]
    case_index = result.result_index["case_index"]

    # Save pickle
    dir_path = Path(root_dir) / str(test_setting_index) / str(sample_index)
    path = dir_path / f"case_{case_index}_result.pickle"
    result.to_pickle(path)

    # Save JSON
    # EstimationResult cannot be converted to JSON.
    # Therefore, alternative text is used.
    alternative_results = []
    for r in result.check_result["results"]:
        if r["name"] == "Consistency":
            alternative_text = "EstimationResult generated in the process of ConsistencyCheck is not dumped to json, check the pickle."
            new_r = copy.deepcopy(r)
            new_r["detail"]["estimation_result"] = alternative_text
            alternative_results.append(new_r)
        else:
            alternative_results.append(r)

    alternative_check_result = copy.deepcopy(result.check_result)
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
