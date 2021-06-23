from quara.protocol import qtomography
import time
from typing import Callable, List, Optional, Union
import copy
from collections import namedtuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from tqdm import tqdm

from quara.objects.povm import Povm
from quara.objects.gate import Gate
from quara.objects.qoperation import QOperation
from quara.objects.state import State
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.standard_qtomography_estimator import (
    StandardQTomographyEstimator,
    StandardQTomographyEstimationResult,
)
from quara.simulation.standard_qtomography_simulation import (
    StandardQTomographySimulationSetting,
    SimulationResult,
)
from quara.utils import matrix_util
from quara.protocol.qtomography.estimator import EstimationResult


def calc_mse_general_norm(
    xs: List[np.ndarray],
    y: np.ndarray,
    norm_function: Callable[[np.ndarray, np.ndarray], np.float64],
) -> np.float64:
    """calculates mse(mean squared error) of ``xs`` and ``y`` according to ``norm_function``.

    Parameters
    ----------
    xs : np.ndarray
        sample values.
    y : np.ndarray
        true value.
    norm_function : Callable[[np.ndarray, np.ndarray], np.float64]
        norm function.

    Returns
    -------
    np.float64
        :math:`\\text{mse} = \\frac{1}{len(x)} \\sum_i \\text{norm_function}(x_i, y)^2`
    """
    norms = []
    for x in xs:
        norm = norm_function(x, y) ** 2
        norms.append(norm)

    mse = np.mean(norms, dtype=np.float64)
    return mse


def calc_covariance_matrix_of_prob_dist(
    prob_dist: np.ndarray, data_num: int
) -> np.ndarray:
    """calculates covariance matrix of probability distribution.

    Parameters
    ----------
    prob_dist : np.ndarray
        probability distribution.
    data_num : int
        number of data.

    Returns
    -------
    np.ndarray
        :math:`\\text{covariance matrix} = \\frac{1}{len(x)} (diag(p) - p \\cdot p^T)`, where N is ``data_num`` and p is ``prob_dist``.
    """
    matrix = np.diag(prob_dist) - np.array([prob_dist]).T @ np.array([prob_dist])
    return matrix / data_num


def calc_covariance_matrix_of_prob_dists(
    prob_dists: List[np.ndarray], data_num: int
) -> np.ndarray:
    """calculates covariance matrix of probability distributions(= direct product of each covariance matrix of probability distribution).

    Parameters
    ----------
    prob_dists : List[np.ndarray]
        probability distributions.
    data_num : int
        number of data.

    Returns
    -------
    np.ndarray
        direct product of each covariance matrix = :math:`\\oplus_j V(p^j)`, where :math:`V(p)` is covariance matrix of p.
    """
    # calculate diagonal blocks
    diag_blocks = [
        calc_covariance_matrix_of_prob_dist(prob_dist, data_num)
        for prob_dist in prob_dists
    ]

    # calculate direct product of each covariance matrix of probability distribution
    matrix_size = np.sum([len(prob_dist) for prob_dist in prob_dists])
    matrix = np.zeros((matrix_size, matrix_size))
    index = 0
    for diag in diag_blocks:
        size = diag.shape[0]
        matrix[index : index + size, index : index + size] = diag
        index += size

    return matrix


def _calc_mse_linear_analytical_mode_qoperation(
    xs: List[QOperation], ys: List[QOperation], with_std: bool = True
) -> np.float64:
    points = []
    for x, y in zip(xs, ys):
        x_vec = x.to_stacked_vector()
        y_vec = y.to_stacked_vector()
        point = np.vdot(x_vec - y_vec, x_vec - y_vec)
        points.append(point)

    mse = np.mean(points, dtype=np.float64)
    if with_std:
        std = np.std(points, dtype=np.float64, ddof=1)
        return mse, std
    else:
        return mse


def _calc_mse_linear_analytical_mode_var(
    xs: List[QOperation], ys: List[QOperation]
) -> np.float64:
    raise NotImplementedError()


# common
# statistical quantity
def calc_mse_qoperations(
    xs: List[QOperation],
    ys: List[QOperation],
    mode: str = "qoperation",
    with_std: bool = True,
) -> np.float64:
    if mode == "qoperation":
        return _calc_mse_linear_analytical_mode_qoperation(xs, ys, with_std=with_std)
    elif mode == "var":
        return _calc_mse_linear_analytical_mode_var(xs, ys)
    else:
        error_message = "​The argument `mode` must be `qoperation` or `var`"
        raise ValueError(error_message)


# common(StandardQTomography)
def convert_to_series(
    results: List[StandardQTomographyEstimationResult], true_object: QOperation
):
    # calc mse
    results_tmp = [result.estimated_qoperation_sequence for result in results]
    results_tmp = [
        list(qoperation_sequence) for qoperation_sequence in zip(*results_tmp)
    ]

    mses = [
        calc_mse_qoperations(
            qoperation_sequence, [true_object] * len(qoperation_sequence)
        )
        for qoperation_sequence in results_tmp
    ]

    stds = [mse[1] for mse in mses]
    mses = [mse[0] for mse in mses]

    # convert to computation time series
    comp_time_tmp = [result.computation_times for result in results]
    comp_time = [list(comp_time) for comp_time in zip(*comp_time_tmp)]

    return mses, stds, comp_time


# StandardQst, StandardQTomographyEstimator
def calc_estimate_with_average_comp_time(
    tester_povms: List[Povm],
    true_object: State,
    num_data: List[int],
    iteration: int,
    estimator=StandardQTomographyEstimator,
    on_para_eq_constraint: bool = True,
) -> StandardQTomographyEstimationResult:
    qst = StandardQst(tester_povms, on_para_eq_constraint=on_para_eq_constraint)

    # generate empi dists
    empi_dists_seqs = []
    for ite in range(iteration):
        seeds = [ite] * len(num_data)
        empi_dists_seq = qst.generate_empi_dists_sequence(true_object, num_data, seeds)
        empi_dists_seqs.append(empi_dists_seq)

    # transpose
    empi_dists_seqs = [list(empi_dists_seq) for empi_dists_seq in zip(*empi_dists_seqs)]

    #  calc estimate
    results = []
    for ite, empi_dists_seq in enumerate(empi_dists_seqs):
        start_time = time.time()
        result = estimator.calc_estimate_sequence(qst, empi_dists_seq)
        comp_time = time.time() - start_time

        info = {
            "iteration": ite + 1,
            "data": empi_dists_seq,
            "estimated_var_sequence": result.estimated_var_sequence,
            "computation_times": comp_time,
        }
        result._computation_times = [comp_time]
        results.append(result)

    return results


# common(show log-log scatter?)
def show_mse(num_data: List[int], mses: List[float], title: str = "Mean squared error"):
    trace = go.Scatter(x=num_data, y=mses, mode="lines+markers")
    data = [trace]
    layout = go.Layout(
        title=title,
        xaxis_title_text="Number of data",
        yaxis_title_text="Mean squared error of estimates and true",
        xaxis_type="log",
        yaxis_type="log",
    )
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def make_mses_graph(
    num_data: List[int],
    mses: List[List[float]],
    error_bar_values_list: List[List[float]] = None,
    title: str = "Mean squared error",
    additional_title_text: str = "",
    names: Optional[List[str]] = None,
    yaxis_title_text: str = "Mean squared error of estimates and true",
) -> List["Figure"]:
    if not names:
        names = [f"data_{i}" for i in range(len(mses))]
    data = []

    for i, mse in enumerate(mses):
        error_y = dict(visible=False)
        if error_bar_values_list and i < len(error_bar_values_list):
            error_y = dict(
                type="data",  # value of error bar given in data coordinates
                array=error_bar_values_list[i],
                visible=True,
            )
        trace = go.Scatter(
            x=num_data,
            y=mse,
            mode="lines+markers",
            name=names[i],
            error_y=error_y,
        )
        data.append(trace)
    if additional_title_text:
        title = f"{title}<br>{additional_title_text}"
    layout = go.Layout(
        title=title,
        xaxis_title_text="Number of data",
        yaxis_title_text=yaxis_title_text,
        xaxis_type="log",
        yaxis_type="log",
    )
    fig = go.Figure(data=data, layout=layout)
    return fig


def _recreate_qoperation(
    source_qoperation: "Qoperation", on_para_eq_constraint: bool
) -> "Qoperation":
    # Make QOperation
    true_object = copy.copy(source_qoperation)
    if type(true_object) == State:
        true_object_copied = State(
            vec=true_object.vec,
            c_sys=true_object.composite_system,
            on_para_eq_constraint=on_para_eq_constraint,
        )
    elif type(true_object) == Povm:
        true_object_copied = Povm(
            vecs=true_object.vecs,
            c_sys=true_object.composite_system,
            on_para_eq_constraint=on_para_eq_constraint,
        )
    elif type(true_object) == Gate:
        true_object_copied = Gate(
            hs=true_object.hs,
            c_sys=true_object.composite_system,
            on_para_eq_constraint=on_para_eq_constraint,
        )
    else:
        print(type(true_object))
        message = f"true_object must be State, Povm, or Gate, not {type(true_object)}"
        raise TypeError(message)
    return true_object_copied


def make_mses_graph_estimation_results(
    estimation_results_list: List["LinearEstimationResult"],
    case_names: List[str],
    true_object,
    num_data: List[int],
    qtomography_list: List["StandardQTomography"],
    title: str = "Mean squared error",
    additional_title_text: str = "",
    show_analytical_results: bool = False,
    estimator_list: list = None,
) -> "Figure":
    mses_list = []
    error_bar_values_list = []
    n_rep = len(estimation_results_list[0])
    display_case_names = case_names[:]
    for estimation_results in estimation_results_list:
        mses, sds, _ = convert_to_series(estimation_results, true_object)
        mses_list.append(mses)
        error_bar_values = [sigma / np.sqrt(n_rep) for sigma in sds]
        error_bar_values_list.append(error_bar_values)

    # calc analytical result
    if show_analytical_results:
        if not estimator_list:
            raise ValueError(
                "'show_analytical_results' is set True. But, 'estimator_list' is None."
            )

        (
            analytical_mses,
            analytical_case_names,
            _,
            _,
        ) = _make_data_for_graphs_mses_analytical(
            estimation_results_list,
            num_data,
            true_object,
            estimator_list,
            qtomography_list,
        )
        mses_list += analytical_mses
        display_case_names += analytical_case_names
    fig = make_mses_graph(
        num_data,
        mses_list,
        names=display_case_names,
        title=title,
        additional_title_text=additional_title_text,
        error_bar_values_list=error_bar_values_list,
    )
    return fig


def _make_data_for_graphs_mses_analytical(
    estimation_results_list: List["EstimationResult"],
    num_data: List[int],
    true_object,
    estimator_list: List[Union["Estimator", str]],
    qtomography_list,
):
    if len(estimation_results_list) != len(estimator_list):
        message = "`estimation_results_list` and `estimator_list` lengths do not match"
        raise ValueError(message)

    mses_list = []

    qtomo_type_dict = {}
    QTomoType = namedtuple(
        "QTomoType", ["qtomography_name", "on_para_eq_constraint", "estimator_name"]
    )

    for i, qtomo in enumerate(qtomography_list):
        if type(estimator_list[i]) == str:
            estimator_name = estimator_list[i]
        else:
            estimator_name = estimator_list[i].__class__.__name__
        qtomo_type = QTomoType(
            qtomo.__class__.__name__,
            qtomo.on_para_eq_constraint,
            estimator_name,
        )
        qtomo_type_dict[qtomo_type] = qtomo

    estimator_table = {
        "LinearEstimator": "calc_mse_linear_analytical",
        "LossMinimizationEstimator": "calc_cramer_rao_bound",
    }

    display_case_names = []
    short_names = []
    parameters = []

    for qtomo_type, qtomo in qtomo_type_dict.items():
        parameter = qtomo_type.on_para_eq_constraint
        estimator_name = qtomo_type.estimator_name
        if estimator_name not in estimator_table:
            continue
        method_name = estimator_table[estimator_name]

        true_object_copied = _recreate_qoperation(
            true_object, on_para_eq_constraint=parameter
        )
        true_mses = []
        for num in num_data:
            method = eval(f"qtomo.{method_name}")
            if method_name == "calc_mse_linear_analytical":
                true_mse = method(true_object_copied, [num] * qtomo.num_schedules)
            elif method_name == "calc_cramer_rao_bound":
                true_mse = method(true_object_copied, num, [num] * qtomo.num_schedules)
            true_mses.append(true_mse)

        mses_list.append(true_mses)
        short_name = estimator_name.replace("Estimator", "")
        if short_name == "LossMinimization":
            short_name = "Cramer-Rao bound"
        display_case_names.append(f"Analytical result ({short_name}, {parameter})")
        short_names.append(short_name)
        parameters.append(parameter)

    return mses_list, display_case_names, short_names, parameters


def make_mses_graph_analytical(
    estimation_results_list: List["LinearEstimationResult"],
    true_object,
    estimator_list: list,
    num_data: List[int],
    qtomography_list: List["StandardQTomography"],
) -> "Figure":

    (
        mses_list,
        display_case_names,
        short_names,
        parameters,
    ) = _make_data_for_graphs_mses_analytical(
        estimation_results_list, num_data, true_object, estimator_list, qtomography_list
    )

    figs = []

    # make graphs by estimator
    data_dict = {}

    for i, estimator_name in enumerate(short_names):
        if estimator_name in data_dict:
            data_dict[estimator_name]["mses"].append(mses_list[i])
            data_dict[estimator_name]["display_case_names"].append(
                display_case_names[i]
            )
            data_dict[estimator_name]["parameters"].append(parameters[i])
        else:
            data_dict[estimator_name] = dict(
                mses=[mses_list[i]],
                display_case_names=[display_case_names[i]],
                parameters=[parameters[i]],
            )

    for key, target_dict in data_dict.items():
        if key == "Cramer-Rao bound":
            fig = make_mses_graph(
                num_data,
                target_dict["mses"],
                names=target_dict["display_case_names"],
                additional_title_text=f"Analytical result<br>{key}",
            )
        else:
            fig = make_mses_graph(
                num_data,
                target_dict["mses"],
                names=target_dict["display_case_names"],
                additional_title_text=f"Analytical result<br>estimator={key}",
            )
        figs.append(fig)

    # make graphs by parameter
    data_dict = {}

    for i, parameter in enumerate(parameters):
        if parameter in data_dict:
            data_dict[parameter]["mses"].append(mses_list[i])
            data_dict[parameter]["display_case_names"].append(display_case_names[i])
            data_dict[parameter]["parameters"].append(parameters[i])
        else:
            data_dict[parameter] = dict(
                mses=[mses_list[i]],
                display_case_names=[display_case_names[i]],
                parameters=[parameters[i]],
            )

    for key, target_dict in data_dict.items():
        fig = make_mses_graph(
            num_data,
            target_dict["mses"],
            names=target_dict["display_case_names"],
            additional_title_text=f"Analytical result<br>parametorization={key}",
        )
        figs.append(fig)

    return figs


def make_mses_graphs_estimator(
    estimation_results_list: List["EstimationResult"],
    simulation_settings: List["StandardQTomographySimulationSetting"],
    true_object,
    qtomography_list: List["StandardQTomography"],
) -> list:
    data_dict = {}
    num_data = simulation_settings[0].num_data

    Category = namedtuple("Category", ["estimator", "loss", "algo"])
    for i, s in enumerate(simulation_settings):
        estimator_name = s.estimator.__class__.__name__
        loss_name = s.loss.__class__.__name__ if s.loss else None
        algo_name = s.algo.__class__.__name__ if s.algo else None
        results = estimation_results_list[i]
        qtomography = qtomography_list[i]
        case_name = s.name
        category = Category(estimator_name, loss_name, algo_name)

        if category in data_dict:
            data_dict[category]["estimation_results"].append(results)
            data_dict[category]["case_names"].append(case_name)
            data_dict[category]["estimators"].append(estimator_name)
            data_dict[category]["losses"].append(loss_name)
            data_dict[category]["algos"].append(algo_name)
            data_dict[category]["qtomography_list"].append(qtomography)
        else:
            data_dict[category] = dict(
                estimation_results=[results],
                case_names=[s.name],  # TODO: case_name?
                estimators=[s.estimator],  # TODO: estimator_name?
                losses=[s.loss],  # TODO: loss_name?
                algos=[s.algo],  # TODO: algo_name?
                qtomography_list=[qtomography],
            )
    figs = []

    for key, target_dict in data_dict.items():
        style = "font-size: 14px;"
        additional_title_text = (
            f'<span style="{style}">Estimator={key.estimator.replace("Estimator", "")}'
        )
        if key.loss is not None:
            additional_title_text += f"<br>Loss={key.loss}"
        if key.algo is not None:
            additional_title_text += f"<br>Algo={key.algo}"
        additional_title_text += "</span>"

        fig = make_mses_graph_estimation_results(
            target_dict["estimation_results"],
            target_dict["case_names"],
            true_object,
            additional_title_text=additional_title_text,
            show_analytical_results=True,
            estimator_list=target_dict["estimators"],
            num_data=num_data,
            qtomography_list=target_dict["qtomography_list"],
        )
        fig.update_layout(title=dict(yanchor="bottom", y=0.96))

        figs.append(fig)
    return figs


def make_mses_graphs_para(
    estimation_results_list: List["EstimationResult"],
    case_names: List[str],
    true_object: "QOperation",
    num_data: List[int],
    parameter_list: List[bool],
    qtomography_list: List["StandardQTomography"],
) -> list:
    # Split data (True/False)
    true_dict = dict(
        title="True", estimation_results=[], case_names=[], qtomography_list=[]
    )
    false_dict = dict(
        title="False", estimation_results=[], case_names=[], qtomography_list=[]
    )

    for i, results in enumerate(estimation_results_list):
        if parameter_list[i]:
            # if _get_parameter(results):
            # on_para_eq_constraint = True
            true_dict["estimation_results"].append(results)
            true_dict["case_names"].append(case_names[i])
            true_dict["qtomography_list"].append(qtomography_list[i])
        else:
            # on_para_eq_constraint = False
            false_dict["estimation_results"].append(results)
            false_dict["case_names"].append(case_names[i])
            false_dict["qtomography_list"].append(qtomography_list[i])

    # Make figure
    figs = []
    for target_dict in [true_dict, false_dict]:
        if not target_dict["case_names"]:
            continue
        fig = make_mses_graph_estimation_results(
            target_dict["estimation_results"],
            target_dict["case_names"],
            true_object,
            num_data=num_data,
            additional_title_text=f"parametrization={target_dict['title']}",
            qtomography_list=target_dict["qtomography_list"],
        )
        figs.append(fig)
    return figs


def show_mses(
    num_data: List[int],
    mses: List[List[float]],
    title: str = "Mean squared error",
    names: Optional[List[str]] = None,
):
    fig = make_mses_graph(num_data=num_data, mses=mses, title=title, names=names)
    fig.show()


# common(depend on "num_data")
def show_computation_times(
    num_data: List[int],
    computation_times_sequence: List[List[float]],
    title: str = "Computation times for each estimate",
    histnorm: str = "count",
):
    if not histnorm in ["count", "percent", "frequency"]:
        raise ValueError(
            f"histnorm is in ['count', 'percent', 'frequency']. histnorm of HS is {histnorm}"
        )

    subplot_titles = [
        f"Number of data = {num}<br>Total count of number = {len(computation_times)}"
        for num, computation_times in zip(num_data, computation_times_sequence)
    ]
    fig = make_subplots(rows=1, cols=len(num_data), subplot_titles=subplot_titles)

    # "count", "percent", "frequency"
    histnorm_quara_to_plotly = {
        "count": "",
        "percent": "percent",
        "frequency": "probability",
    }
    histnorm_param = histnorm_quara_to_plotly[histnorm]
    for index, computation_times in enumerate(computation_times_sequence):
        trace = go.Histogram(
            x=computation_times,
            xbins=dict(start=0),
            histnorm=histnorm_param,
            marker=dict(color="Blue"),
        )
        fig.append_trace(trace, 1, index + 1)

    fig.update_layout(
        title_text=title,
        xaxis_title_text="Computation time(sec)",
        yaxis_title_text=histnorm,
        showlegend=False,
        width=1200,
    )
    fig.show()


# common(show scatter)
def show_average_computation_times(
    num_data: List[int],
    computation_times_sequence: List[float],
    num_of_runs,
    title: str = None,
):
    trace = go.Scatter(x=num_data, y=computation_times_sequence, mode="lines+markers")
    data = [trace]

    if title is None:
        title = f"Computation times for estimates<br> number of runs for averaging = {num_of_runs}"
    max_value = np.max(computation_times_sequence)
    layout = go.Layout(
        title=title,
        xaxis_title_text="Number of data",
        yaxis_title_text="Average of computation times(sec)",
        # yaxis_min=0,
        xaxis_type="log",
        yaxis_range=[0, max_value * 1.1],
    )
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def extract_empi_dists_sequences_old(
    results: List["EstimationResult"],
) -> List[List[List[np.ndarray]]]:
    converted = []
    num_data_len = len(results[0].data)
    n_rep = len(results)
    for num_data_index in range(num_data_len):
        converted_dists_seq = []
        for rep_index in tqdm(range(n_rep)):
            result = results[rep_index]
            empi_dists = result.data[num_data_index]
            # list of tuple -> list of np.ndarray
            converted_dists = [data[1] for data in empi_dists]
            converted_dists_seq.append(converted_dists)
        converted.append(converted_dists_seq)
    return converted


def extract_empi_dists_sequences(
    source_empi_dists_sequences,
) -> List[List[List[np.ndarray]]]:
    converted = []
    num_data_len = len(source_empi_dists_sequences[0])
    n_rep = len(source_empi_dists_sequences)
    for num_data_index in range(num_data_len):
        converted_dists_seq = []
        for rep_index in tqdm(range(n_rep)):
            empi_dists_seq = source_empi_dists_sequences[rep_index]
            empi_dists = empi_dists_seq[num_data_index]
            # list of tuple -> list of np.ndarray
            converted_dists = [data[1] for data in empi_dists]
            converted_dists_seq.append(converted_dists)
        converted.append(converted_dists_seq)
    return converted


def make_empi_dists_mse_graph(
    simulation_result: SimulationResult, true_object: "QOperation"
):
    num_data = simulation_result.simulation_setting.num_data
    n_rep = simulation_result.simulation_setting.n_rep
    qtomography = simulation_result.qtomography
    num_schedules = qtomography.num_schedules

    # Data
    display_names = ["Empirical distributions"]
    para = qtomography.on_para_eq_constraint
    true_object_copied = _recreate_qoperation(true_object, para)
    empi_dists = extract_empi_dists_sequences(simulation_result.empi_dists_sequences)

    xs_list_list = empi_dists
    ys_list_list = [[qtomography.calc_prob_dists(true_object_copied)] * n_rep] * len(
        num_data
    )

    mses = []
    error_bar_values = []
    for i in range(len(num_data)):
        mse, std = matrix_util.calc_mse_prob_dists(xs_list_list[i], ys_list_list[i])
        mses.append(mse)
        sigma = std
        error_bar_value = sigma / np.sqrt(n_rep)
        error_bar_values.append(error_bar_value)
    mses_list = [mses]
    error_bar_values_list = [error_bar_values]

    # Analytical
    true_mses = []
    for num in num_data:
        true_mse = qtomography.calc_mse_empi_dists_analytical(
            true_object_copied, [num] * num_schedules
        )
        true_mses.append(true_mse)

    mses_list.append(true_mses)
    display_names.append(f"Analytical result")

    fig = make_mses_graph(
        mses=mses_list,
        num_data=num_data,
        names=display_names,
        error_bar_values_list=error_bar_values_list,
        yaxis_title_text="Mean squared error",
    )
    return fig
