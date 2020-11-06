from quara.protocol.qtomography.estimator import EstimationResult
import time
from typing import Callable, List, Optional, Union

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from tqdm import tqdm

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.povm import (
    Povm,
    get_x_measurement,
    get_y_measurement,
    get_z_measurement,
)
from quara.objects.qoperation import QOperation
from quara.objects.state import State
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.standard_qtomography_estimator import (
    StandardQTomographyEstimator,
    StandardQTomographyEstimationResult,
)
from quara.utils import matrix_util


def calc_mse_general_norm(
    xs: List[np.array],
    y: np.array,
    norm_function: Callable[[np.array, np.array], np.float64],
) -> np.float64:
    """calculates mse(mean squared error) of ``xs`` and ``y`` according to ``norm_function``.

    Parameters
    ----------
    xs : np.array
        sample values.
    y : np.array
        true value.
    norm_function : Callable[[np.array, np.array], np.float64]
        norm function.

    Returns
    -------
    np.float64
        mse = 1/len(x) \sum_i norm_function(x_i, y)^2
    """
    norms = []
    for x in xs:
        norm = norm_function(x, y) ** 2
        norms.append(norm)

    mse = np.mean(norms, dtype=np.float64)
    return mse


def calc_covariance_matrix_of_prob_dist(prob_dist: np.array, data_num: int) -> np.array:
    """calculates covariance matrix of probability distribution.

    Parameters
    ----------
    prob_dist : np.array
        probability distribution.
    data_num : int
        number of data.

    Returns
    -------
    np.array
        covariance matrix = 1/N (diag(p) - p \cdot p^T), where N is ``data_num`` and p is ``prob_dist``.
    """
    matrix = np.diag(prob_dist) - np.array([prob_dist]).T @ np.array([prob_dist])
    return matrix / data_num


def calc_covariance_matrix_of_prob_dists(
    prob_dists: List[np.array], data_num: int
) -> np.array:
    """calculates covariance matrix of probability distributions(= direct product of each covariance matrix of probability distribution).

    Parameters
    ----------
    prob_dists : List[np.array]
        probability distributions.
    data_num : int
        number of data.

    Returns
    -------
    np.array
        direct product of each covariance matrix = \oplus_j V(p^j), where V(p) is covariance matrix of p.
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


# common
# statistical quantity
def calc_mse_qoperations(xs: List[QOperation], ys: List[QOperation]) -> np.float64:
    points = []
    for x, y in zip(xs, ys):
        x_vec = x.to_stacked_vector()
        y_vec = y.to_stacked_vector()
        point = np.vdot(x_vec - y_vec, x_vec - y_vec)
        points.append(point)

    mse = np.mean(points, dtype=np.float64)
    std = np.std(points, dtype=np.float64, ddof=1)
    return mse, std


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
def calc_estimate(
    tester_povms: List[Povm],
    true_object: State,
    num_data: List[int],
    iteration: int,
    estimator=StandardQTomographyEstimator,
    on_para_eq_constraint: bool = True,
) -> List[StandardQTomographyEstimationResult]:
    qst = StandardQst(tester_povms, on_para_eq_constraint=on_para_eq_constraint)

    # generate empi dists and calc estimate
    results = []
    for ite in range(iteration):
        empi_dists_seq = qst.generate_empi_dists_sequence(true_object, num_data)
        result = estimator.calc_estimate_sequence(
            qst, empi_dists_seq, is_computation_time_required=True
        )

        info = {
            "iteration": ite + 1,
            "data": empi_dists_seq,
            "estimated_var_sequence": result.estimated_var_sequence,
            "computation_times": result.computation_times,
        }
        print(info)
        results.append(result)

    return results


# common
def _estimate(
    qtomography: "StandardQTomography",
    true_object: QOperation,
    num_data: List[int],
    estimator=StandardQTomographyEstimator,
) -> StandardQTomographyEstimationResult:
    empi_dists_seq = qtomography.generate_empi_dists_sequence(true_object, num_data)
    result = estimator.calc_estimate_sequence(
        qtomography, empi_dists_seq, is_computation_time_required=True
    )
    return result


# common
def estimate(
    qtomography: "StandardQTomography",
    true_object: QOperation,
    num_data: List[int],
    estimator: StandardQTomographyEstimator,
    iteration: Optional[int] = None,
) -> Union[
    StandardQTomographyEstimationResult, List[StandardQTomographyEstimationResult],
]:

    if iteration is None:
        result = _estimate(qtomography, true_object, num_data, estimator)
        return result
    else:
        results = []
        for _ in tqdm(range(iteration)):
            result = _estimate(qtomography, true_object, num_data, estimator)
            results.append(result)
        return results


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
    title: str = "Mean squared error",
    names: Optional[List[str]] = None,
    yaxis_title_text: str = "Mean squared error of estimates and true",
) -> List["Figure"]:
    if not names:
        names = [f"data_{i}" for i in range(len(mses))]
    data = []
    for i, mse in enumerate(mses):
        trace = go.Scatter(x=num_data, y=mse, mode="lines+markers", name=names[i])
        data.append(trace)

    layout = go.Layout(
        title=title,
        xaxis_title_text="Number of data",
        # yaxis_title_text="Mean squared error of estimates and true",
        yaxis_title_text=yaxis_title_text,
        xaxis_type="log",
        yaxis_type="log",
    )
    fig = go.Figure(data=data, layout=layout)
    return fig


def make_mses_graph_estimation_results(
    estimation_results_list: List["LinearEstimationResult"],
    case_names: List[str],
    num_data,
    true_object,
    title: str = None,
    show_analytical_results: bool = True,
    tester_objects: List[QOperation] = None,
) -> "Figure":
    mses_list = []
    display_case_names = case_names[:]
    for estimation_results in estimation_results_list:
        mses, *_ = convert_to_series(estimation_results, true_object)
        mses_list.append(mses)

    # calc analytical result
    if show_analytical_results:
        if not (tester_objects):
            error_message = "Specify 'tester_objects' if 'show_analytical_results' is True to show the analutical result."
            raise ValueError(error_message)

        qtomography_classes = set(
            [results[0].qtomography.__class__ for results in estimation_results_list]
        )

        for qtomography_class in qtomography_classes:
            for parameter in [True, False]:
                # Make QOperation
                true_object_copied = true_object.copy()
                true_object_copied._on_para_eq_constraint = parameter

                # Make QTomography
                args = dict(on_para_eq_constraint=parameter,)
                if type(true_object) == Povm:
                    args["measurement_n"] = len(true_object.vecs)
                tmp_tomography = qtomography_class(tester_objects, **args)

                true_mses = []
                for num in num_data:
                    true_mse = tmp_tomography.calc_mse_linear_analytical(
                        true_object_copied, [num] * len(tester_objects)
                    )
                    true_mses.append(true_mse)
                mses_list.append(true_mses)
                display_case_names.append(f"Analytical result (Linear, {parameter})")
    fig = make_mses_graph(num_data, mses_list, names=display_case_names, title=title)
    return fig


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


def extract_empi_dists(results: List["EstimationResult"]) -> List[List[List[np.array]]]:
    converted = []
    num_data_len = len(results[0].data)  # num_dataの要素数
    n_rep = len(results)
    for num_data_index in range(num_data_len):  # num_dataの要素数だけ回る
        converted_dists_seq = []
        for rep_index in tqdm(range(n_rep)):  # Nrepの数だけ回る
            result = results[rep_index]
            empi_dists = result.data[num_data_index]
            # list of tuple -> list of np.array
            converted_dists = [data[1] for data in empi_dists]
            converted_dists_seq.append(converted_dists)
        converted.append(converted_dists_seq)
    return converted


def make_empi_dists_mse_graph(
    estimation_results: List["LinearEstimationResult"], true_object: "QOperation"
):
    qtomography = estimation_results[0]._qtomography
    num_data = estimation_results[0].num_data
    n_rep = len(estimation_results)
    mses_list = []

    # Data
    display_names = ["Empirical distributions"]

    empi_dists = extract_empi_dists(estimation_results)
    xs_list_list = empi_dists
    ys_list_list = [[qtomography.calc_prob_dists(true_object)] * n_rep] * len(num_data)

    mses = []
    for i in range(len(num_data)):
        mses.append(matrix_util.calc_mse_prob_dists(xs_list_list[i], ys_list_list[i]))
    mses_list.append(mses)

    # Analytical
    true_mses = []
    for num in num_data:
        true_mse = qtomography.calc_mse_empi_dists_analytical(true_object, [num] * 3)
        true_mses.append(true_mse)

    mses_list.append(true_mses)
    display_names.append(f"Analytical result")

    fig = make_mses_graph(
        mses=mses_list,
        num_data=num_data,
        names=display_names,
        yaxis_title_text="Mean squared error",
    )
    return fig


# def make_empi_dists_mse_graph_for_debug(
#     estimation_results_list: List[List[EstimationResult]],
#     qtomographies: List["StandardQTomography"],
#     true_object: "QOperation",
#     num_data: List[int],
#     n_rep: int,
#     tester_objects: List["QOperation"],
#     case_names: List["str"],
# ):
#     mses_list = []

#     # Data
#     display_names = [f"Empirical distributions ({name})" for name in case_names]

#     for j, results in enumerate(estimation_results_list):
#         empi_dists = extract_empi_dists(results)
#         xs_list_list = empi_dists
#         ys_list_list = [[qtomographies[j].calc_prob_dists(true_object)] * n_rep] * len(
#             num_data
#         )

#         mses = []
#         for i in range(len(num_data)):
#             mses.append(
#                 matrix_util.calc_mse_prob_dists(xs_list_list[i], ys_list_list[i])
#             )
#         mses_list.append(mses)

#     qtomography = qtomographies[0]
#     estimation_results = estimation_results_list[0]
#     # Analytical
#     true_mses = []
#     for num in num_data:
#         true_mse = qtomography.calc_mse_empi_dists_analytical(
#                 true_object, [num] * 3
#             )

#         true_mses.append(true_mse)
#     mses_list.append(true_mses)
#     display_names.append(f"Analytical result")

#     fig = make_mses_graph(
#         mses=mses_list,
#         num_data=num_data,
#         names=display_names,
#         yaxis_title_text="Mean squared error",
#     )
#     return fig
