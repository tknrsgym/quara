import warnings
from typing import List, Tuple, Dict, Any, Union, Optional
from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from tqdm import tqdm

from quara.protocol.qtomography.estimator import EstimationResult
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate
from quara.settings import Settings


def get_sorted_eigenvalues_list(
    estimated_qobject_list: List["State"],
) -> List[List[float]]:
    sorted_eigenvalues_list = []

    for estimated_qobject in tqdm(estimated_qobject_list):
        sorted_eigenvalues = sorted(
            [x.real for x in estimated_qobject.calc_eigenvalues()], reverse=True
        )
        sorted_eigenvalues_list.append(sorted_eigenvalues)
    return sorted_eigenvalues_list


def get_sum_of_eigenvalues_violation(
    eigenvalues_list: List[List[float]],
) -> Tuple[List[float], List[float]]:
    sum_eig_less_than_zero_list = []
    sum_eig_greater_than_one_list = []
    # sorted_eigenvalues_list = sorted(eigenvalues_list, reverse=True)
    sorted_eigenvalues_list = eigenvalues_list
    eps = 10 ** (-13)
    for i, values in enumerate(sorted_eigenvalues_list):
        eig_less_than_zero_list = [v for v in values if v < 0 - eps]
        if eig_less_than_zero_list:
            sum_eig_less_than_zero_list.append(np.sum(eig_less_than_zero_list))

        eig_greater_than_one_list = [v for v in values if v > 1 + eps]
        if eig_greater_than_one_list:
            sum_eig_greater_than_one_list.append(np.sum(eig_greater_than_one_list))

    return sum_eig_less_than_zero_list, sum_eig_greater_than_one_list


def get_sum_of_eigenvalues_violation_povm(
    estimated_povms: List["Povm"],
) -> Dict[int, List[float]]:

    minus_eigenvalues_dict = defaultdict(lambda: [])

    for est in tqdm(estimated_povms):
        eigenvalues = est.calc_eigenvalues()
        sorted_eigenvalues = []
        for eigs in eigenvalues:
            eigs = sorted(eigs, reverse=True)
            sorted_eigenvalues.append(eigs)

        eps = Settings.get_atol()
        for x_i, values in enumerate(sorted_eigenvalues):
            for e in values:
                if e.imag >= Settings.get_atol():
                    message = f"​The imaginary part of the eigenvalue is larger than {Settings.get_atol()}"
                    warnings.warn(message)

            minus_values = [e.real for e in values if e.real < 0 - eps]
            if minus_values:
                sum_values = sum(minus_values)
                if minus_eigenvalues_dict[x_i]:
                    minus_eigenvalues_dict[x_i].append(sum_values)
                else:
                    minus_eigenvalues_dict[x_i] = [sum_values]
    return minus_eigenvalues_dict


def get_trace_list(estimated_state_list: List["State"],) -> List[float]:
    trace_list = []

    for estimated_state in estimated_state_list:
        tr = np.trace(estimated_state.to_density_matrix())
        value = tr.real
        if tr.imag >= 10 ** -14:
            message = "Imaginary number of trace >= 10 ** -14"
            warnings.warn(message)
        trace_list.append(value)
    return trace_list


def get_sum_vecs(estimated_povms: List["Povm"]) -> np.array:
    sum_vecs = None

    for est in tqdm(estimated_povms):
        if sum_vecs is not None:
            sum_vecs = np.vstack([sum_vecs, sum(est.vecs)])
        else:
            sum_vecs = sum(est.vecs)

    sum_vecs = sum_vecs.T
    return sum_vecs


# Plot
# Common
def make_prob_dist_histogram(
    values: List[float],
    bin_size: Union[int, float, List[float]],
    num_data: int,
    annotation_vlines: List[Union[float, int]] = None,
    xaxis_title_text: str = "Value",
    x_abs_min: float = 10 ** (-3),
    title: str = None,
    additional_title_text: str = None,
):
    if list(values):
        x_min, x_max = min(values), max(values)
    else:
        x_min, x_max = None, None

    # Adjust size of bin
    if type(bin_size) in [int, float]:
        xbins = dict(size=bin_size)
    elif type(bin_size) == go.histogram.XBins:
        xbins = bin_size
    else:
        error_message = (
            f"bin_size must be int, float, or go.histogram.Xbins, not {type(bin_size)}"
        )
        raise TypeError(error_message)

    hist = go.Histogram(x=values, xbins=xbins, histnorm="probability",)
    layout = go.Layout()
    fig = go.Figure(hist, layout=layout)

    # Annotation
    if annotation_vlines:
        for x_value in annotation_vlines:
            fig.add_shape(
                type="line",
                line_color="black",
                line_width=1,
                opacity=0.5,
                x0=x_value,
                x1=x_value,
                xref="x",
                y0=0,
                y1=1,
                yref="paper",
            )
    # Y axis
    ytickvals = [y * 0.1 for y in range(0, 10)]
    fig.update_yaxes(range=[0, 1], tickvals=ytickvals, title="Frequency")
    # X axis
    if x_min is not None and x_max is not None:
        additional_text = (
            f"<br><br>Min: {x_min}<br>Max: {x_max}<br>|Max-Min|: {abs(x_max-x_min)}"
        )
        xaxis_title_text += additional_text
    fig.update_xaxes(title=xaxis_title_text)

    # Adjust range of xaxis
    if annotation_vlines:
        ref_x = annotation_vlines[0]
        if (x_min is None and x_max is None) or (
            (ref_x - x_abs_min) < x_min and x_max < (ref_x + x_abs_min)
        ):
            x_range = ((ref_x - x_abs_min), (ref_x + x_abs_min))
            fig.update_xaxes(range=[x_range[0], x_range[1]], autorange=False)

    # Title
    n_rep = len(values)
    if not title:
        title = f"N={num_data}, Nrep={n_rep}"
    if additional_title_text:
        title += additional_title_text

    n_lines = title.count("<br>") + 1
    fig.update_layout(
        title=title, margin={"t": 40 * n_lines, "b": 0},
    )
    return fig


# Common
def make_prob_dist_histograms(
    values_set: np.array, bin_size: int, x_range: Optional[tuple] = None
) -> "Figure":
    if x_range:
        x_start, x_end = x_range

    fig = make_subplots(rows=len(values_set), cols=1)

    for i, values in enumerate(values_set):
        trace = go.Histogram(
            x=values, xbins=dict(size=bin_size), histnorm="probability",
        )
        fig.append_trace(trace, i + 1, 1)

    layout = go.Layout(xaxis=dict(title="Value", dtick=0), yaxis=dict(title="Prob"))
    fig["layout"].update(layout)
    ytickvals = [y * 0.1 for y in range(0, 10)]
    fig.update_yaxes(range=[0, 1], tickvals=ytickvals, title="Frequency")

    return fig


# common
def _convert_result_to_qoperation(
    estimation_results: List[EstimationResult], num_data_index: int = 0
) -> List["QOperation"]:
    if num_data_index == 0:
        estimated_qoperations = [
            result.estimated_qoperation for result in estimation_results
        ]
    else:
        estimated_qoperations = [
            result.estimated_qoperation_sequence[num_data_index]
            for result in estimation_results
        ]
    return estimated_qoperations


# Qst, on_para_eq_constraint=True
def make_graph_trace(
    estimation_results: List[EstimationResult],
    num_data: List[int],
    num_data_index: int = 0,
    bin_size: float = 0.0001,
) -> "Figure":
    estimated_states = [
        result.estimated_qoperation_sequence[num_data_index]
        for result in estimation_results
    ]

    trace_list = get_trace_list(estimated_states)

    num_data = num_data[num_data_index]
    fig = make_prob_dist_histogram(
        trace_list, num_data=num_data, bin_size=bin_size, annotation_vlines=[1]
    )
    return fig


# common, on_para_eq_constraint=False
def make_graphs_eigenvalues(
    estimation_results: List[EstimationResult],
    true_object: "QOperation",
    num_data: List[int],
    num_data_index: int = 0,
    bin_size: float = 0.0001,
) -> Union[List["Figure"], List[List["Figure"]]]:
    estimated_qoperations = _convert_result_to_qoperation(
        estimation_results, num_data_index=num_data_index
    )

    n_data = num_data[num_data_index]
    if type(true_object) == State:
        figs = _make_graphs_eigenvalues_state(
            estimated_qoperations, true_object, n_data, bin_size
        )
    elif type(true_object) == Povm:
        figs = _make_graphs_eigenvalues_povm(
            estimated_qoperations, true_object, n_data, bin_size
        )
    elif type(true_object) == Gate:
        raise NotImplementedError()
    else:
        message = f"true_object must be State, Povm, or Gate, not {type(true_object)}"
        raise TypeError(message)
    return figs


def make_graphs_sum_unphysical_eigenvalues(
    estimation_results: List[EstimationResult],
    num_data_index: int = 0,
    bin_size: float = 0.0001,
):
    num_data = estimation_results[0].num_data
    estimated_qoperations = _convert_result_to_qoperation(
        estimation_results, num_data_index=num_data_index
    )
    n_data = num_data[num_data_index]
    sample_object = estimated_qoperations[0]
    if type(sample_object) == State:
        figs = _make_graphs_sum_unphysical_eigenvalues_state(
            estimated_qoperations, n_data, bin_size
        )
    elif type(sample_object) == Povm:
        figs = _make_graphs_sum_unphysical_eigenvalues_povm(
            estimated_qoperations, n_data, bin_size
        )
    elif type(sample_object) == Gate:
        raise NotImplementedError()
    else:
        # TODO: message
        raise TypeError()
    return figs


def make_xbins(
    ref_x: float, min_x: float, max_x: float, bin_size: float
) -> List[float]:
    bin_list = []
    bin_list.append(ref_x - bin_size / 2)
    bin_list.append(ref_x + bin_size / 2)

    x = min(bin_list)

    while x > min_x:
        x -= bin_size
        bin_list.append(x)

    x = max(bin_list)

    while x < max_x:
        x += bin_size
        bin_list.append(x)

    return go.histogram.XBins(start=min(bin_list), end=max(bin_list), size=bin_size)


def make_graphs_sum_vecs(
    estimation_results: List["EstimatedResult"],
    true_object: "Povm",
    num_data_index: int,
    bin_size: float = 0.0001,
) -> List["Figure"]:
    num_data = estimation_results[0].num_data

    estimated_povms = _convert_result_to_qoperation(
        estimation_results, num_data_index=num_data_index
    )

    vlines_list = [np.sqrt(true_object.dim), 0, 0, 0]
    sum_vecs = get_sum_vecs(estimated_povms)
    fig_list = []

    for i, value_list in enumerate(sum_vecs):
        min_x, max_x = min(value_list), min(value_list)
        xbins = make_xbins(
            ref_x=vlines_list[i], min_x=min_x, max_x=max_x, bin_size=bin_size
        )

        fig = make_prob_dist_histogram(
            value_list,
            bin_size=xbins,
            num_data=num_data,
            annotation_vlines=[vlines_list[i]],
        )
        title = f"N={num_data[num_data_index]}, α={i}"
        fig.update_layout(title=title)
        fig_list.append(fig)
    return fig_list


def _make_graphs_eigenvalues_state(
    estimated_states: List[State],
    true_object: State,
    num_data: int,
    bin_size: float = 0.0001,
) -> List["Figure"]:
    # Eigenvalues of True State
    true_eigs = sorted(
        [eig.real for eig in true_object.calc_eigenvalues()], reverse=True
    )

    # Eigenvalues of Estimated States
    sorted_eigenvalues_list = get_sorted_eigenvalues_list(estimated_states)
    sorted_eigenvalues_list = np.array(sorted_eigenvalues_list).T.tolist()

    figs = []
    for i, values in enumerate(sorted_eigenvalues_list):
        # plot eigenvalues of estimated states
        fig = make_prob_dist_histogram(
            values,
            num_data=num_data,
            bin_size=bin_size,
            xaxis_title_text=f"Eigenvalue (i={i})",
            annotation_vlines=[true_eigs[i]],
        )
        fig.update_layout(title=f"N={num_data}, Nrep={len(values)}")
        figs.append(fig)

    return figs


def get_sorted_eigenvalues_list_povm(
    estimated_povms: List[Povm],
) -> Dict[int, np.array]:
    eigenvalues_dict = defaultdict(lambda: [])

    for est in tqdm(estimated_povms):
        eigenvalues = est.calc_eigenvalues()

        # TODO: 虚部が10**(-13)より大きい場合はwarningを出す
        sorted_eigenvalues = []
        for eigs in eigenvalues:  # 各測定値の固有値
            eigs = sorted(eigs, reverse=True)
            eigs_real = [eig.real for eig in eigs]
            sorted_eigenvalues.append(eigs_real)

        for x_i, values in enumerate(sorted_eigenvalues):
            if eigenvalues_dict[x_i]:
                eigenvalues_dict[x_i].append(values)
            else:
                eigenvalues_dict[x_i] = [values]

    for x_i, values_list in eigenvalues_dict.items():
        eigenvalues_dict[x_i] = np.array(values_list).T
    return eigenvalues_dict


def _make_graphs_eigenvalues_povm(
    estimated_povms: List[Povm],
    true_object: Povm,
    num_data: int,
    bin_size: float = 0.0001,
) -> List[List["Figure"]]:
    n_rep = len(estimated_povms)
    sorted_true_eigs_list = []
    for eigenvalues in true_object.calc_eigenvalues():
        sorted_true_eigs = sorted([eig.real for eig in eigenvalues], reverse=True)
        sorted_true_eigs_list.append(sorted_true_eigs)

    # Eigenvalues of Estimated Povms
    eigenvalues_dict = get_sorted_eigenvalues_list_povm(estimated_povms)
    fig_list_list = []
    for x_i, eigs_list in tqdm(eigenvalues_dict.items()):  # each measurement
        fig_list = []
        for i, value_list in tqdm(enumerate(eigs_list)):  # each eigenvalue
            title = f"N={num_data}, Nrep={n_rep}, x={x_i}"

            fig = make_prob_dist_histogram(
                value_list,
                bin_size=bin_size,
                num_data=num_data,
                xaxis_title_text=f"Eigenvalues (i={i})",
                annotation_vlines=[sorted_true_eigs_list[x_i][i]],
            )
            fig.update_layout(title=title)
            fig_list.append(fig)
        fig_list_list.append(fig_list)

    return fig_list_list


# only State
def _make_graphs_sum_unphysical_eigenvalues_state(
    estimated_states: List[State], num_data: int, bin_size: float = 0.0001
) -> List["Figure"]:

    sorted_eigenvalues_list = get_sorted_eigenvalues_list(estimated_states)
    less_than_zero_list, greater_than_one_list = get_sum_of_eigenvalues_violation(
        sorted_eigenvalues_list
    )

    n_rep = len(sorted_eigenvalues_list)
    figs = []
    # Figure 1
    xaxis_title_text = f"Sum of unphysical eigenvalues (<0)"
    n_unphysical = len(less_than_zero_list)

    fig = make_prob_dist_histogram(
        less_than_zero_list,
        bin_size=bin_size,
        num_data=num_data,
        annotation_vlines=[0],
        xaxis_title_text=xaxis_title_text,
        additional_title_text=f"<br>Number of unphysical estimates={n_unphysical}",
    )
    # title = (
    #     f"N={num_data}, Nrep={n_rep}"
    # )
    # fig.update_layout(title=title)
    figs.append(fig)

    # Figure 2
    xaxis_title_text = f"Sum of unphysical eigenvalues (>1)"
    n_unphysical = len(less_than_zero_list)
    fig = make_prob_dist_histogram(
        greater_than_one_list,
        bin_size=bin_size,
        num_data=num_data,
        annotation_vlines=[1],
        xaxis_title_text=xaxis_title_text,
        additional_title_text=f"<br>Number of unphysical estimates={n_unphysical}",
    )

    figs.append(fig)

    return figs


def _make_graphs_sum_unphysical_eigenvalues_povm(
    estimated_povms: List["Povm"], num_data: int, bin_size: float = 0.0001
) -> List["Figure"]:
    figs = []
    n_rep = len(estimated_povms)
    minus_eigenvalues_dict = get_sum_of_eigenvalues_violation_povm(estimated_povms)
    measurement_n = len(estimated_povms[0].vecs)

    xaxis_title_text = f"Sum of unphysical eigenvalues (<0)"
    for x_i in range(measurement_n):
        value_list = []
        if x_i in minus_eigenvalues_dict:
            value_list = minus_eigenvalues_dict[x_i]

        title = f"N={num_data}, Nrep={n_rep}, x={x_i}"
        title += f"<br>Number of unphysical estimates={len(value_list)}"
        fig = make_prob_dist_histogram(
            value_list,
            bin_size=bin_size,
            annotation_vlines=[0],
            num_data=num_data,
            xaxis_title_text=xaxis_title_text,
            title=title,
        )
        figs.append(fig)

    return figs
