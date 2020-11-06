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

        # TODO: 虚部が10**(-13)より大きい場合はwarningを出す
        for x_i, values in enumerate(sorted_eigenvalues):
            sum_values = sum([e.real for e in sorted_eigenvalues[x_i] if e.real < 0])
            if minus_eigenvalues_dict[x_i]:
                minus_eigenvalues_dict[x_i].append(sum_values)
            else:
                minus_eigenvalues_dict[x_i] = [sum_values]
    return minus_eigenvalues_dict


# TODO: rename
def get_physicality_violation_result_for_state_affine(
    estimated_state_list: List["State"],
) -> List[float]:
    value_list = []

    for estimated_state in estimated_state_list:
        tr = np.trace(estimated_state.to_density_matrix())
        value = tr.real
        if tr.imag >= 10 ** -14:
            message = "Imaginary number of trace >= 10 ** -14"
            warnings.warn(message)
        value_list.append(value)
    return value_list


def get_sum_vecs(estimated_povms: List["Povm"]) -> np.array:
    sum_vecs = None

    for est in tqdm(estimated_povms):
        if sum_vecs is not None:
            sum_vecs = np.vstack([sum_vecs, sum(est.vecs)])
        else:
            sum_vecs = sum(est.vecs)

    sum_vecs = sum_vecs.T
    return sum_vecs


# TODO: rename
def check_physicality_violation(
    estimation_results: List[EstimationResult], num_data_index: int = 0
) -> Dict[str, Any]:
    qoperation = estimation_results[0].estimated_qoperation
    estimated_qoperations = [
        result.estimated_qoperation_sequence[num_data_index]
        for result in estimation_results
    ]
    if type(qoperation) == State:
        result = _check_physicality_violation_for_state(estimated_qoperations)
    elif type(qoperation) == Povm:
        result = _check_physicality_violation_for_povm(estimated_qoperations)
    else:
        raise NotImplementedError()
    return result


def _check_physicality_violation_for_state(
    estimated_qobjects: List["State"],
) -> Dict[str, Any]:
    on_para_eq_constraint = estimated_qobjects[0].on_para_eq_constraint

    if on_para_eq_constraint:
        trace_list = get_physicality_violation_result_for_state_affine(
            estimated_qobjects
        )
        return dict(trace_list=trace_list)
    else:
        sorted_eigenvalues_list = get_sorted_eigenvalues_list(estimated_qobjects)
        sorted_eigenvalues_list_T = np.array(sorted_eigenvalues_list).T.tolist()
        less_than_zero_list, greater_than_one_list = get_sum_of_eigenvalues_violation(
            sorted_eigenvalues_list
        )
        return dict(
            sorted_eigenvalues_list=sorted_eigenvalues_list_T,
            sum_of_eigenvalues=dict(
                less_than_zero=less_than_zero_list,
                greater_than_one=greater_than_one_list,
            ),
        )


def _check_physicality_violation_for_povm(estimated_povms: List["Povm"]) -> dict:
    on_para_eq_constraint = estimated_qobjects[0].on_para_eq_constraint

    if on_para_eq_constraint:
        sum_vecs = get_sum_vecs(estimated_povms)
        return dict(sum_vecs=sum_vecs)
    else:
        raise NotImplementedError()


# Plot
# Common
def make_prob_dist_histogram(
    values: List[float],
    bin_size: int,
    num_data: int,
    x_range: Optional[tuple] = None,
    annotation_vlines: List[Union[float, int]] = None,
):
    if x_range:
        x_start, x_end = x_range

    hist = go.Histogram(x=values, xbins=dict(size=bin_size), histnorm="probability",)

    layout = go.Layout(xaxis=dict(title="Value", dtick=0), yaxis=dict(title="Prob"))

    fig = go.Figure(hist, layout=layout)
    ytickvals = [y * 0.1 for y in range(0, 10)]

    if annotation_vlines:
        for x_value in annotation_vlines:
            # 指定した分位点を青線で表示
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

    fig.update_yaxes(range=[0, 1], tickvals=ytickvals, title="Frequency")
    n_rep = len(values)
    title = f"N={num_data}, Nrep={n_rep}"
    fig.update_layout(title=title)

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
    violation_result = check_physicality_violation(
        estimation_results, num_data_index=num_data_index
    )
    num_data = num_data[num_data_index]
    fig = make_prob_dist_histogram(
        violation_result["trace_list"], num_data=num_data, bin_size=bin_size
    )
    return fig


# common, on_para_eq_constraint=False
def make_graphs_eigenvalues(
    estimation_results: List[EstimationResult],
    true_object: "QOperation",
    num_data: List[int],
    num_data_index: int = 0,
    bin_size: float = 0.0001,
) -> List["Figure"]:
    estimated_qoperations = _convert_result_to_qoperation(
        estimation_results, num_data_index=num_data_index
    )

    n_data = num_data[num_data_index]
    if type(true_object) == State:
        figs = _make_graphs_eigenvalues_state(
            estimated_qoperations, true_object, n_data, bin_size
        )
    else:
        # TODO: message
        raise TypeError()
    return figs


def make_graphs_sum_unphysical_eigenvalues(
    estimation_results: List[EstimationResult],
    num_data: List[int],
    num_data_index: int = 0,
    bin_size: float = 0.0001,
):
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
    else:
        # TODO: message
        raise TypeError()
    return figs


def make_graphs_sum_vecs(
    estimation_results: List["EstimatedResult"],
    true_object: "Povm",
    num_data_index: int,
) -> List["Figure"]:
    num_data = estimation_results[0].num_data

    estimated_povms = _convert_result_to_qoperation(
        estimation_results, num_data_index=num_data_index
    )

    vlines_list = [np.sqrt(true_object.dim), 0, 0, 0]
    sum_vecs = get_sum_vecs(estimated_povms)
    fig_list = []
    for i, value_list in enumerate(sum_vecs):
        fig = make_prob_dist_histogram(
            value_list,
            bin_size=0.001,
            num_data=num_data,
            annotation_vlines=[vlines_list[i]],
        )
        title = f"N={num_data[num_data_index]}, α={i}"
        fig.update_layout(title=title)  # TODO
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
        fig = make_prob_dist_histogram(values, num_data=num_data, bin_size=bin_size)
        fig.update_layout(title=f"N={num_data}, Nrep={len(values)}")
        fig.update_xaxes(title=f"Eigenvalue (i={i})")

        # plot eigenvalues of true state
        x_value = true_eigs[i]
        fig.add_shape(
            type="line",
            line_color="red",
            line_width=2,
            opacity=0.5,
            x0=x_value,
            x1=x_value,
            xref="x",
            y0=0,
            y1=1,
            yref="paper",
        )
        figs.append(fig)

    return figs


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
    fig = make_prob_dist_histogram(
        less_than_zero_list, bin_size=bin_size, num_data=num_data, annotation_vlines=[0]
    )

    n_unphysical = len(less_than_zero_list)
    title = (
        f"N={num_data}, Nrep={n_rep}<br>Number of unphysical estimates={n_unphysical}"
    )
    fig.update_layout(title=title)
    fig.update_xaxes(title=f"Sum of unphysical eigenvalues (<0)")
    figs.append(fig)

    # Figure 2
    fig = make_prob_dist_histogram(
        greater_than_one_list,
        bin_size=bin_size,
        num_data=num_data,
        annotation_vlines=[1],
    )

    n_unphysical = len(less_than_zero_list)
    title = (
        f"N={num_data}, Nrep={n_rep}<br>Number of unphysical estimates={n_unphysical}"
    )
    fig.update_layout(title=title)  # TODO
    fig.update_xaxes(title=f"Sum of unphysical eigenvalues (>1)")
    figs.append(fig)

    return figs


def _make_graphs_sum_unphysical_eigenvalues_povm(
    estimated_povms: List["Povm"], num_data: int, bin_size: float = 0.0001
) -> List["Figure"]:
    figs = []
    minus_eigenvalues_dict = get_sum_of_eigenvalues_violation_povm(estimated_povms)
    for x_i, value_list in minus_eigenvalues_dict.items():
        fig = make_prob_dist_histogram(
            value_list, bin_size=0.001, annotation_vlines=[0], num_data=num_data
        )
        # TODO: modify
        title = f"各測定値の負の値の固有値の総和の頻度分布"
        title += f"<br>N={num_data}, x={x_i}"
        fig.update_layout(title=title)  # TODO
        figs.append(fig)

    return figs


def _generate_graph_sum_eigenvalues_seq(
    estimation_results: List["EstimationResult"],
    case_id: int,
    true_object,
    num_data: List[int],
) -> list:

    fig_info_list_list = []
    for num_data_index in range(len(num_data)):
        fig_list = physicality_violation_check.make_graphs_sum_unphysical_eigenvalues(
            estimation_results, num_data=num_data, num_data_index=num_data_index,
        )
        fig_info_list = []

        for i, fig in enumerate(fig_list):
            fig_name = f"case={case_id}_sum-unphysical-eigenvalues_num={num_data_index}_type={i}"

            # output
            # TODO
            dir_path = Path(
                "/Users/tomoko/project/rcast/workspace/quara/tutorials/images"
            )
            path = str(dir_path / f"{fig_name}.png")
            fig.update_layout(width=500, height=400)
            dir_path.mkdir(exist_ok=True)
            fig.write_image(path)

            fig_info_list.append(dict(image_path=path, fig=fig, fig_name=fig_name))

        fig_info_list_list.append(fig_info_list)
    return fig_info_list_list


def _generate_sum_eigenvalues_div(fig_info_list_list) -> str:
    graph_block_html_all = ""
    for fig_info_list in fig_info_list_list:
        graph_block_html = ""
        for fig_info in fig_info_list:
            graph_subblock = (
                f"<div class='box'><img src={fig_info['image_path']}></div>"
            )
            graph_block_html += graph_subblock

        graph_block_html_all += f"<div>{graph_block_html}</div>"
    graph_block_html_all = f"<div>{graph_block_html_all}</div>"

    return graph_block_html_all


def generate_sum_eigenvalues_div(
    estimation_results: List["EstimationResult"],
    case_id: int,
    num_data: List[int],
    true_object,
):
    fig_info_list_list = _generate_graph_sum_eigenvalues_seq(
        estimation_results, case_id=case_id, true_object=true_object, num_data=num_data
    )
    div_html = _generate_eigenvalues_div(fig_info_list_list)
    return div_html
