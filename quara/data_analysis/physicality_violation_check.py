import warnings
from typing import List, Tuple, Dict, Any, Union, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from tqdm import tqdm

from quara.protocol.qtomography.estimator import EstimationResult
from quara.objects.state import State


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


def check_physicality_violation(
    estimation_results: List[EstimationResult],
) -> Dict[str, Any]:
    qoperation = estimation_results[0].estimated_qoperation
    estimated_qoperations = [qo.estimated_qoperation for qo in estimation_results]
    if type(qoperation) == State:
        result = _check_physicality_violation_for_state(estimated_qoperations)
    else:
        # TODO: error message
        raise ValueError()
    return result


def _check_physicality_violation_for_state(
    estimated_qobject_list: List["State"],
) -> Dict[str, Any]:
    on_para_eq_constraint = estimated_qobject_list[0].on_para_eq_constraint

    if on_para_eq_constraint:
        violation_list = get_physicality_violation_result_for_state_affine(
            estimated_qobject_list
        )
        return dict(violation_list=violation_list)
    else:
        sorted_eigenvalues_list = get_sorted_eigenvalues_list(estimated_qobject_list)
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


def make_prob_dist_histogram(
    values: List[float],
    bin_size: int,
    x_range: Optional[tuple] = None,
    annotation_vlines: List[Union[float, int]] = None,
):
    if x_range:
        x_start, x_end = x_range

    hist = go.Histogram(x=values, xbins=dict(size=bin_size), histnorm="probability",)

    layout = go.Layout(xaxis=dict(title="value", dtick=0), yaxis=dict(title="prob"))

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

    return fig


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

    layout = go.Layout(xaxis=dict(title="value", dtick=0), yaxis=dict(title="prob"))
    fig["layout"].update(layout)
    ytickvals = [y * 0.1 for y in range(0, 10)]
    fig.update_yaxes(range=[0, 1], tickvals=ytickvals, title="Frequency")

    return fig
