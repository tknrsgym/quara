import time
from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

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


# common
# statistical quantity
def calc_statistical_quantity(xs: List[QOperation], ys: List[QOperation]) -> np.float64:
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
        calc_statistical_quantity(
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
) -> StandardQTomographyEstimationResult:
    qst = StandardQst(tester_povms, on_para_eq_constraint=on_para_eq_constraint)

    # generate empi dists and calc estimate
    results = []
    for ite in range(iteration):
        seeds = [ite] * len(num_data)
        empi_dists_seq = qst.generate_empi_dists_sequence(true_object, num_data, seeds)

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
def show_mse(num_data: List[int], mses: List[float], title: str = "Mean Square Value"):
    trace = go.Scatter(x=num_data, y=mses, mode="lines+markers")
    data = [trace]
    layout = go.Layout(
        title=title,
        xaxis_title_text="number of data",
        yaxis_title_text="Mean Square Error of estimates and true",
        xaxis_type="log",
        yaxis_type="log",
    )
    fig = go.Figure(data=data, layout=layout)
    fig.show()


# common(depend on "num_data")
def show_computation_times(
    num_data: List[int],
    computation_times_sequence: List[List[float]],
    title: str = "computation times for each estimate",
    histnorm: str = "count",
):
    if not histnorm in ["count", "percent", "frequency"]:
        raise ValueError(
            f"histnorm is in ['count', 'percent', 'frequency']. histnorm of HS is {histnorm}"
        )

    subplot_titles = [
        f"number of data = {num}<br>total count of number = {len(computation_times)}"
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
        xaxis_title_text="computation time(sec)",
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
        title = f"computation times for estimates<br> number of runs for averaging = {num_of_runs}"
    max_value = np.max(computation_times_sequence)
    layout = go.Layout(
        title=title,
        xaxis_title_text="number of data",
        yaxis_title_text="average of computation times(sec)",
        # yaxis_min=0,
        xaxis_type="log",
        yaxis_range=[0, max_value * 1.1],
    )
    fig = go.Figure(data=data, layout=layout)
    fig.show()
