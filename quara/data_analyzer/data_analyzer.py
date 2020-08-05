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
def calc_mse(xs: List[np.array], ys: List[np.array]) -> np.float64:
    points = []
    for x, y in zip(xs, ys):
        point = np.dot(x - y, x - y)
        points.append(point)

    mse = np.mean(points, dtype=np.float64)
    return mse


# common(StandardQTomography)
def convert_to_series(
    results: List[StandardQTomographyEstimationResult], true_object: QOperation
):
    # calc mse
    var_tmp = [result.estimated_var_sequence for result in results]
    var_tmp = [list(var) for var in zip(*var_tmp)]
    mses = [calc_mse(var, true_object.to_stacked_vector()) for var in var_tmp]

    # convert to computation time series
    comp_time_tmp = [result.computation_times for result in results]
    comp_time = [list(comp_time) for comp_time in zip(*comp_time_tmp)]

    return mses, comp_time


# StandardQst, StandardQTomographyEstimator
def calc_estimate(
    tester_povms: List[Povm],
    true_object: State,
    num_data: List[int],
    iterations: int,
    estimator=StandardQTomographyEstimator,
    on_para_eq_constraint: bool = True,
) -> StandardQTomographyEstimationResult:
    qst = StandardQst(tester_povms, on_para_eq_constraint=on_para_eq_constraint)

    # generate empi dists and calc estimate
    results = []
    for iteration in range(iterations):
        seeds = [iteration] * len(num_data)
        empi_dists_seq = qst.generate_empi_dists_sequence(true_object, num_data, seeds)

        reult = estimator.calc_estimate_sequence(
            qst, empi_dists_seq, is_computation_time_required=True
        )

        info = {
            "iteration": iteration + 1,
            "data": empi_dists_seq,
            "estimated_var_sequence": reult.estimated_var_sequence,
            "computation_times": reult.computation_times,
        }
        print(info)
        results.append(reult)

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
    if not histnorm in ["count", "percent", "probability"]:
        raise ValueError(
            f"histnorm is in ['count', 'percent', 'probability']. histnorm of HS is {histnorm}"
        )

    subplot_titles = [
        f"number of data = {num}<br>total count of number = {len(computation_times)}"
        for num, computation_times in zip(num_data, computation_times_sequence)
    ]
    fig = make_subplots(rows=1, cols=len(num_data), subplot_titles=subplot_titles)

    # "count", "percent", "probability"
    histnorm_param = "" if histnorm == "count" else histnorm
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
