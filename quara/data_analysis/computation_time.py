from typing import List, Union

import plotly.graph_objects as go
import numpy as np


def make_histogram(
    values: List[float],
    num_data: int,
    annotation_vlines: List[Union[float, int]] = None,
    xaxis_title_text: str = "Value",
    x_abs_min: float = 10 ** (-3),
    title: str = None,
    additional_title_text: str = None,
    bin_n: int = 20,
):
    if list(values):
        x_min, x_max = min(values), max(values)
    else:
        x_min, x_max = None, None

    # Adjust size of bin
    bin_size = abs(x_max - x_min) / bin_n

    if type(bin_size) in [int, float]:
        xbins = dict(size=bin_size)
    elif type(bin_size) == go.histogram.XBins:
        xbins = bin_size
    else:
        error_message = (
            f"bin_size must be int, float, or go.histogram.Xbins, not {type(bin_size)}"
        )
        raise TypeError(error_message)

    hist = go.Histogram(
        x=values,
        xbins=xbins,
        histnorm="probability",
    )
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
    else:
        additional_text = f"<br><br>Min: -<br>Max: -<br>|Max-Min|: -"
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
        title=title,
        margin={"t": 40 * n_lines, "b": 0},
        xaxis=dict(range=[0, x_max + bin_size]),
    )
    return fig


def make_computation_time_histogram(
    estimation_results: List["EstimationResult"],
    num_data: List[int],
    num_data_index: int,
    unit: str = "sec",
):
    time_unit = 1
    if unit == "min":
        time_unit = 60
    elif unit == "sec":
        time_unit = 1
    else:
        error_message = f"'unit' must be 'min' or 'sec', not {unit}"
        raise ValueError(error_message)

    n = num_data[num_data_index]
    values = [
        result.computation_times[num_data_index] / time_unit
        for result in estimation_results
    ]
    fig = make_histogram(
        values,
        num_data=n,
        annotation_vlines=[np.mean(values)],
        x_abs_min=0.5,
        xaxis_title_text=f"Time ({unit})",
        additional_title_text=f"<br>Mean = {np.mean(values)}({unit})",
    )
    return fig


def make_computation_time_histograms(
    estimation_results: List["EstimationResult"], num_data: List[int], unit: str = "sec"
):
    num_data_len = len(num_data)

    figs = []
    for i in range(num_data_len):
        fig = make_computation_time_histogram(
            estimation_results, num_data=num_data, num_data_index=i, unit=unit
        )
        figs.append(fig)
    return figs
