import tempfile
import shutil

from typing import List, Optional
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from xhtml2pdf import pisa
from xhtml2pdf.config.httpconfig import httpConfig

from quara.data_analysis import (
    physicality_violation_check,
    data_analysis,
)
from quara.data_analysis import computation_time as ctime
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate
from quara.protocol.qtomography.estimator import EstimationResult
from quara.simulation import consistency_check
from quara.simulation import standard_qtomography_simulation_check
from quara.simulation.standard_qtomography_simulation import (
    SimulationResult,
    StandardQTomographySimulationSetting,
)

_temp_dir_path = ""

_css = f"""
body {{color: #666666;}}
h1 {{
    line-height: 100%;
    border-top: 2px #dcdcdc solid;
    padding: 20px 0 0 0;
    font-size: 25px;}}
h2 {{font-size: 20px;
line-height:90%;
padding: 5px 0 5px 0;
margin: 10px 0 0 0;}}
h3 {{font-size: 15px;
    color: #618CBC;
    line-height:90%;
    padding: 5px 0 5px 0;
    margin: 2px 0 0 0;}}
h4 {{color:#EB9348;
font-size: 15px;
-pdf-outline: false;
line-height:90%;
padding: 5px 0 5px 0;
margin: 0 0 0 0;
}}
h5 {{color:#666666;
font-size: 13px;
-pdf-outline: false;
padding: 0 0 0 0;
margin: 0 0 0 0;
line-height:90%;
vertical-align: text-bottom;}}
h6 {{color:#666666;
font-size: 13px;
font-style:italic;
-pdf-outline: false;
padding: 0 0 0 0;
margin: 0 0 0 0;}}
#footer_content {{text-align: right;}}
"""

_table_css = """
table{
  border: solid 1px #d3d3d3;
  border-collapse: collapse;
  border-spacing: 0;
  table-layout: fixed;
  width:100%;
}

table tr{
  border: solid 1px #d3d3d3;
}

table th{
  text-align: right;
  background-color: #666666;
  color: #ffffff;
  border: solid 1px #ffffff;
  font-size: 13px;
  width: 100px;
  padding-top: 3px;
  padding-right: 3px;
}

table td{
  text-align: right;
  font-size: 13px;
  padding-top: 3px;
  padding-right: 3px;
  width: 400px;
  word-break: break-all;
}

.comp_time_table {
    width: 380px;
}
.comp_time_table th{
    width: 50px;
}
"""

_table_contents_css = """
pdftoc {
    color: #666;
}
pdftoc.pdftoclevel0 {
    font-weight: bold;
    margin-top: 0.5em;
}
pdftoc.pdftoclevel1 {
    margin-left: 1em;
}
pdftoc.pdftoclevel2 {
    margin-left: 2em;
}
pdftoc.pdftoclevel3 {
    margin-left: 3em;
    font-style: italic;
}
"""

_inline_block_css = """
.box{
 display: inline-block;
 width: 400px;
}

.box_col2{
 display: inline-block;
 width: 400px;
 padding: 0;
}

.box_col3{
 display: inline-block;
 width: 250px;
 padding: 0;
}

.box_col4{
 display: inline-block;
 width: 190px;
 padding: 0;
}

.div_line{
    padding: 0 0 0 0;
    margin: 0 0 35px 0;
}
"""

_col2_fig_width = 500
_col2_fig_height = 400


def _convert_html2pdf(source_html: str, output_path: str):
    httpConfig.save_keys("nosslcheck", True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w+b") as f:
        pisa_status = pisa.CreatePDF(source_html, dest=f)
    return pisa_status.err


def _save_fig_to_tmp_dir(fig: "Figure", fig_name: str) -> str:
    dir_path = Path(_temp_dir_path)
    path = str(dir_path / f"{fig_name}.png")
    dir_path.mkdir(exist_ok=True)
    fig.write_image(path)

    return path


def _make_graph_trace_seq(
    estimation_results: List["EstimationResult"], num_data: List[int], case_id: int
) -> list:
    fig_info_list = []
    for i, num in enumerate(num_data):
        fig = physicality_violation_check.make_graph_trace(
            estimation_results, num_data_index=i, num_data=num_data
        )

        fig_name = f"case={case_id}_trace_num={num}_0"
        fig.update_layout(width=_col2_fig_width, height=_col2_fig_height)

        path = _save_fig_to_tmp_dir(fig, fig_name)

        fig_info_list.append(dict(image_path=path, fig=fig, fig_name=fig_name))
    return fig_info_list


def _generate_trace_div(fig_info_list: List[dict]) -> str:
    col_n = len(fig_info_list) if len(fig_info_list) <= 4 else 4
    css_class = f"box_col{col_n}"
    div_lines = []
    div_line = ""
    for i, fig_info in enumerate(fig_info_list):
        div_line += f"<div class='{css_class}'><img src={fig_info['image_path']}></div>"

        if i % col_n == col_n - 1:
            div_lines.append(f"<div class='div_line'>{div_line}</div>")
            div_line = ""
    else:
        if div_line:
            div_lines.append(f"<div class='div_line'>{div_line}</div>")

    graph_block_html = f"<div class='div_line'>{''.join(div_lines)}</div>"
    return graph_block_html


def generate_trace_div(
    estimation_results: List["EstimationResult"], num_data: List[int], case_id: int
):
    fig_info_list = _make_graph_trace_seq(
        estimation_results, num_data=num_data, case_id=case_id
    )
    div_html = _generate_trace_div(fig_info_list)
    return div_html


def _make_graph_sum_vecs_seq(
    estimation_results: List["EstimationResult"],
    num_data: List[int],
    case_id: int,
    true_object: Povm,
) -> List[List["Figure"]]:
    fig_info_list_list = []

    for num_data_index, num in enumerate(num_data):
        figs = physicality_violation_check.make_graphs_sum_vecs(
            estimation_results,
            true_object,
            num_data=num_data,
            num_data_index=num_data_index,
        )
        fig_info_list = []
        for alpha, fig in enumerate(figs):
            fig_name = f"case={case_id}_trace_num={num}_alpha={alpha}"
            fig.update_layout(width=_col2_fig_width, height=_col2_fig_height)
            path = _save_fig_to_tmp_dir(fig, fig_name)

            fig_info_list.append(
                dict(image_path=path, fig=fig, fig_name=fig_name, num=num, alpha=alpha)
            )
        fig_info_list_list.append(fig_info_list)

    return fig_info_list_list


def _generate_fig_info_list_list_div(
    fig_info_list_list: List[List[dict]], col_n=2
) -> str:
    graph_block_html_all = ""
    css_class = "box" if col_n <= 2 else "box_col4"

    for fig_info_list in fig_info_list_list:  # num
        num = fig_info_list[0]["num"]
        graph_block_html = f"<h5>N={num}</h5>"
        div_lines = []
        div_line = ""
        for i, fig_info in enumerate(fig_info_list):  # alpha
            div_line += (
                f"<div class='{css_class}'><img src={fig_info['image_path']}></div>"
            )
            if i % col_n == col_n - 1:
                div_lines.append(f"<div class='div_line'>{div_line}</div>")
                div_line = ""
        else:
            div_lines.append(f"<div class='div_line'>{div_line}</div>")
        graph_block_html_all += graph_block_html + "".join(div_lines)

    return graph_block_html_all


def generate_sum_vecs_div(
    estimation_results: List["EstimationResult"],
    num_data: List[int],
    case_id: int,
    true_object: Povm,
    col_n: int,
):
    fig_info_list_list = _make_graph_sum_vecs_seq(
        estimation_results, num_data=num_data, case_id=case_id, true_object=true_object
    )
    div_html = _generate_fig_info_list_list_div(fig_info_list_list, col_n=col_n)
    return div_html


def _generate_graph_eigenvalues_seq(
    estimation_results: List["EstimationResult"],
    num_data: List[int],
    case_id: int,
    true_object: "QOperation",
    bin_size: float = 0.0001,
) -> list:

    fig_info_list_list = []
    for num_data_index in range(len(num_data)):
        fig_list = physicality_violation_check.make_graphs_eigenvalues(
            estimation_results,
            true_object,
            num_data=num_data,
            num_data_index=num_data_index,
            bin_size=bin_size,
        )
        fig_info_list = []
        num = num_data[num_data_index]

        for i, fig in enumerate(fig_list):
            fig_name = f"case={case_id}_eigenvalues_num={num_data_index}_i={i}"
            fig.update_layout(width=_col2_fig_width, height=_col2_fig_height)

            path = _save_fig_to_tmp_dir(fig, fig_name)

            fig_info_list.append(
                dict(image_path=path, fig=fig, fig_name=fig_name, num=num)
            )

        fig_info_list_list.append(fig_info_list)
    return fig_info_list_list


def _generate_eigenvalues_div(
    fig_info_list_list: List[List[dict]], col_n: int = 2
) -> str:
    graph_block_html_all = ""
    for fig_info_list in fig_info_list_list:
        num = fig_info_list[0]["num"]
        graph_block_html = f"<h5>N={num}</h5>"
        graph_block_html = _generate_figs_div(fig_info_list, col_n=col_n)
        graph_block_html_all += graph_block_html

    return graph_block_html_all


def _generate_eigenvalues_div_3loop(
    fig_info_list3: List[List[List[dict]]], col_n: int
) -> str:
    graph_block_html_all = ""
    fig_n = fig_info_list3[0][0]
    col_n = len(fig_n) if len(fig_n) <= 4 else 4
    css_class = f"box_col{col_n}"

    for fig_info_list2 in fig_info_list3:  # num_data
        num = fig_info_list2[0][0]["num"]
        graph_block_html = f"<h5>N={num}</h5>"

        for fig_info_list in fig_info_list2:  # measurement
            x_i = fig_info_list[0]["x"]
            div_lines = []
            div_line = ""

            for i, fig_info in enumerate(fig_info_list):
                div_line += (
                    f"<div class='{css_class}'><img src={fig_info['image_path']}></div>"
                )

                if i % col_n == col_n - 1:
                    div_lines.append(f"<div class='div_line'>{div_line}</div>")
                    div_line = ""
            else:
                if div_line:
                    div_lines.append(f"<div class='div_line'>{div_line}</div>")

            graph_block_html += f"<h6>x={x_i}</h6>" + "".join(div_lines)

        graph_block_html_all += graph_block_html

    return graph_block_html_all


def _generate_graph_eigenvalues_seq_3loop(
    estimation_results: List["EstimationResult"],
    num_data: List[int],
    case_id: int,
    true_object: "QOperation",
) -> list:
    # For State
    fig_info_list3 = []
    for num_data_index in range(len(num_data)):
        fig_list_list = physicality_violation_check.make_graphs_eigenvalues(
            estimation_results,
            true_object,
            num_data=num_data,
            num_data_index=num_data_index,
        )
        fig_info_list2 = []

        for x_i, fig_list in enumerate(fig_list_list):
            fig_info_list = []
            for i, fig in enumerate(fig_list):
                fig_name = (
                    f"case={case_id}_eigenvalues_num={num_data_index}_x={x_i}_i={i}"
                )
                fig.update_layout(width=_col2_fig_width, height=_col2_fig_height)
                path = _save_fig_to_tmp_dir(fig, fig_name)

                fig_info = dict(
                    image_path=path,
                    fig=fig,
                    fig_name=fig_name,
                    num=num_data[num_data_index],
                    x=x_i,
                    i=i,
                )
                fig_info_list.append(fig_info)
            fig_info_list2.append(fig_info_list)
        fig_info_list3.append(fig_info_list2)
    return fig_info_list3


def generate_eigenvalues_div(
    estimation_results: List["EstimationResult"],
    num_data: List[int],
    case_id: int,
    true_object: "QOperation",
):
    if type(true_object) == State:
        fig_info_list_list = _generate_graph_eigenvalues_seq(
            estimation_results,
            num_data=num_data,
            case_id=case_id,
            true_object=true_object,
        )
        vals = true_object.calc_eigenvalues()
        col_n = 2 if len(vals) <= 2 else 4
        div_html = _generate_eigenvalues_div(fig_info_list_list, col_n=col_n)
    elif type(true_object) == Povm:
        fig_info_list3 = _generate_graph_eigenvalues_seq_3loop(
            estimation_results,
            num_data=num_data,
            case_id=case_id,
            true_object=true_object,
        )
        vals = true_object.calc_eigenvalues()
        col_n = 2 if len(vals[0]) <= 2 else 4
        div_html = _generate_eigenvalues_div_3loop(fig_info_list3, col_n=col_n)
    elif type(true_object) == Gate:
        fig_info_list_list = _generate_graph_eigenvalues_seq(
            estimation_results,
            num_data=num_data,
            case_id=case_id,
            true_object=true_object,
        )
        v, _ = np.linalg.eig(true_object.to_choi_matrix())
        col_n = 2 if len(v) <= 2 else 4
        div_html = _generate_eigenvalues_div(fig_info_list_list, col_n=col_n)
    else:
        raise TypeError()
    return div_html


def _generate_graph_sum_eigenvalues_seq(
    estimation_results: List["EstimationResult"],
    num_data: List[int],
    case_id: int,
    true_object,
) -> List[List[dict]]:
    fig_info_list_list = []
    for num_data_index in range(len(num_data)):
        fig_list = physicality_violation_check.make_graphs_sum_unphysical_eigenvalues(
            estimation_results,
            num_data=num_data,
            num_data_index=num_data_index,
        )
        n_unphysical = physicality_violation_check.calc_unphysical_qobjects_n(
            estimation_results, num_data_index=num_data_index
        )
        fig_info_list = []

        for i, fig in enumerate(fig_list):
            fig_name = f"case={case_id}_sum-unphysical-eigenvalues_num={num_data_index}_type={i}"
            fig.update_layout(width=_col2_fig_width, height=_col2_fig_height)
            path = _save_fig_to_tmp_dir(fig, fig_name)

            fig_info_list.append(
                dict(
                    image_path=path,
                    fig=fig,
                    fig_name=fig_name,
                    num=num_data[num_data_index],
                    n_unphysical=n_unphysical,
                )
            )

        fig_info_list_list.append(fig_info_list)
    return fig_info_list_list


def _generate_sum_eigenvalues_div(fig_info_list_list: List[List[dict]]) -> str:
    graph_block_html_all = ""
    for fig_info_list in fig_info_list_list:
        num = fig_info_list[0]["num"]
        n_unphysical = fig_info_list[0]["n_unphysical"]
        graph_block_html = (
            f"<h5>N={num}<br>Number of unphysical estimates={n_unphysical}</h5>"
        )

        for fig_info in fig_info_list:
            graph_subblock = (
                f"<div class='box'><img src={fig_info['image_path']}></div>"
            )
            graph_block_html += graph_subblock

        graph_block_html_all += f"<div class='div_line'>{graph_block_html}</div>"

    return graph_block_html_all


def generate_sum_eigenvalues_div(
    estimation_results: List["EstimationResult"],
    num_data: List[int],
    case_id: int,
    true_object,
):
    fig_info_list_list = _generate_graph_sum_eigenvalues_seq(
        estimation_results, num_data=num_data, case_id=case_id, true_object=true_object
    )
    div_html = _generate_sum_eigenvalues_div(fig_info_list_list)
    return div_html


def _calc_legend_y(num_legend):
    return -0.07 * num_legend - 0.1


def generate_mse_analytical_div(
    estimation_results_list: List[List[EstimationResult]],
    true_object: "QOperation",
    estimator_list: list,
    num_data: List[int],
    qtomography_list,
) -> str:
    figs = data_analysis.make_mses_graph_analytical(
        estimation_results_list=estimation_results_list,
        true_object=true_object,
        estimator_list=estimator_list,
        num_data=num_data,
        qtomography_list=qtomography_list,
    )

    mse_div_list = []
    for i, fig in enumerate(figs):
        fig_name = f"mse_analytical_{i}"
        fig.update_layout(width=600, height=600)
        num_legend = len(fig.data)
        legend_y = _calc_legend_y(num_legend)
        fig.update_layout(
            legend=dict(yanchor="bottom", y=legend_y, xanchor="left", x=0)
        )
        path = _save_fig_to_tmp_dir(fig, fig_name)

        if i % 2 == 0:
            mse_div_list.append("<div>")

        mse_div = f"<div class='box'><img src='{path}'></div>"
        mse_div_list.append(mse_div)

        if i % 2 == 1 or i + 1 == len(figs):
            mse_div_list.append("</div>")
    return "".join(mse_div_list)


def generate_empi_dist_mse_div(
    simulation_result: SimulationResult,
    true_object: "QOperation",
) -> str:

    # fig = data_analysis.make_empi_dists_mse_graph(
    #     estimation_results_list[0], true_object
    # )
    fig = data_analysis.make_empi_dists_mse_graph(simulation_result, true_object)

    fig_name = f"empi_dists_mse"
    path = _save_fig_to_tmp_dir(fig, fig_name)

    div = f"<img src='{path}'>"
    return div


def _convert_object_to_datafrane(qoperation: "QOperation") -> pd.DataFrame:
    values = [v.__str__().replace("\n", "<br>") for v in qoperation._info().values()]
    item_names = qoperation._info().keys()
    df = pd.DataFrame(values, item_names).rename(columns={0: "value"})

    return df


def _convert_objects_to_multiindex_dataframe(
    qoperations: List["QOperation"],
) -> pd.DataFrame:
    df_dict = {}

    for i in range(len(qoperations)):
        df_dict[i] = _convert_object_to_datafrane(qoperations[i])

    objects_df_multiindex = pd.concat(df_dict, axis=0)
    return objects_df_multiindex


def _generate_physicality_violation_test_div_for_state(
    estimation_results_list: List[List["EstimationResult"]],
    num_data: List[int],
    case_name_list: List[str],
    true_object: State,
):
    test_eq_const_divs = ""
    test_ineq_const_eigenvalues_divs = ""
    test_ineq_const_sum_eigenvalues_divs = ""

    for case_id, case_name in enumerate(case_name_list):
        estimation_results = estimation_results_list[case_id]
        # Test of equality constraint violation
        div = generate_trace_div(estimation_results, num_data=num_data, case_id=case_id)

        # <h5> is dummy
        test_eq_const_divs += f"""
            <h4>Case {case_id}: {case_name}<h4>
            <h5></h5>
            {div}
            """
        # Test of inequality constraint violation
        div = generate_eigenvalues_div(
            estimation_results,
            num_data=num_data,
            case_id=case_id,
            true_object=true_object,
        )
        test_ineq_const_eigenvalues_divs += f"""
            <h4>Case {case_id}: {case_name}<h4>
            {div}
            """

        div = generate_sum_eigenvalues_div(
            estimation_results,
            num_data=num_data,
            case_id=case_id,
            true_object=true_object,
        )
        test_ineq_const_sum_eigenvalues_divs += f"""
            <h4>Case {case_id}: {case_name}<h4>
            {div}
            """

    eq_all_div = f"""
        <h2>Test of equality constraint violation</h2>
        {test_eq_const_divs}
    """
    ineq_all_div = f"""
        <h2>Test of inequality constraint violation</h2>
        <h3>Eigenvalue</h3>
        {test_ineq_const_eigenvalues_divs}
        <h3>Sum of unphysical eigenvalues </h3>
        {test_ineq_const_sum_eigenvalues_divs}
    """

    return eq_all_div, ineq_all_div


def _generate_physicality_violation_test_div_for_povm(
    estimation_results_list: List[List["EstimationResult"]],
    num_data: List[int],
    case_name_list: List[str],
    true_object: Povm,
):
    test_eq_const_divs = ""
    test_ineq_const_eigenvalues_divs = ""
    test_ineq_const_sum_eigenvalues_divs = ""

    for case_id, case_name in enumerate(case_name_list):
        estimation_results = estimation_results_list[case_id]
        # Test of equality constraint violation
        div = generate_sum_vecs_div(
            estimation_results,
            num_data=num_data,
            case_id=case_id,
            true_object=true_object,
            col_n=4,
        )
        # <h5> is dummy
        test_eq_const_divs += f"""
            <h4>Case {case_id}: {case_name}<h4>
            <h5></h5>
            {div}
            """
        # Test of inequality constraint violation
        div = generate_eigenvalues_div(
            estimation_results,
            num_data=num_data,
            case_id=case_id,
            true_object=true_object,
        )
        test_ineq_const_eigenvalues_divs += f"""
            <h4>Case {case_id}: {case_name}<h4>
            {div}
            """

        div = generate_sum_eigenvalues_div(
            estimation_results,
            num_data=num_data,
            case_id=case_id,
            true_object=true_object,
        )
        test_ineq_const_sum_eigenvalues_divs += f"""
            <h4>Case {case_id}: {case_name}<h4>
            {div}
            """

    eq_all_div = f"""
        <h2>Test of equality constraint violation</h2>
        {test_eq_const_divs}
    """
    ineq_all_div = f"""
        <h2>Test of inequality constraint violation</h2>
        <h3>Eigenvalue</h3>
        {test_ineq_const_eigenvalues_divs}
        <h3>Sum of unphysical eigenvalues </h3>
        {test_ineq_const_sum_eigenvalues_divs}
    """

    return eq_all_div, ineq_all_div


def _generate_physicality_violation_test_div_for_gate(
    estimation_results_list: List[List["EstimationResult"]],
    num_data: List[int],
    case_name_list: List[str],
    true_object: State,
):
    test_eq_const_divs = ""
    test_eq_const_error_sum_divs = ""
    test_ineq_const_eigenvalues_divs = ""
    test_ineq_const_sum_eigenvalues_divs = ""

    for case_id, case_name in enumerate(case_name_list):
        estimation_results = estimation_results_list[case_id]
        # Test of equality constraint violation
        div = generate_fig_list_list_div(
            estimation_results=estimation_results,
            case_id=case_id,
            fig_type="physicality-violation-eq-trace-error",
            make_graphs_func=physicality_violation_check.make_graphs_trace_error,
            col_n=4,
            num_data=num_data,
        )
        # <h5> is dummy
        test_eq_const_divs += f"""
            <h4>Case {case_id}: {case_name}<h4>
            <h5></h5>
            {div}
            """
        num_data_len = len(num_data)
        col_n = num_data_len if num_data_len <= 4 else 4

        div = generate_figs_div(
            func=_make_fig_info_list,
            estimation_results=estimation_results,
            case_id=case_id,
            fig_type="physicality-violation-eq-trace-sum-error",
            size=(_col2_fig_width, _col2_fig_height),
            make_graphs_func=physicality_violation_check.make_graphs_trace_error_sum,
            col_n=col_n,
            num_data=num_data,
        )

        test_eq_const_error_sum_divs += f"""
            <h4>Case {case_id}: {case_name}<h4>
            <h5></h5>
            {div}
        """
        # Test of inequality constraint violation
        div = generate_eigenvalues_div(
            estimation_results,
            num_data=num_data,
            case_id=case_id,
            true_object=true_object,
        )
        test_ineq_const_eigenvalues_divs += f"""
            <h4>Case {case_id}: {case_name}<h4>
            {div}
            """

        div = generate_sum_eigenvalues_div(
            estimation_results,
            num_data=num_data,
            case_id=case_id,
            true_object=true_object,
        )
        test_ineq_const_sum_eigenvalues_divs += f"""
            <h4>Case {case_id}: {case_name}<h4>
            {div}
            """

    eq_all_div = f"""
        <h2>Test of equality constraint violation</h2>
        <h3>Error</h3>
        {test_eq_const_divs}
        <h3>Sum of Error</h3>
        {test_eq_const_error_sum_divs}
    """
    ineq_all_div = f"""
        <h2>Test of inequality constraint violation</h2>
        <h3>Eigenvalue</h3>
        {test_ineq_const_eigenvalues_divs}
        <h3>Sum of unphysical eigenvalues </h3>
        {test_ineq_const_sum_eigenvalues_divs}
    """

    return eq_all_div, ineq_all_div


def generate_physicality_violation_test_div(
    estimation_results_list: List[List["EstimationResult"]],
    case_name_list: List[str],
    true_object: "QOperation",
    num_data: List[int],
):
    if type(true_object) == State:
        (
            true_all_div,
            false_all_div,
        ) = _generate_physicality_violation_test_div_for_state(
            estimation_results_list, num_data, case_name_list, true_object
        )
    elif type(true_object) == Povm:
        (
            true_all_div,
            false_all_div,
        ) = _generate_physicality_violation_test_div_for_povm(
            estimation_results_list, num_data, case_name_list, true_object
        )
    elif type(true_object) == Gate:
        (
            true_all_div,
            false_all_div,
        ) = _generate_physicality_violation_test_div_for_gate(
            estimation_results_list, num_data, case_name_list, true_object
        )
    else:
        message = f"true_object must be State, Povm, or Gate, not {type(true_object)}"
        raise TypeError(message)

    physicality_violation_test_div = f"""
        {true_all_div}
        {false_all_div}
    """

    return physicality_violation_test_div


def generate_case_table(
    case_name_list: List["str"],
    qtomography_list: List["QTomography"],
    estimator_list: List["Estimator"],
):
    para_list = [qtomo.on_para_eq_constraint for qtomo in qtomography_list]
    case_dict = dict(
        Name=case_name_list,
        Param=para_list,
        Tomography=[t.__class__.__name__ for t in qtomography_list],
        Estimator=[
            e.__class__.__name__.replace("Estimator", "") for e in estimator_list
        ],
    )

    case_df = pd.DataFrame(case_dict)
    styles = [
        dict(selector=".col0", props=[("width", "400px")]),
        dict(selector=".col1", props=[("width", "180px")]),
        dict(selector=".col2", props=[("width", "200px")]),
    ]
    case_table = case_df.style.set_table_styles(styles).render()
    return case_table


def generate_condition_table(
    qtomography_list: List["QTomography"],
    n_rep: int,
    num_data: List[int],
    seed: Optional[int],
) -> str:
    type_tomography_values = list(
        set([qt.__class__.__name__ for qt in qtomography_list])
    )

    info = {
        "Type of tomography": type_tomography_values,
        "Nrep": [n_rep],
        "N": [num_data],
        "RNG seed": [seed],
    }
    condition_df = pd.DataFrame(info).T
    condition_table = condition_df.to_html(
        classes="condition_table", escape=False, header=False
    )
    return condition_table


def generate_consistency_check_table(simulation_results: List[SimulationResult]):
    check_results = []
    simulation_settings = []
    for sim_result in simulation_results:
        check_results.append(sim_result.check_result)
        simulation_settings.append(sim_result.simulation_setting)

    qtomography_list = [sim_result.qtomography for sim_result in simulation_results]
    result_list = []
    para_list = [qtomo.on_para_eq_constraint for qtomo in qtomography_list]

    if len(check_results) == len(simulation_results):
        # Use the results of pre-run checks
        def _extract_consistency_check_results(check_result: "CheckResult") -> dict:
            for r in check_result["results"]:
                if r["name"] == "Consistency":
                    return r["detail"]

        result_list = [_extract_consistency_check_results(cr) for cr in check_results]
    else:
        # Execute Consistency Check
        for sim_result in simulation_results:
            sim_check = standard_qtomography_simulation_check.StandardQTomographySimulationCheck(
                sim_result
            )
            result_dict = sim_check.execute_consistency_check(show_detail=False)
            result_list.append(result_dict)

    def _insert_white_space(text: str) -> str:
        # If there is an upper case, insert a half-width space.
        # Before: LossMinimization
        # After: Loss Minimization
        # Line breaks are not applied if there is no half-width space.
        converted = text[0]
        for char in text[1:]:
            if char.isupper():
                converted += f" {char}"
            else:
                converted += char
        return converted

    type_tomography_values = [
        _insert_white_space(qt.__class__.__name__) for qt in qtomography_list
    ]
    type_estimator_values = [
        _insert_white_space(s.estimator.__class__.__name__.replace("Estimator", ""))
        for s in simulation_settings
    ]

    type_loss_values = [
        _insert_white_space(s.loss.__class__.__name__) if s.loss else "None"
        for s in simulation_settings
    ]
    type_algo_values = [
        _insert_white_space(s.algo.__class__.__name__) if s.algo else "None"
        for s in simulation_settings
    ]
    result_dict = {
        "Name": [s.name for s in simulation_settings],
        "Type of tomography": type_tomography_values,
        "Param": para_list,
        "Estimator": type_estimator_values,
        "Loss": type_loss_values,
        "Algo": type_algo_values,
        "Squared Error to True": [
            f"{r['squared_error_to_true']:.2e}" for r in result_list
        ],
        "Possibly OK": [f"{'OK' if r['possibly_ok'] else 'NG'}" for r in result_list],
        "To be checked": [
            f"{'need debug' if r['to_be_checked'] else 'not need debug'}"
            for r in result_list
        ],
    }

    styles = [
        dict(selector=".col0", props=[("width", "400px"), ("font-size", "10px")]),
        dict(selector=".col1", props=[("width", "250px"), ("font-size", "10px")]),
        dict(selector=".col2", props=[("width", "120px"), ("font-size", "10px")]),
        dict(selector=".col3", props=[("width", "200px"), ("font-size", "10px")]),
        dict(selector=".col4", props=[("width", "300px"), ("font-size", "10px")]),
        dict(selector=".col5", props=[("width", "300px"), ("font-size", "10px")]),
        dict(selector=".col6", props=[("width", "150px"), ("font-size", "10px")]),
        dict(selector=".col7", props=[("width", "150px"), ("font-size", "10px")]),
        dict(selector=".col8", props=[("width", "150px"), ("font-size", "10px")]),
    ]

    table_df = pd.DataFrame(result_dict)
    consistency_check_table = table_df.style.set_table_styles(styles).render()
    return consistency_check_table


def generate_computation_time_table(
    estimation_results_list: List[List["EstimationResult"]],
) -> pd.DataFrame:
    total_time = 0
    for results in estimation_results_list:
        total_time += sum([sum(r.computation_times) for r in results])
    computation_time_text = "{0}".format(total_time / 60) + "min."

    info = {
        "Total": [computation_time_text],
    }

    computation_time_table = pd.DataFrame(info).T.to_html(
        classes="computation_time_table", escape=False, header=False
    )

    return computation_time_table


def generate_tolerance_table_div() -> pd.DataFrame:
    data = [
        [
            physicality_violation_check.get_eq_const_eps(True),
            physicality_violation_check.get_eq_const_eps(False),
        ],
        [physicality_violation_check.get_ineq_const_eps()] * 2,
    ]
    first_index = "Tolerance at physicality violation test"
    index = [[first_index] * 2, ["equality constraint", "inequality constraint"]]
    columns = ["True", "False"]
    df = pd.DataFrame(data, index=index, columns=columns)
    df = df.applymap(lambda x: f"{x:.2e}")

    styles = [
        dict(selector=".col0", props=[("width", "100px")]),
        dict(selector=".col1", props=[("width", "100px")]),
    ]

    tolerance_table = df.style.set_table_styles(styles).render()

    tolerance_table_div = f"""
        <h1>Tolerance of physicality constraint violation</h1>
    <div>
        {tolerance_table}
    </div>
        """

    return tolerance_table_div


def _make_graphs_mses(make_graphs_func, mse_type: "str", **kwargs) -> list:
    figs = make_graphs_func(**kwargs)
    fig_info_list = []

    for i, fig in enumerate(figs):
        fig_name = f"mse_type={mse_type}_{i}"
        fig.update_layout(width=600, height=600)
        num_legend = len(fig.data)
        legend_y = _calc_legend_y(num_legend)
        fig.update_layout(
            legend=dict(yanchor="bottom", y=legend_y, xanchor="left", x=0)
        )
        path = _save_fig_to_tmp_dir(fig, fig_name)
        fig_info_list.append(dict(image_path=path, fig=fig, fig_name=fig_name))
    return fig_info_list


def _make_fig_info_list(
    make_graphs_func, fig_type: "str", case_id: int = None, size=(600, 600), **kwargs
) -> list:
    arg_names = make_graphs_func.__code__.co_varnames[
        : make_graphs_func.__code__.co_argcount
    ]
    new_kwargs = {k: v for k, v in kwargs.items() if k in arg_names}

    figs = make_graphs_func(**new_kwargs)
    fig_info_list = []
    if type(figs) != list:
        figs = [figs]

    for i, fig in enumerate(figs):
        fig_name = f"fig_type={fig_type}_{i}"
        if case_id is not None:
            fig_name = f"case={case_id}_{fig_name}"
        fig.update_layout(width=size[0], height=size[1])
        fig.update_layout(legend=dict(yanchor="bottom", y=-0.5, xanchor="left", x=0))
        path = _save_fig_to_tmp_dir(fig, fig_name)
        fig_info_list.append(dict(image_path=path, fig=fig, fig_name=fig_name))
    return fig_info_list


def _make_fig_info_list_list(
    estimation_results: List["EstimationResult"],
    num_data: List[int],
    case_id: int,
    fig_type: str,
    make_graphs_func,
    **kwargs,
) -> List[List["Figure"]]:
    fig_info_list_list = []

    for num_data_index, num in enumerate(num_data):
        func_parameter_names = make_graphs_func.__code__.co_varnames[
            : make_graphs_func.__code__.co_argcount
        ]
        if "num_data" in func_parameter_names:
            figs = make_graphs_func(
                estimation_results=estimation_results,
                num_data=num_data,
                num_data_index=num_data_index,
                **kwargs,
            )
        else:
            figs = make_graphs_func(
                estimation_results=estimation_results,
                num_data_index=num_data_index,
                **kwargs,
            )
        if type(figs) != list:
            figs = [figs]
        fig_info_list = []
        for alpha, fig in enumerate(figs):
            fig_name = f"case={case_id}_{fig_type}_num={num}_alpha={alpha}"
            fig.update_layout(width=_col2_fig_width, height=_col2_fig_height)
            path = _save_fig_to_tmp_dir(fig, fig_name)

            fig_info_list.append(
                dict(image_path=path, fig=fig, fig_name=fig_name, num=num, alpha=alpha)
            )
        fig_info_list_list.append(fig_info_list)

    return fig_info_list_list


def _generate_figs_div(fig_info_list: List[dict], col_n: int = 2) -> str:
    graph_block_html = ""
    subblock_list = []
    css_class = "box" if col_n <= 2 else "box_col4"
    for fig_info in fig_info_list:
        graph_subblock = (
            f"<div class='{css_class}'><img src={fig_info['image_path']}></div>"
        )
        subblock_list.append(graph_subblock)

    div_line = ""
    div_lines = []
    for i, block in enumerate(subblock_list):
        div_line += block
        if i % col_n == col_n - 1:
            div_lines.append(f"<div class='div_line'>{div_line}</div>")
            div_line = ""
    else:
        if div_line:
            div_lines.append(f"<div class='div_line'>{div_line}</div>")

    graph_block_html = "".join(div_lines)
    return graph_block_html


def generate_fig_list_list_div(
    estimation_results: List["EstimationResult"],
    num_data,
    case_id,
    fig_type: str,
    make_graphs_func,
    col_n: int = 2,
    **kwargs,
):
    fig_info_list_list = _make_fig_info_list_list(
        estimation_results, num_data, case_id, fig_type, make_graphs_func, **kwargs
    )
    div_html = _generate_fig_info_list_list_div(fig_info_list_list, col_n=col_n)
    return div_html


def generate_figs_div(func, **kwargs):
    fig_info_list = func(**kwargs)
    if "col_n" in kwargs:
        col_n = kwargs["col_n"]
        div_html = _generate_figs_div(fig_info_list, col_n=col_n)
    else:
        div_html = _generate_figs_div(fig_info_list)
    return div_html


def generate_computation_time_of_estimators_table(
    estimation_results_list, simulation_settings, unit: str = "sec"
) -> str:
    def _generate_computation_time_df(
        estimation_results: list, name, unit
    ) -> pd.DataFrame:
        n_rep = len(estimation_results)
        if unit == "min":
            time_unit = 60
        elif unit == "sec":
            time_unit = 1
        else:
            raise ValueError("'unit' must be 'sec' or 'min'.")

        num_list = []
        mean_list = []
        std_list = []

        num_data = simulation_settings[0].num_data
        for i, num in enumerate(num_data):
            comp_times = [result.computation_times[i] for result in estimation_results]
            num_list.append(num)
            mean_list.append(np.mean(comp_times) / time_unit)
            std_list.append(np.std(comp_times) / time_unit)

        data_dict = {
            "Name": [name] + ['   "   '] * (len(num_list) - 1),
            "N": num_list,
            "Nrep": [n_rep] * len(num_list),
            f"Mean ({unit})": mean_list,
            f"Std ({unit})": std_list,
        }
        time_df = pd.DataFrame(data_dict)
        return time_df

    styles = [
        dict(selector=".col0", props=[("width", "250px")]),
        dict(selector=".col1", props=[("width", "100px")]),
        dict(selector=".col2", props=[("width", "100px")]),
        dict(selector=".col3", props=[("width", "100px")]),
    ]

    df_list = []
    for i, s in enumerate(simulation_settings):
        time_df = _generate_computation_time_df(
            estimation_results_list[i], s.name, unit=unit
        )
        df_list.append(time_df)

    time_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    time_table = time_df.style.set_table_styles(styles).render()
    time_div = f"<div><h2>Table</h2>{time_table}</div>"
    return time_div


def generate_computation_time_of_estimators_graph(
    estimation_results_list: List[List["EstimationResult"]],
    simulation_settings: List[StandardQTomographySimulationSetting],
) -> str:
    all_divs = ""

    graph_n = len(simulation_settings[0].num_data)

    col_n = graph_n if graph_n <= 4 else 4

    for case_id, simulation_setting in enumerate(simulation_settings):
        case_name = simulation_setting.name
        estimation_results = estimation_results_list[case_id]
        div = generate_figs_div(
            func=_make_fig_info_list,
            estimation_results=estimation_results,
            num_data=simulation_setting.num_data,
            case_id=case_id,
            fig_type="computation_time",
            size=(_col2_fig_width, _col2_fig_height),
            make_graphs_func=ctime.make_computation_time_histograms,
            col_n=col_n,
        )

        # <h5> is dummy
        all_divs += f"""
            <h4>Case {case_id}: {case_name}<h4>
            <h5></h5>
            {div}
        """
    all_divs = f"<h2>Histogram</h2><div>{all_divs}</div>"
    return all_divs


def generate_computation_time_of_estimators_div(
    estimation_results_list: List[List["EstimationResult"]], simulation_settings: list
) -> str:
    div = generate_computation_time_of_estimators_table(
        estimation_results_list, simulation_settings
    )
    div += generate_computation_time_of_estimators_graph(
        estimation_results_list, simulation_settings
    )
    return div


def _load_simulation_results(
    root_dir: str,
    test_setting_index: int,
    sample_index: int,
    case_index: int = None,
) -> list:
    print("Loading SimulationResult pickles ...")
    print(
        f"(test_setting_index, sample_index, case_index) = ({test_setting_index}, {sample_index}, {case_index})"
    )
    result_dir_path = Path(root_dir) / str(test_setting_index) / str(sample_index)
    simulation_results = []
    if case_index is not None:
        # load specific pickle file
        file_name = f"case_{case_index}_result.pickle"
        file_path = result_dir_path / file_name
        print(file_path)
        with open(file_path, "rb") as f:
            simulation_result = pickle.load(f)
        simulation_results.append(simulation_result)
    else:
        # load some pickle files
        file_paths = sorted(result_dir_path.glob("case_*_result.pickle"))
        for file_path in file_paths:
            print(file_path)
            with open(file_path, "rb") as f:
                simulation_result = pickle.load(f)
            simulation_results.append(simulation_result)
    print(
        f"Completed to load SimulationResult pickles. ({len(simulation_results)} files)"
    )
    return simulation_results


def export_report_from_index(
    input_root_dir: str,
    test_setting_index: int,
    sample_index: int,
    output_path: str,
    case_index: int = None,
    display_items: dict = None,
) -> None:
    simulation_results = _load_simulation_results(
        input_root_dir, test_setting_index, sample_index, case_index=case_index
    )

    export_report(
        output_path, simulation_results=simulation_results, display_items=display_items
    )


def setup_display_items(display_items: dict) -> dict:
    expected_items = [
        "consistency",
        "mse_of_estimators",
        "mse_of_empi_dists",
        "physicality_violation",
    ]
    if not display_items:
        display_items = {item: True for item in expected_items}
    else:
        for item in display_items:
            if item not in expected_items:
                error_message = f"The key '{item}' of the argument 'display_items' is invalid. 'display_items' can be used with the following keys: {expected_items}"
                raise KeyError(error_message)

        for item in expected_items:
            if item not in display_items.keys():
                display_items[item] = True
    return display_items


def export_report(
    path: str,
    simulation_results: List[SimulationResult],
    keep_tmp_files: bool = False,
    display_items: dict = None,
):
    """Output a PDF report with simulation settings and results.

    Parameters
    ----------
    path : str
        pdf file path.
    estimation_results_list : List[List[
        List containing a list of estimated results for each simulation.
    simulation_settings : List[StandardQTomographySimulationSetting]
        Settings for each simulation.
    check_results : List[CheckResult]
        CheckResults for each simulation. If SimulationCheck has already been executed, giving the result to this argument will reduce the processing time to generate the report. If not specified, SimulationCheck will be executed during report generation. See the documentation for the StandardQTomographySimulationCheck class to see what SimulationCheck does.
    keep_tmp_files : bool, optional
        [description], by default False
    show_physicality_violation_check : bool, optional
        [description], by default True
    """
    display_items = setup_display_items(display_items)

    estimation_results_list = []
    simulation_settings = []
    for sim_result in simulation_results:
        estimation_results_list.append(sim_result.estimation_results)
        simulation_settings.append(sim_result.simulation_setting)

    temp_dir_path = tempfile.mkdtemp()
    global _temp_dir_path
    _temp_dir_path = Path(temp_dir_path)

    true_object = simulation_settings[0].true_object
    for simulation_setting in simulation_settings:
        # set same CompositeSystem
        simulation_setting.true_object._composite_system = true_object.composite_system

    tester_objects = simulation_settings[0].tester_objects
    for tester_object in tester_objects:
        # set same CompositeSystem
        tester_object._composite_system = true_object.composite_system

    seed = simulation_settings[0].seed_data

    num_data = simulation_results[0].simulation_setting.num_data
    n_rep = simulation_results[0].simulation_setting.n_rep
    qtomography_list = [sim_result.qtomography for sim_result in simulation_results]
    parameter_list = [qtomo.on_para_eq_constraint for qtomo in qtomography_list]
    for qtomography in qtomography_list:
        # set same CompositeSystem
        for qobj in qtomography._experiment.states:
            if qobj:
                qobj._composite_system = true_object.composite_system
        for qobj in qtomography._experiment.gates:
            if qobj:
                qobj._composite_system = true_object.composite_system
        for qobj in qtomography._experiment.povms:
            if qobj:
                qobj._composite_system = true_object.composite_system

    case_name_list = [s.name for s in simulation_settings]
    estimator_list = [s.estimator for s in simulation_settings]

    # Computation Time
    print("​Generating table of computation time ...")
    computation_time_table = generate_computation_time_table(estimation_results_list)

    # Tolerance of physicality constraint violation
    print("​Generating table of tolerance of physicality constraint violation ...")
    tolerance_table_div = generate_tolerance_table_div()

    # Experiment Condition
    print("​Generating table of experimental conditions ...")
    condition_table = generate_condition_table(qtomography_list, n_rep, num_data, seed)

    # True Object
    true_object_table = _convert_object_to_datafrane(true_object).to_html(
        classes="true_object_table", escape=False, header=False
    )
    # Tester Object
    tester_table = _convert_objects_to_multiindex_dataframe(tester_objects).to_html(
        classes="tester_objects_table", escape=False, header=False
    )

    # Cases
    print("Generating case list ...")
    case_table = generate_case_table(case_name_list, qtomography_list, estimator_list)

    # Computation time of estimators
    print("Computation time of estimators ...")
    comp_time_of_est_div = generate_computation_time_of_estimators_div(
        estimation_results_list, simulation_settings
    )

    # MSE of Empirical Distributions
    empi_dists_mse_block = ""
    if display_items["mse_of_empi_dists"]:
        print("​​Generating MSE of empirical distributions blocks ...")
        empi_dists_mse_div = generate_empi_dist_mse_div(
            simulation_results[0], true_object
        )
        empi_dists_mse_block = f"""<h1>MSE of empirical distributions</h1>
    <div>{empi_dists_mse_div}</div>"""

    # Consistency Test
    consistency_check_block = ""
    if display_items["consistency"]:
        print("​​Generating consictency test blocks ...")
        consistency_check_table = generate_consistency_check_table(simulation_results)
        consistency_check_block = f"""<h1>Consistency test</h1>
    <div>{consistency_check_table}</div>"""

    # MSE of estimators
    mse_of_est_block = ""
    if display_items["mse_of_estimators"]:
        print("​Generating a graph for MSE of estimators ...")

        # 1. Comparison of analytical results
        mse_analytical_results_div = generate_mse_analytical_div(
            estimation_results_list,
            true_object,
            estimator_list,
            num_data,
            qtomography_list,
        )
        # 2. Comparison of parametrization
        mse_para_div = generate_figs_div(
            _make_graphs_mses,
            make_graphs_func=data_analysis.make_mses_graphs_estimator,
            mse_type="estimator",
            estimation_results_list=estimation_results_list,
            simulation_settings=simulation_settings,
            true_object=true_object,
            qtomography_list=qtomography_list,
        )
        # 3. Comparison of estimators
        mse_est_div = generate_figs_div(
            _make_graphs_mses,
            make_graphs_func=data_analysis.make_mses_graphs_para,
            mse_type="para",
            estimation_results_list=estimation_results_list,
            case_names=case_name_list,
            true_object=true_object,
            num_data=num_data,
            parameter_list=parameter_list,
            qtomography_list=qtomography_list,
        )
        mse_of_est_block = f"""
<h1>MSE of estimators</h1>
    <h2>Comparison of analytical results</h2>
        <div>
            {mse_analytical_results_div}
        </div>
    <h2>Comparison of parametrization</h2>
        <div>
            {mse_para_div}
        </div>
    <h2>Comparison of estimators</h2>
        <div>
            {mse_est_div}
        </div>
        """

    # Physicality Violation Test
    print("​​Generating physicality violation test blocks ...")
    physicality_violation_check_block = ""
    if display_items["physicality_violation"]:
        physicality_violation_test_div = generate_physicality_violation_test_div(
            estimation_results_list, case_name_list, true_object, num_data
        )
        physicality_violation_check_block = f"""<h1>Physicality violation test</h1>
    <div>
        {physicality_violation_test_div}
    </div>"""

    report_html = f"""<html>
<head>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style type="text/css">
        <!--
            {_css}
            {_inline_block_css}
            {_table_css}
            {_table_contents_css}
         -->
    </style>
    <style>
    @page {{
        size: a4 portrait;
        @frame content_frame {{
            left: 20pt; right: 20pt; top: 20pt; height: 702pt;
        }}
        @frame footer_frame {{
            -pdf-frame-content: footer_content;
            left: 20pt; right: 20pt; top: 812pt; height: 20pt;
        }}
    }}
    </style>
<title>Quara Report</title>
</head>
<body>
<div id="table_of_contents">
    <h1>Table of contents</h1>
    <pdf:toc />
</div>
<h1>Computation time in total</h1>
    <div>
        {computation_time_table}
    </div>
{tolerance_table_div}
<h1>Experimental condition</h1>
    <div>
        {condition_table}
    </div>
<h2>True object</h2>
    <div>
        {true_object_table}
    </div>
<h2>Tester objects</h2>
    <div>
        {tester_table}
    </div>
<h2>Cases</h2>
    <div>
        {case_table}
    </div>
<h1>Computation time of estimators</h1>
    <div>{comp_time_of_est_div}</div>
{empi_dists_mse_block}
{consistency_check_block}
{mse_of_est_block}
{physicality_violation_check_block}
<div id="footer_content">
    <pdf:pagenumber>
</div>
</body>
</html>"""

    with open(Path(_temp_dir_path) / "quara_report.html", "w") as f:
        f.write(report_html)

    print("Converting to PDF report ...")
    _convert_html2pdf(report_html, path)
    if keep_tmp_files:
        import datetime as dt

        identity = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        material_path = f"quara_report_{identity}"
        Path(material_path).mkdir(parents=True, exist_ok=True)
        shutil.copytree(_temp_dir_path, material_path, dirs_exist_ok=True)
        print("Completed to copy temporary files. dst: {material_path}")

    print("​Deleting temporary files ...")
    shutil.rmtree(_temp_dir_path)
    _temp_dir_path = ""
    print(f"Completed to export pdf. ({path})")
