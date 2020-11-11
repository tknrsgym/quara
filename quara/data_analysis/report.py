import tempfile
import shutil

from typing import List, Tuple, Optional
from pathlib import Path

import pandas as pd
from xhtml2pdf import pisa
from xhtml2pdf.config.httpconfig import httpConfig

from quara.data_analysis import (
    physicality_violation_check,
    data_analysis,
    consistency_check,
)
from quara.protocol.qtomography.qtomography_estimator import QTomographyEstimator
from quara.protocol.qtomography.estimator import EstimationResult

from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate


_temp_dir_path = ""

_css = f"""
body {{color: #666666;}}
h1 {{margin-top: 60px;
    border-top: 2px #dcdcdc solid;
    padding-top: 10px;
    font-size: 25px}}
h2 {{font-size: 20px}}
h3 {{font-size: 15px;
    color: #618CBC;}}
h4 {{color:#EB9348; font-size: 15px;}}
h5 {{color:#666666; font-size: 13px;}}
h6 {{color:#666666; font-size: 13px; font-style:italic;}}
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
  width: 350px;
  word-break: break-all;
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
pdftoc.pdftoclevel4 {
    margin-left: 4em;
    font-style: italic;
}
pdftoc.pdftoclevel5 {
    margin-left: 5em;
    font-style: italic;
}
"""

_inline_block_css = """
.box{
 display: inline-block;
 width: 400px;
}

.box_col4{
 display: inline-block;
 width: 190px;
 padding: 0;
}
"""


def _convert_html2pdf(source_html: str, output_path: str):
    # TODO: make parent directory
    # TODO: check file extension
    httpConfig.save_keys("nosslcheck", True)
    with open(output_path, "w+b") as f:
        pisa_status = pisa.CreatePDF(source_html, dest=f)
    return pisa_status.err


def _make_graph_trace_seq(
    estimation_results: List["EstimationResult"], case_id: int
) -> list:
    num_data = estimation_results[0].num_data

    fig_info_list = []
    for i, num in enumerate(num_data):
        fig = physicality_violation_check.make_graph_trace(
            estimation_results, num_data_index=i, num_data=num_data
        )

        fig_name = f"case={case_id}_trace_num={num}_0"

        # output
        dir_path = Path(_temp_dir_path)
        path = str(dir_path / f"{fig_name}.png")
        fig.update_layout(width=500, height=400)
        dir_path.mkdir(exist_ok=True)
        fig.write_image(path)

        fig_info_list.append(dict(image_path=path, fig=fig, fig_name=fig_name))
    return fig_info_list


def _generate_trace_div(fig_info_list: List[dict]) -> str:
    graph_block_html = ""
    for fig_info in fig_info_list:
        graph_subblock = f"<div class='box'><img src={fig_info['image_path']}></div>"
        graph_block_html += graph_subblock

    graph_block_html = f"<div>{graph_block_html}</div>"

    return graph_block_html


def generate_trace_div(estimation_results: List["EstimationResult"], case_id: int):
    fig_info_list = _make_graph_trace_seq(estimation_results, case_id=case_id)
    div_html = _generate_trace_div(fig_info_list)
    return div_html


def _make_graph_sum_vecs_seq(
    estimation_results: List["EstimationResult"], case_id: int, true_object: Povm
) -> List[List["Figure"]]:
    fig_info_list_list = []
    num_data = estimation_results[0].num_data

    for num_data_index, num in enumerate(num_data):
        figs = physicality_violation_check.make_graphs_sum_vecs(
            estimation_results, true_object, num_data_index=num_data_index
        )
        fig_info_list = []
        for alpha, fig in enumerate(figs):
            fig_name = f"case={case_id}_trace_num={num}_alpha={alpha}"

            # output
            dir_path = Path(_temp_dir_path)
            path = str(dir_path / f"{fig_name}.png")
            fig.update_layout(width=500, height=400)
            dir_path.mkdir(exist_ok=True)
            fig.write_image(path)
            fig_info_list.append(
                dict(image_path=path, fig=fig, fig_name=fig_name, num=num, alpha=alpha)
            )
        fig_info_list_list.append(fig_info_list)

    return fig_info_list_list


def _generate_sum_vecs_div(fig_info_list_list: List[List[dict]]) -> str:
    graph_block_html_all = ""

    for fig_info_list in fig_info_list_list:  # num
        num = fig_info_list[0]["num"]
        graph_block_html = f"<h5>N={num}</h5>"
        for fig_info in fig_info_list:  # alpha
            graph_subblock = (
                f"<div class='box_col4'><img src={fig_info['image_path']}></div>"
            )
            graph_block_html += graph_subblock

        graph_block_html_all += f"<div>{graph_block_html}</div>"
    graph_block_html_all = f"<div>{graph_block_html_all}</div>"

    return graph_block_html_all


def generate_sum_vecs_div(
    estimation_results: List["EstimationResult"], case_id: int, true_object: Povm,
):
    fig_info_list_list = _make_graph_sum_vecs_seq(
        estimation_results, case_id=case_id, true_object=true_object
    )
    div_html = _generate_sum_vecs_div(fig_info_list_list)
    return div_html


def _generate_graph_eigenvalues_seq(
    estimation_results: List["EstimationResult"],
    case_id: int,
    true_object: "QOperation",
) -> list:
    num_data = estimation_results[0].num_data
    fig_info_list_list = []
    for num_data_index in range(len(num_data)):
        fig_list = physicality_violation_check.make_graphs_eigenvalues(
            estimation_results, true_object, num_data, num_data_index=num_data_index,
        )
        fig_info_list = []

        for i, fig in enumerate(fig_list):
            fig_name = f"case={case_id}_eigenvalues_num={num_data_index}_i={i}"

            # output
            dir_path = Path(_temp_dir_path)
            path = str(dir_path / f"{fig_name}.png")
            fig.update_layout(width=500, height=400)
            dir_path.mkdir(exist_ok=True)
            fig.write_image(path)

            fig_info_list.append(dict(image_path=path, fig=fig, fig_name=fig_name))

        fig_info_list_list.append(fig_info_list)
    return fig_info_list_list


def _generate_eigenvalues_div(fig_info_list_list: List[List[dict]]) -> str:
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


def _generate_eigenvalues_div_3loop(fig_info_list3: List[List[List[dict]]]) -> str:
    graph_block_html_all = ""
    for fig_info_list2 in fig_info_list3:  # num_data
        num = fig_info_list2[0][0]["num"]
        graph_block_html = f"<h5>N={num}</h5>"

        for fig_info_list in fig_info_list2:  # measurement
            x_i = fig_info_list[0]["x"]
            sub_graph_block_html = f"<h6>x={x_i}</h6>"
            for fig_info in fig_info_list:
                graph_subblock = (
                    f"<div class='box'><img src={fig_info['image_path']}></div>"
                )
                sub_graph_block_html += graph_subblock
            graph_block_html += f"<div>{sub_graph_block_html}</div>"

        graph_block_html_all += f"<div>{graph_block_html}</div>"
    graph_block_html_all = f"<div>{graph_block_html_all}</div>"

    return graph_block_html_all


def _generate_graph_eigenvalues_seq_3loop(
    estimation_results: List["EstimationResult"],
    case_id: int,
    true_object: "QOperation",
) -> list:
    num_data = estimation_results[0].num_data
    # For State
    fig_info_list3 = []
    for num_data_index in range(len(num_data)):
        fig_list_list = physicality_violation_check.make_graphs_eigenvalues(
            estimation_results, true_object, num_data, num_data_index=num_data_index,
        )
        fig_info_list2 = []

        for x_i, fig_list in enumerate(fig_list_list):
            fig_info_list = []
            for i, fig in enumerate(fig_list):
                fig_name = (
                    f"case={case_id}_eigenvalues_num={num_data_index}_x={x_i}_i={i}"
                )

                # output
                dir_path = Path(_temp_dir_path)
                path = str(dir_path / f"{fig_name}.png")
                fig.update_layout(width=500, height=400)
                dir_path.mkdir(exist_ok=True)
                fig.write_image(path)

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
    case_id: int,
    true_object: "QOperation",
):
    if type(true_object) == State:
        fig_info_list_list = _generate_graph_eigenvalues_seq(
            estimation_results, case_id=case_id, true_object=true_object,
        )
        div_html = _generate_eigenvalues_div(fig_info_list_list)
    elif type(true_object) == Povm:
        fig_info_list3 = _generate_graph_eigenvalues_seq_3loop(
            estimation_results, case_id=case_id, true_object=true_object,
        )
        div_html = _generate_eigenvalues_div_3loop(fig_info_list3)
    else:
        raise NotImplementedError()
    return div_html


def _generate_graph_sum_eigenvalues_seq(
    estimation_results: List["EstimationResult"], case_id: int, true_object,
) -> List[List[dict]]:
    num_data = estimation_results[0].num_data
    fig_info_list_list = []
    for num_data_index in range(len(num_data)):
        fig_list = physicality_violation_check.make_graphs_sum_unphysical_eigenvalues(
            estimation_results, num_data=num_data, num_data_index=num_data_index,
        )
        fig_info_list = []

        for i, fig in enumerate(fig_list):
            fig_name = f"case={case_id}_sum-unphysical-eigenvalues_num={num_data_index}_type={i}"

            # output
            dir_path = Path(_temp_dir_path)
            path = str(dir_path / f"{fig_name}.png")
            fig.update_layout(width=500, height=400)
            dir_path.mkdir(exist_ok=True)
            fig.write_image(path)

            fig_info_list.append(
                dict(
                    image_path=path,
                    fig=fig,
                    fig_name=fig_name,
                    num=num_data[num_data_index],
                )
            )

        fig_info_list_list.append(fig_info_list)
    return fig_info_list_list


def _generate_sum_eigenvalues_div(fig_info_list_list: List[List[dict]]) -> str:
    graph_block_html_all = ""
    for fig_info_list in fig_info_list_list:
        num = fig_info_list[0]["num"]
        graph_block_html = f"<h5>N={num}</h5>"

        for fig_info in fig_info_list:
            graph_subblock = (
                f"<div class='box'><img src={fig_info['image_path']}></div>"
            )
            graph_block_html += graph_subblock

        graph_block_html_all += f"<div>{graph_block_html}</div>"
    graph_block_html_all = f"<div>{graph_block_html_all}</div>"

    return graph_block_html_all


def generate_sum_eigenvalues_div(
    estimation_results: List["EstimationResult"], case_id: int, true_object,
):
    fig_info_list_list = _generate_graph_sum_eigenvalues_seq(
        estimation_results, case_id=case_id, true_object=true_object
    )
    div_html = _generate_sum_eigenvalues_div(fig_info_list_list)
    return div_html


def generate_mse_div(
    estimation_results_list: List[List[EstimationResult]],
    case_name_list: List[str],
    true_object: "QOperation",
    num_data: List[int],
    n_rep: int = None,
    show_analytical_results: bool = True,
    tester_objects: List["QOperation"] = None,
) -> str:

    title = f"Mean squared error"
    if not n_rep:
        title += "<br>Nrep={n_rep}"

    display_name_list = [f"Case {i}: {name}" for i, name in enumerate(case_name_list)]
    fig = data_analysis.make_mses_graph_estimation_results(
        estimation_results_list=estimation_results_list,
        case_names=display_name_list,
        num_data=num_data,
        true_object=true_object,
        show_analytical_results=show_analytical_results,
        tester_objects=tester_objects,
    )

    fig_name = f"mse"

    dir_path = Path(_temp_dir_path)
    path = str(dir_path / f"{fig_name}.png")
    dir_path.mkdir(exist_ok=True)
    fig.write_image(path)

    mse_div = f"""<div><img src="{path}"></div>
    """
    return mse_div


def generate_empi_dist_mse_div(
    estimation_results_list: List[List[EstimationResult]], true_object: "QOperation",
) -> str:

    title = f"Mean squared error"
    n_rep = len(estimation_results_list[0])
    title += "<br>Nrep={n_rep}"

    fig = data_analysis.make_empi_dists_mse_graph(
        estimation_results_list[0], true_object
    )

    fig_name = f"empi_dists_mse"

    dir_path = Path(_temp_dir_path)
    path = str(dir_path / f"{fig_name}.png")
    dir_path.mkdir(exist_ok=True)
    fig.write_image(path)

    div = f"""<div><img src="{path}"></div>
    """
    return div


def _parse_qoperation_desc(qoperation: "QOperation") -> List[str]:
    desc = str(qoperation)
    continue_flag, before_t = False, ""
    value_list = []
    pre_value_list = desc.split("\t")[1:]

    for i, t in enumerate(pre_value_list):
        if continue_flag:
            target = before_t + t
        else:
            target = t[: t.find("\n")] if "\n" in t else t

        if t.endswith("\n") and i < len(pre_value_list) - 1:
            # 続く場合
            continue_flag = True
            before_t = t
        else:
            # この行が最後の場合
            continue_flag = False
            before_t = ""
            value_list.append(target)
    value_list = [v.replace("\n", "<br>") for v in value_list]
    return value_list


def _convert_object_to_datafrane(qoperation: "QOperation") -> pd.DataFrame:
    desc = str(qoperation)

    # parse description of QOperation
    item_names = [t[t.rfind("\n") + 1 :] for t in desc.split("\t")][:-1]
    item_names = [item.replace(":", "") for item in item_names if item]
    value_list = _parse_qoperation_desc(qoperation)

    df = pd.DataFrame(value_list, item_names).rename(columns={0: "value"})

    return df


def _convert_objects_to_multiindex_dataframe(
    qoperations: List["QOperation"],
) -> pd.DataFrame:
    df_dict = {}

    for i, tester in enumerate(qoperations):
        df_dict[i] = _convert_object_to_datafrane(qoperations[i])

    objects_df_multiindex = pd.concat(df_dict, axis=0)
    return objects_df_multiindex


def _generate_physicality_violation_test_div_for_state(
    estimation_results_list: List[List["EstimationResult"]],
    case_name_list: List[str],
    true_object: State,
):
    test_eq_const_divs = ""
    test_ineq_const_eigenvalues_divs = ""
    test_ineq_const_sum_eigenvalues_divs = ""

    for case_id, case_name in enumerate(case_name_list):
        estimation_results = estimation_results_list[case_id]
        # Test of equality constraint violation
        div = generate_trace_div(estimation_results, case_id=case_id)
        test_eq_const_divs += f"""
            <h4>Case {case_id}: {case_name}<h4>
            {div}
            """
        # Test of inequality constraint violation
        div = generate_eigenvalues_div(
            estimation_results, case_id=case_id, true_object=true_object,
        )
        test_ineq_const_eigenvalues_divs += f"""
            <h4>Case {case_id}: {case_name}<h4>
            {div}
            """

        div = generate_sum_eigenvalues_div(
            estimation_results, case_id=case_id, true_object=true_object,
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
        {test_ineq_const_eigenvalues_divs}
    """

    return eq_all_div, ineq_all_div


def _generate_physicality_violation_test_div_for_povm(
    estimation_results_list: List[List["EstimationResult"]],
    case_name_list: List[str],
    true_object: State,
):
    test_eq_const_divs = ""
    test_ineq_const_eigenvalues_divs = ""
    test_ineq_const_sum_eigenvalues_divs = ""

    for case_id, case_name in enumerate(case_name_list):
        estimation_results = estimation_results_list[case_id]
        # Test of equality constraint violation
        div = generate_sum_vecs_div(
            estimation_results, case_id=case_id, true_object=true_object
        )
        test_eq_const_divs += f"""
            <h4>Case {case_id}: {case_name}<h4>
            {div}
            """
        # Test of inequality constraint violation
        div = generate_eigenvalues_div(
            estimation_results, case_id=case_id, true_object=true_object,
        )
        test_ineq_const_eigenvalues_divs += f"""
            <h4>Case {case_id}: {case_name}<h4>
            {div}
            """

        div = generate_sum_eigenvalues_div(
            estimation_results, case_id=case_id, true_object=true_object,
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


def generate_physicality_violation_test_div(
    estimation_results_list: List[List["EstimationResult"]],
    case_name_list: List[str],
    true_object: "QOperation",
):

    if type(true_object) == State:
        (
            true_all_div,
            false_all_div,
        ) = _generate_physicality_violation_test_div_for_state(
            estimation_results_list, case_name_list, true_object
        )
    elif type(true_object) == Povm:
        (
            true_all_div,
            false_all_div,
        ) = _generate_physicality_violation_test_div_for_povm(
            estimation_results_list, case_name_list, true_object
        )
    else:
        raise NotImplementedError()

    physicality_violation_test_div = f"""
        {true_all_div}
        {false_all_div}
    """

    return physicality_violation_test_div


def generate_case_table(
    case_name_list: List["str"],
    qtomography_list: List["QTomography"],
    para_list: List[int],
    estimator_list: List["Estimator"],
):
    case_dict = dict(
        Name=case_name_list,
        Parameterization=para_list,
        Tomography=[t.__class__.__name__ for t in qtomography_list],
        Estimator=[e.__class__.__name__ for e in estimator_list],
    )
    case_df = pd.DataFrame(case_dict)
    styles = [
        dict(selector=".col0", props=[("width", "400px")]),
        dict(selector=".col1", props=[("width", "200px")]),
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


def generate_consistency_check_table(
    qtomography_list: List["QTomography"],
    para_list: List[bool],
    estimator_list: List["Estimator"],
    true_object: "QOperation",
):
    result_list = []

    for i, qtomo in enumerate(qtomography_list):
        estimator = estimator_list[i]
        diff = consistency_check.calc_mse_of_true_estimated(
            true_object=true_object, qtomography=qtomo, estimator=estimator
        )
        result_list.append(diff)

    type_tomography_values = [qt.__class__.__name__ for qt in qtomography_list]
    type_estimator_values = [e.__class__.__name__ for e in estimator_list]

    result_dict = {
        "Type of tomography": type_tomography_values,
        "Parametorization": para_list,
        "Estimator": type_estimator_values,
        "Result": result_list,
    }

    consistency_check_table = pd.DataFrame(result_dict).to_html(
        classes="consistency_check_table", escape=False
    )
    return consistency_check_table


def export_report(
    path: str,
    estimation_results_list: List[List["EstimationResult"]],
    case_name_list: List[str],
    qtomography_list: List["QTomography"],
    para_list: List[bool],
    estimator_list: List["Estimator"],
    true_object: "QOperation",
    tester_objects: List["QOperation"],
    num_data: List[int],
    n_rep: int,
    save_materials: bool = False,
    seed: int = None,
):
    temp_dir_path = tempfile.mkdtemp()
    global _temp_dir_path
    _temp_dir_path = Path(temp_dir_path)

    # Experiment Condition
    condition_table = generate_condition_table(qtomography_list, n_rep, num_data, seed)

    # Cases
    case_table = generate_case_table(
        case_name_list, qtomography_list, para_list, estimator_list
    )

    # MSE
    mse_div = generate_mse_div(
        estimation_results_list=estimation_results_list,
        case_name_list=case_name_list,
        true_object=true_object,
        num_data=num_data,
        n_rep=n_rep,
        show_analytical_results=True,
        tester_objects=tester_objects,
    )

    # Physicality Violation Test
    physicality_violation_test_div = generate_physicality_violation_test_div(
        estimation_results_list, case_name_list, true_object
    )

    # True Object
    true_object_table = _convert_object_to_datafrane(true_object).to_html(
        classes="true_object_table", escape=False, header=False
    )
    # Tester Object
    tester_table = _convert_objects_to_multiindex_dataframe(tester_objects).to_html(
        classes="tester_objects_table", escape=False, header=False
    )

    # MSE of Empirical Distributions
    empi_dists_mse_div = generate_empi_dist_mse_div(
        estimation_results_list, true_object
    )

    # Consistency Test
    consistency_check_table = generate_consistency_check_table(
        qtomography_list, para_list, estimator_list, true_object,
    )

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
            left: 20pt; right: 20pt; top: 50pt; height: 672pt;
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
<h1>MSE of Empirical Distributions</h1>
    <div>{empi_dists_mse_div}</div>
<h1>Consistency test</h1>
    <div>{consistency_check_table}</div>
    <h1>MSE</h1>
        <div>
        {mse_div}
        </div>
<h1>Physicality violation test</h1>
    <div>
        {physicality_violation_test_div}
    </div>
<div id="footer_content">
    <pdf:pagenumber>
</div>
</body>
</html>"""

    with open(Path(_temp_dir_path) / "quara_report.html", "w") as f:
        f.write(report_html)

    _convert_html2pdf(report_html, path)
    if save_materials:
        import datetime as dt

        identity = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        material_path = f"quara_report_{identity}"
        Path(material_path).mkdir(parents=True, exist_ok=True)
        shutil.copytree(_temp_dir_path, material_path, dirs_exist_ok=True)

    shutil.rmtree(_temp_dir_path)
    _temp_dir_path = ""
    print(f"Completed to export pdf. ({path})")
