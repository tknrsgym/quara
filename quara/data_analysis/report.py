from quara.protocol.qtomography.qtomography_estimator import QTomographyEstimator
from quara.protocol.qtomography.estimator import EstimationResult
from typing import List, Tuple, Optional
from pathlib import Path

from xhtml2pdf import pisa
from xhtml2pdf.config.httpconfig import httpConfig

from quara.data_analysis import physicality_violation_check, data_analysis


def _convert_html2pdf(source_html: str, output_path: str):
    # TODO: make parent directory
    # TODO: check file extension
    httpConfig.save_keys("nosslcheck", True)
    with open(output_path, "w+b") as f:
        pisa_status = pisa.CreatePDF(source_html, dest=f)
    return pisa_status.err


# 等式制約のグラフを作る
def _make_graph_trace_seq(
    estimation_results: List["EstimationResult"], case_id: int, num_data: List[int]
) -> list:
    fig_info_list = []
    for i, num in enumerate(num_data):
        fig = physicality_violation_check.make_graph_trace(
            estimation_results, num_data_index=i, num_data=num_data
        )

        fig_name = f"case={case_id}_trace_num={num}_0"

        # output
        # TODO
        dir_path = Path("/Users/tomoko/project/rcast/workspace/quara/tutorials/images")
        path = str(dir_path / f"{fig_name}.png")
        fig.update_layout(width=500, height=400)
        dir_path.mkdir(exist_ok=True)
        fig.write_image(path)

        fig_info_list.append(dict(image_path=path, fig=fig, fig_name=fig_name))
    return fig_info_list


def _generate_trace_div(fig_info_list) -> str:
    graph_block_html = ""
    for fig_info in fig_info_list:
        graph_subblock = f"<div class='box'><img src={fig_info['image_path']}></div>"
        graph_block_html += graph_subblock

    graph_block_html = f"<div>{graph_block_html}</div>"

    return graph_block_html


def generate_trace_div(
    estimation_results: List["EstimationResult"], case_id: int, num_data: List[int]
):
    fig_info_list = _make_graph_trace_seq(
        estimation_results, case_id=case_id, num_data=num_data
    )
    div_html = _generate_trace_div(fig_info_list)
    return div_html


def _generate_graph_eigenvalues_seq(
    estimation_results: List["EstimationResult"],
    case_id: int,
    true_object,
    num_data: List[int],
) -> list:

    fig_info_list_list = []
    for num_data_index in range(len(num_data)):
        fig_list = physicality_violation_check.make_graphs_eigenvalues(
            estimation_results, true_object, num_data, num_data_index=num_data_index,
        )
        fig_info_list = []

        for i, fig in enumerate(fig_list):
            fig_name = f"case={case_id}_eigenvalues_num={num_data_index}_i={i}"

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


def _generate_eigenvalues_div(fig_info_list_list) -> str:
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


def generate_eigenvalues_div(
    estimation_results: List["EstimationResult"],
    case_id: int,
    num_data: List[int],
    true_object,
):
    fig_info_list_list = _generate_graph_eigenvalues_seq(
        estimation_results, case_id=case_id, true_object=true_object, num_data=num_data
    )
    div_html = _generate_eigenvalues_div(fig_info_list_list)
    return div_html


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


def generate_mse_div(
    estimation_results_list: List[List[EstimationResult]],
    case_name_list: List[str],
    true_object,
    num_data: List[int],
    n_rep: int = None,
    qtomographies: List["StandardQTomography"] = None,
) -> str:
    mses_list = []
    display_name_list = []

    for result in estimation_results_list:
        mses, *_ = data_analysis.convert_to_series(result, true_object)
        mses_list.append(mses)
        display_name_list = [
            f"case {i}: {name}" for i, name in enumerate(case_name_list)
        ]

    if qtomographies:
        for qtomography in qtomographies:
            true_mses = []
            for num in num_data:
                true_mse = qtomography.calc_mse_linear(true_object, [num] * 3)
                true_mses.append(true_mse)
            mses_list.append(true_mses)
            display_name_list.append("analytical solution")

    title = f"Mean Square Value"
    if not n_rep:
        title += "<br>Nrep={n_rep}"

    fig = data_analysis.make_mses_graph(
        num_data=num_data, mses=mses_list, names=display_name_list, title=title
    )

    fig_name = f"mse"

    # TODO:
    dir_path = Path("/Users/tomoko/project/rcast/workspace/quara/tutorials/images")
    path = str(dir_path / f"{fig_name}.png")
    dir_path.mkdir(exist_ok=True)
    fig.write_image(path)

    mse_div = f"""<div><img src="{path}"></div>
    """
    return mse_div
