from typing import List
from pathlib import Path

from xhtml2pdf import pisa
from xhtml2pdf.config.httpconfig import httpConfig

from quara.data_analysis import physicality_violation_check

_TEMPLATE_BASE = """<html>
    <body>
        {}
        <hr>
        {}
    </body>
</html>
"""
_TEMPLATE_SECTION_PHYSICALITY_VIOLATION_CHECK = """<div>
    <h2>Physicality Violation Check</h2>
    {}
</div>"""

_TAMPLATE_SUBSECTION_PYHYICALITY_VIOLATION_CHECK = """<div>
    <h3>Distribution of Eigenvalues</h3>
    <img src="images/figure_1.png">
    <h3>Sum of Unphysival Eigenvalues</h3>
    <img src="images/figure_2.png">
</div>
"""
_TEMPLATE_SECTION_MSE = """<div>
    <h2>Mean Square Error</h2>
    <img src="images/figure_3.png">
</div>"""


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
