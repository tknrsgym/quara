from xhtml2pdf import pisa

from quara.data_analysis.physicality_violation_check import check_physicality_violation

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


def _convert_html2pdf(source_html, output_path):
    # TODO: make parent directory
    # TODO: check file extension
    with open(output_path, "w+b") as f:
        pisa_status = pisa.CreatePDF(source_html, dest=f)
    return pisa_status.err
