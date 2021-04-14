from quara.simulation.standard_qtomography_simulation import Result
from quara.simulation.standard_qtomography_simulation_flow import _print_summary


def test_print_summary(capfd):
    # Arrange
    check_result = {
        "name": "dummy case 0",
        "total_result": True,
        "results": [
            {"name": "MSE of Empirical Distributions", "result": True, "detail": None},
            {
                "name": "Consistency",
                "result": True,
                "detail": {"possibly_ok": True, "to_be_checked": False},
            },
            {"name": "MSE of estimators", "result": True, "detail": None},
            {"name": "Physicality Violation", "result": True, "detail": None},
        ],
    }
    result_0 = Result(
        result_index=None,
        simulation_setting=None,
        estimation_results=None,
        check_result=check_result,
    )

    check_result = {
        "name": "dummy case 1",
        "total_result": False,
        "results": [
            {"name": "MSE of Empirical Distributions", "result": True, "detail": None},
            {
                "name": "Consistency",
                "result": False,
                "detail": {"possibly_ok": False, "to_be_checked": False},
            },
            {"name": "MSE of estimators", "result": True, "detail": None},
            {"name": "Physicality Violation", "result": False, "detail": None},
        ],
    }
    result_1 = Result(
        result_index=None,
        simulation_setting=None,
        estimation_results=None,
        check_result=check_result,
    )

    check_result = {
        "name": "dummy case 2",
        "total_result": False,
        "results": [
            {"name": "MSE of Empirical Distributions", "result": True, "detail": None},
            {
                "name": "Consistency",
                "result": False,
                "detail": {"possibly_ok": False, "to_be_checked": True},
            },
            {"name": "MSE of estimators", "result": False, "detail": None},
            {"name": "Physicality Violation", "result": False, "detail": None},
        ],
    }
    result_2 = Result(
        result_index=None,
        simulation_setting=None,
        estimation_results=None,
        check_result=check_result,
    )

    check_result = {
        "name": "dummy case 3",
        "total_result": False,
        "results": [
            {"name": "MSE of Empirical Distributions", "result": False, "detail": None},
            {
                "name": "Consistency",
                "result": True,
                "detail": {"possibly_ok": True, "to_be_checked": False},
            },
            {"name": "MSE of estimators", "result": False, "detail": None},
            {"name": "Physicality Violation", "result": False, "detail": None},
        ],
    }
    result_3 = Result(
        result_index=None,
        simulation_setting=None,
        estimation_results=None,
        check_result=check_result,
    )
    dummy_results = [result_0, result_1, result_2, result_3]

    # Act
    _print_summary(results=dummy_results, elapsed_time=500)
    out, _ = capfd.readouterr()

    # Assert
    start_red = "\033[31m"
    start_green = "\033[32m"
    start_yellow = "\033[33m"
    end_color = "\033[0m"
    expected = f"""{start_yellow}=============== Summary ================={end_color}
MSE of Empirical Distributions:
{start_green}OK: 3 cases{end_color}, {start_red}NG: 1 cases{end_color}

Consistency:
{start_green}OK: 2 cases{end_color}, {start_red}NG: 2 cases{end_color}
You need to check report: 1 cases

MSE of estimators:
{start_green}OK: 2 cases{end_color}, {start_red}NG: 2 cases{end_color}

Physicality Violation:
{start_green}OK: 1 cases{end_color}, {start_red}NG: 3 cases{end_color}

Time:
500.0s (0:08:20)
{start_yellow}========================================={end_color}\n"""

    assert out == expected
