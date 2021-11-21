import pytest

from quara.simulation import standard_qtomography_simulation_report as sim_report


def test_setup_display_items():
    expected_items = [
        "consistency",
        "mse_of_estimators",
        "mse_of_empi_dists",
        "physicality_violation",
    ]

    # Act
    actual = sim_report.setup_display_items(None)
    # Assert
    expected = {
        "consistency": True,
        "mse_of_estimators": True,
        "mse_of_empi_dists": True,
        "physicality_violation": True,
    }
    assert actual == expected

    # Act
    source = {"consistency": False}
    actual = sim_report.setup_display_items(source)
    # Assert
    expected = {
        "consistency": False,
        "mse_of_estimators": True,
        "mse_of_empi_dists": True,
        "physicality_violation": True,
    }
    assert actual == expected

    # Act
    source = {
        "consistency": True,
        "mse_of_estimators": False,
        "mse_of_empi_dists": True,
        "physicality_violation": False,
    }
    actual = sim_report.setup_display_items(source)
    # Assert
    assert actual == source

    # Act
    source = {"invalid": True}
    with pytest.raises(KeyError):
        _ = sim_report.setup_display_items(source)
