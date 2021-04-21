import pickle
from pathlib import Path
import os

import numpy.testing as npt

from quara.simulation import standard_qtomography_simulation as sim
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate


def assert_equal_estimation_result(result_source, result_target):
    source_qoperations = result_source.estimated_qoperation_sequence
    target_qoperations = result_target.estimated_qoperation_sequence

    assert len(source_qoperations) == len(target_qoperations)
    for s, t in zip(source_qoperations, target_qoperations):
        assert_equal_qoperation(s, t)


def assert_equal_qoperation(source, target):
    assert type(source) == type(target)
    if type(source) == State:
        npt.assert_almost_equal(source.vec, target.vec, decimal=15)
    elif type(source) == Povm:
        assert len(source.vecs) == len(target.vecs)
        for vec_a, vec_b in zip(source.vecs, target.vecs):
            npt.assert_almost_equal(vec_a, vec_b, decimal=15)
    elif type(source) == Gate:
        npt.assert_almost_equal(source.hs, target.hs, decimal=15)
    else:
        raise NotImplementedError()


def test_re_estimate_sequence():
    # Arrange
    input_root_dir = Path(os.path.dirname(__file__)) / "data/source_re_simulation_qst"
    result_path = Path(input_root_dir) / "0" / "0" / "case_1_result.pickle"
    test_setting_path = Path(input_root_dir) / "0" / "test_setting.pickle"

    with open(result_path, "rb") as f:
        source_result = pickle.load(f)

    with open(test_setting_path, "rb") as f:
        source_test_setting = pickle.load(f)

    # Act
    actual_results = sim.re_estimate_sequence(source_test_setting, source_result)

    # Assert
    expected_results = source_result.estimation_results
    assert len(actual_results) == len(expected_results)
    for actual, expected in zip(actual_results, expected_results):
        assert_equal_estimation_result(actual, expected)


def test_re_estimate_sequence_from_path():
    input_root_dir = Path(os.path.dirname(__file__)) / "data/source_re_simulation_qst"
    result_path = Path(input_root_dir) / "0" / "0" / "case_2_result.pickle"
    test_setting_path = Path(input_root_dir) / "0" / "test_setting.pickle"

    # Act
    actual_results = sim.re_estimate_sequence_from_path(test_setting_path, result_path)

    # Assert
    with open(result_path, "rb") as f:
        source_result = pickle.load(f)
    expected_results = source_result.estimation_results
    assert len(actual_results) == len(expected_results)
    for actual, expected in zip(actual_results, expected_results):
        assert_equal_estimation_result(actual, expected)


def test_re_estimate_sequence_from_index():
    # Arrange
    input_root_dir = Path(os.path.dirname(__file__)) / "data/source_re_simulation_qst"

    # Act
    actual_results = sim.re_estimate_sequence_from_index(input_root_dir, 1, 0, 3)

    # Assert
    result_path = Path(input_root_dir) / "1" / "0" / "case_3_result.pickle"
    with open(result_path, "rb") as f:
        source_result = pickle.load(f)
    # Assert
    expected_results = source_result.estimation_results
    assert len(actual_results) == len(expected_results)
    for actual, expected in zip(actual_results, expected_results):
        assert_equal_estimation_result(actual, expected)
