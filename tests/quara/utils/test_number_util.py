from pathlib import Path

import pytest
import numpy as np

from quara.utils import number_util


def test_check_positive_number():
    parameter_name = "dim"
    # valid
    target = 1
    number_util.check_positive_number(target, parameter_name)

    # invalid
    target = 0

    with pytest.raises(ValueError):
        number_util.check_positive_number(target, parameter_name)


def test_check_nonnegative_number():
    parameter_name = "dim"
    # valid
    target = 0
    number_util.check_nonnegative_number(target, parameter_name)

    # invalid
    target = -0.1

    with pytest.raises(ValueError):
        number_util.check_nonnegative_number(target, parameter_name)


def test_to_stream():
    # seed_or_stream :default
    np.random.seed(7)
    actual = number_util.to_stream()
    assert actual.randint(10) == 4
    assert actual.randint(10) == 9

    # seed_or_stream = None
    np.random.seed(7)
    actual = number_util.to_stream(None)
    assert actual.randint(10) == 4
    assert actual.randint(10) == 9

    # seed_or_stream = int:7
    actual = number_util.to_stream(7)
    assert actual.randint(10) == 4
    assert actual.randint(10) == 9

    # seed_or_stream = RandomState(7)
    actual = number_util.to_stream(np.random.RandomState(7))
    assert actual.randint(10) == 4
    assert actual.randint(10) == 9
