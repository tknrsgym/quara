from pathlib import Path

import pytest

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
