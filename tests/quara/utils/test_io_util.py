from pathlib import Path

import pytest

from quara.utils import io_util


def test_check_file_extension():
    # valid
    valid_path = "hoge.csv"
    io_util.check_file_extension(valid_path)

    # invalid
    invalid_path = "hoge.tsv"

    with pytest.raises(ValueError):
        io_util.check_file_extension(invalid_path)


def test_check_positive_number():
    parameter_name = "dim"
    # valid
    target = 1
    io_util.check_positive_number(target, parameter_name)

    # invalid
    target = 0

    with pytest.raises(ValueError):
        io_util.check_positive_number(target, parameter_name)


def test_check_nonnegative_number():
    parameter_name = "dim"
    # valid
    target = 0
    io_util.check_nonnegative_number(target, parameter_name)

    # invalid
    target = -0.1

    with pytest.raises(ValueError):
        io_util.check_nonnegative_number(target, parameter_name)
