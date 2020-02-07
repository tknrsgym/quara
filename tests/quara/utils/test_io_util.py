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
