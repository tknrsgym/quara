from pathlib import Path

import pytest

from quara.utils import file_util


def test_check_file_extension():
    # valid
    valid_path = "hoge.csv"
    file_util.check_file_extension(valid_path)

    # invalid
    invalid_path = "hoge.tsv"

    with pytest.raises(ValueError):
        file_util.check_file_extension(invalid_path)
