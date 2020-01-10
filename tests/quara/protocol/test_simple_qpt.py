import pytest

import quara.protocol.simple_qpt as s_qpt


def test_check_file_extension():
    # valid
    valid_path = "hoge.csv"
    s_qpt.check_file_extension(valid_path)

    # invalid
    invalid_path = "hoge.tsv"

    with pytest.raises(ValueError):
        s_qpt.check_file_extension(invalid_path)

