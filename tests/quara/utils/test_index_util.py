import pytest

from quara.utils import index_util


def test_index_multi_dimensional_from_index_serial():
    nums_length = [2, 3, 5]

    # case 1
    actual = index_util.index_multi_dimensional_from_index_serial(nums_length, 0)
    assert actual == (0, 0, 0)

    # case 2
    actual = index_util.index_multi_dimensional_from_index_serial(nums_length, 21)
    assert actual == (1, 1, 1)

    # case 3
    actual = index_util.index_multi_dimensional_from_index_serial(nums_length, 28)
    assert actual == (1, 2, 3)


def test_index_serial_from_index_multi_dimensional():
    nums_length = [2, 3, 5]

    # case 1
    index_multi_dimensional = (0, 0, 0)
    actual = index_util.index_serial_from_index_multi_dimensional(
        nums_length, index_multi_dimensional
    )
    assert actual == 0

    # case 2
    index_multi_dimensional = (1, 1, 1)
    actual = index_util.index_serial_from_index_multi_dimensional(
        nums_length, index_multi_dimensional
    )
    assert actual == 21

    # case 3
    index_multi_dimensional = (1, 2, 3)
    actual = index_util.index_serial_from_index_multi_dimensional(
        nums_length, index_multi_dimensional
    )
    assert actual == 28

    # case 4: index_multi_dimensional is too short
    index_multi_dimensional = (1, 2)
    with pytest.raises(ValueError):
        index_util.index_serial_from_index_multi_dimensional(
            nums_length, index_multi_dimensional
        )

    # case 5: index_multi_dimensional is too long
    index_multi_dimensional = (1, 2, 3, 4)
    with pytest.raises(ValueError):
        index_util.index_serial_from_index_multi_dimensional(
            nums_length, index_multi_dimensional
        )
