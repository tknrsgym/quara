import pytest

from quara.utils import index_util


def test_index_tuple_from_index_serial():
    nums_length = [2, 3, 5]

    # case 1
    actual = index_util.index_tuple_from_index_serial(nums_length, 0)
    assert actual == (0, 0, 0)

    # case 2
    actual = index_util.index_tuple_from_index_serial(nums_length, 21)
    assert actual == (1, 1, 1)

    # case 3
    actual = index_util.index_tuple_from_index_serial(nums_length, 28)
    assert actual == (1, 2, 3)


def test_index_serial_from_index_tuple():
    nums_length = [2, 3, 5]

    # case 1
    index_tuple = (0, 0, 0)
    actual = index_util.index_serial_from_index_tuple(nums_length, index_tuple)
    assert actual == 0

    # case 2
    index_tuple = (1, 1, 1)
    actual = index_util.index_serial_from_index_tuple(nums_length, index_tuple)
    assert actual == 21

    # case 3
    index_tuple = (1, 2, 3)
    actual = index_util.index_serial_from_index_tuple(nums_length, index_tuple)
    assert actual == 28

    # case 4: index_tuple is too short
    index_tuple = (1, 2)
    with pytest.raises(ValueError):
        index_util.index_serial_from_index_tuple(nums_length, index_tuple)

    # case 5: index_tuple is too long
    index_tuple = (1, 2, 3, 4)
    with pytest.raises(ValueError):
        index_util.index_serial_from_index_tuple(nums_length, index_tuple)
