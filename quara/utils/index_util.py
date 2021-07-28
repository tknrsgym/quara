from typing import List, Tuple, Union

import numpy as np


def index_multi_dimensional_from_index_serial(
    nums_length: List[int], index_serial: int
) -> Tuple[int]:
    """calculates multi-dimensional index from `nums_length` and `index_serial`.

    Parameters
    ----------
    nums_length : List[int]
        sizes of the possible values for index tuple.
    index_serial : int
        serial index.

    Returns
    -------
    Tuple[int]
        multi-dimensional index.
        0 <= index_multi_dimensional[i] < nums_length[i] for all i.
    """
    tmp_index_serial = index_serial
    index_multi_dimensional = []
    for local_length in reversed(nums_length):
        local_index = tmp_index_serial % local_length
        index_multi_dimensional.append(local_index)
        tmp_index_serial = tmp_index_serial // local_length

    return tuple(reversed(index_multi_dimensional))


def index_serial_from_index_multi_dimensional(
    nums_length: List[int], index_multi_dimensional: Tuple[int]
) -> int:
    """calculates serial index from `nums_length` and `index_multi_dimensional`.

    Parameters
    ----------
    nums_length : List[int]
        sizes of the possible values for index_multi_dimensional.
    index_multi_dimensional : Tuple[int]
        tuple of indices.
        0 <= index_multi_dimensional[i] < nums_length[i] for all i.

    Returns
    -------
    int
        serial index.

    Raises
    ------
    ValueError
        whether the length of nums_length does not equal the length of index_multi_dimensional.
    """
    # whether the length of nums_length equals the length of index_multi_dimensional
    if len(nums_length) != len(index_multi_dimensional):
        raise ValueError(
            f"the length of nums_length must equal the length of index_multi_dimensional. length of nums_length={len(nums_length)}, length of index_multi_dimensional={len(index_multi_dimensional)}"
        )

    # calculate serial index by traversing index_multi_dimensional from the back.
    # for example, if the length of nums_length is 3 and each values are len0, len1, len2,
    # then the serial index of index_multi_dimensional(x0, x1, x2) can be calculated the following expression:
    #   x0 * (len1 * len2) + x1 * len2 + x2
    serial_index = 0
    temp_len = 1
    for length, local_index in reversed(
        list(zip(nums_length, index_multi_dimensional))
    ):
        serial_index += local_index * temp_len
        temp_len = temp_len * length
    return serial_index
