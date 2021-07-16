from typing import Tuple, Union

import numpy as np


def index_tuple_from_index_serial(
    nums_local_outcomes: List[int], index_serial: int
) -> Tuple[int]:
    index_tuple = []
    for local_outcomes in enumerate(reversed(nums_local_outcomes)):
        local_index = index_serial % local_outcomes
        index_tuple.append(local_index)
        index_serial -= local_index
    return tuple(index_tuple)


def index_serial_from_index_tuple(
    nums_local_outcomes: List[int], index_tuple: Tuple[int]
) -> int:
    # whether size of tuple equals length of the list of measurements
    if len(index_tuple) != len(nums_local_outcomes):
        raise ValueError(
            f"length of tuple must equal length of the list of measurements. length of tuple={len(index)}, length of the list of measurements={len(self.nums_local_outcomes)}"
        )

    # calculate index in _vecs by traversing the tuple from the back.
    # for example, if length of _measurements is 3 and each numbers are len1, len2, len3,
    # then index in _basis of tuple(x1, x2, x3) can be calculated the following expression:
    #   x1 * (len2 * len3) + x2 * len3 + x3
    serial_index = 0
    temp_len = 1
    for position, local_index in enumerate(reversed(index_tuple)):
        serial_index += local_index * temp_len
        temp_len = temp_len * (nums_local_outcomes[position])
    return serial_index
