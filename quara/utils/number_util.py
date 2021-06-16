from pathlib import Path
from typing import Union

import numpy as np


def check_positive_number(target: float, parameter_name: str) -> None:
    """check if ``target`` is positive number.

    Parameters
    ----------
    target : int
        the number to check.
    parameter_name : str
        the parameter name of the number.
        this is used for error message.

    Raises
    ------
    ValueError
        ``target`` is not positive number.
    """
    # check data
    if target <= 0:
        raise ValueError(
            f"Invalid value range '{parameter_name}'. expected>0, actual={target}"
        )


def check_nonnegative_number(target: float, parameter_name: str) -> None:
    """check if ``target`` is nonnegative number.

    Parameters
    ----------
    target : int
        the number to check.
    parameter_name : str
        the parameter name of the number.
        this is used for error message.

    Raises
    ------
    ValueError
        ``target`` is not nonnegative number.
    """
    # check data
    if target < 0:
        raise ValueError(
            f"Invalid value range '{parameter_name}'. expected>=0, actual={target}"
        )


def to_stream(
    seed_or_stream: Union[int, np.random.RandomState] = None
) -> np.random.RandomState:
    """returns RandomState to generate random numbers.

    Parameters
    ----------
    seed_or_stream : Union[int, np.random.RandomState], optional
        If the type is int, generates RandomState with seed `seed_or_stream` and returned generated RandomState.
        If the type is RandomState, returns RandomState.
        If argument is None, returns np.random.
        Default value is None.

    Returns
    -------
    np.random.RandomState
        RandomState to generate random numbers.
    """
    if seed_or_stream is None:
        stream = np.random
    elif type(seed_or_stream) == int:
        stream = np.random.RandomState(seed_or_stream)
    else:
        stream = seed_or_stream
    return stream
