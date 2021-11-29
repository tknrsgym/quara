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
    seed_or_generator: Union[int, np.random.Generator] = None
) -> np.random.Generator:
    """returns np.random.Generator to generate random numbers.

    Parameters
    ----------
    seed_or_generator : Union[int, np.random.Generator], optional
        If the type is int, generates Generator with seed `seed_or_generator` and returned generated Generator.
        If the type is Generator, returns Generator.
        If argument is None, returns np.random.
        Default value is None.

    Returns
    -------
    np.random.Generator
        Generator for random numbers.
    """
    if seed_or_generator is None:
        stream = np.random
    elif type(seed_or_generator) == int:
        stream = np.random.Generator(np.random.MT19937(seed_or_generator))
    else:
        stream = seed_or_generator
    return stream
