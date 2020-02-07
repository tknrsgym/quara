import math
import numpy as np

from quara.utils.io_util import check_file_extension


def load_state_list(path: str, dim: int, num_state: int) -> np.ndarray:
    """Load state list from a csv file.
    The csv file must satisfy the followings:

    - the file extension is `csv`.
    - number of columns is equal to ``dim``.
    - number of rows is equal to ``dim * num_state``.

    Parameters
    ----------
    path : str
        the csv file path to load.
    dim : int
        dimension of Hilbert space.
    num_state : int
        number of state in the csv file.

    Returns
    -------
    np.ndarray
        state list represented by ndarray of dtype ``np.complex128``.
        its shape is ``(num_state, dim * dim)``.

    Raises
    ------
    ValueError
        the file extension is not `csv`.
    ValueError
        number of columns is not equal to ``dim``.
    ValueError
        number of rows is not equal to  ``dim * num_state``.
    """
    # check file extension
    check_file_extension(path)

    raw_data = np.loadtxt(path, delimiter=",", dtype=np.complex128)

    # check data
    if raw_data.shape[1] != dim:
        raise ValueError(
            f"Invalid number of columns in state list '{path}'. expected={dim}(dim), actual={raw_data.shape[1]}"
        )
    if raw_data.shape[0] != num_state * dim:
        raise ValueError(
            f"Invalid number of rows in state list '{path}'. expected={dim * num_state}(dim * num_state), actual={raw_data.shape[0]}"
        )

    state_list = np.reshape(raw_data, (num_state, dim * dim))
    return state_list


def load_povm_list(path: str, dim: int, num_povm: int, num_outcome: int) -> np.ndarray:
    """Load povm list from a csv file.
    The csv file must satisfy the followings:

    - the file extension is `csv`.
    - number of columns is equal to ``dim``.
    - number of rows is equal to ``dim * num_outcome * num_povm``.

    Parameters
    ----------
    path : str
        the csv file path to load.
    dim : int
        dimension of Hilbert space.
    num_povm : int
        number of povm in the csv file.
    num_outcome : int
        number of outcome in the csv file.

    Returns
    -------
    np.ndarray
        povm list represented by ndarray of dtype ``np.complex128``.
        its shape is ``(num_povm, num_outcome, dim * dim)``.

    Raises
    ------
    ValueError
        the file extension is not `csv`.
    ValueError
        number of columns is not equal to ``dim``.
    ValueError
        number of rows is not equal to  ``dim * num_outcome * num_povm``.
    """
    # check file extension
    check_file_extension(path)

    raw_data = np.loadtxt(path, delimiter=",", dtype=np.complex128)

    # check data
    if raw_data.shape[1] != dim:
        raise ValueError(
            f"Invalid number of columns in povm list '{path}'. expected={dim}(dim), actual={raw_data.shape[1]}"
        )
    if raw_data.shape[0] != dim * num_outcome * num_povm:
        raise ValueError(
            f"Invalid number of rows in povm list '{path}'. expected={dim * num_outcome * num_povm}(dim * num_outcome * num_povm), actual={raw_data.shape[0]}"
        )

    povm_list = np.reshape(raw_data, (num_povm, num_outcome, dim * dim))
    return povm_list


def load_schedule(path: str, num_state: int, num_povm: int) -> (int, np.ndarray):
    """Load schedule list from a csv file.
    The csv file must satisfy the followings:

    - the file extension is `csv`.
    - number of columns is equal to two.
    - each value of first column is less than or equal to ``num_state - 1``.
    - each value of first column is greater than or equal to ``0``.
    - each value of second column is less than or equal to ``num_povm - 1``.
    - each value of second column is greater than or equal to ``0``. 

    Parameters
    ----------
    path : str
        the csv file path to load.
    num_state : int
        number of state in the csv file.
    num_povm : int
        number of povm in the csv file.

    Returns
    -------
    int
        number of schedule
    np.ndarray
        schedule list represented by ndarray of dtype ``np.uint16``.
        its shape is ``(number of schedule, 2)``.

    Raises
    ------
    ValueError
        the file extension is not `csv`.
    ValueError
        number of columns is not equal to two.
    ValueError
        at least one value of first column is not less than or equal to ``num_state - 1``.
    ValueError
        at least one value of first column is not greater than or equal to ``0``.
    ValueError
        at least one value of second column is not less than or equal to ``num_povm - 1``.
    ValueError
        at least one value of second column is not greater than or equal to ``0``.
    """
    # check file extension
    check_file_extension(path)

    raw_data = np.loadtxt(path, delimiter=",", dtype=np.int16)
    num_schedule = raw_data.shape[0]

    # check data
    if raw_data.shape[1] != 2:
        raise ValueError(
            f"Invalid number of columns in schedule list '{path}'. expected=2, actual={raw_data.shape[1]}"
        )
    state_max = np.max(raw_data[:, 0])
    if state_max > num_state - 1:
        raise ValueError(
            f"Invalid state in schedule list '{path}'. expected<={num_state - 1}(num_state - 1), actual={state_max}"
        )
    state_min = np.min(raw_data[:, 0])
    if state_min < 0:
        raise ValueError(
            f"Invalid state in schedule list '{path}'. expected>=0, actual={state_min}"
        )
    povm_max = np.max(raw_data[:, 1])
    if povm_max > num_povm - 1:
        raise ValueError(
            f"Invalid povm in schedule list '{path}'. expected<={num_povm - 1}(num_povm - 1), actual={povm_max}"
        )
    povm_min = np.min(raw_data[:, 1])
    if povm_min < 0:
        raise ValueError(
            f"Invalid povm in schedule list '{path}'. expected>=0, actual={povm_min}"
        )

    return num_schedule, raw_data.astype(np.uint16)


def load_empi_list(path: str, num_schedule: int, num_outcome: int) -> np.ndarray:
    """Load empi list from a csv file.
    The csv file must satisfy the followings:

    - the file extension is `csv`.
    - number of columns is equal to ``num_outcome``.
    - number of rows is equal to ``num_schedule``.
    - each value is a non-negative real number.
    - sum of each row is equal to ``1``.

    Parameters
    ----------
    path : str
        the csv file path to load.
    num_schedule : int
        number of schedule in the csv file.
    num_outcome : int
        number of outcome in the csv file.

    Returns
    -------
    np.ndarray
        empi list represented by ndarray of dtype ``np.float64``.
        its shape is ``(num_schedule, num_outcome)``.

    Raises
    ------
    ValueError
        the file extension is not `csv`.
    ValueError
        number of columns is not equal to ``num_outcome``.
    ValueError
        number of rows is not equal to ``num_schedule``.
    ValueError
        at least one value is not a non-negative real number.
    ValueError
        at least one sum of each row is not equal to ``1``.
    """
    # check file extension
    check_file_extension(path)

    raw_data = np.loadtxt(path, delimiter=",", dtype=np.float64)

    # check data
    if raw_data.shape[1] != num_outcome:
        raise ValueError(
            f"Invalid number of columns in empi list '{path}'. expected={num_outcome}(num_outcome), actual={raw_data.shape[1]}"
        )
    if raw_data.shape[0] != num_schedule:
        raise ValueError(
            f"Invalid number of rows in empi list '{path}'. expected={num_schedule}(num_schedule), actual={raw_data.shape[0]}"
        )
    empi_min = np.min(raw_data)
    if empi_min < 0:
        raise ValueError(
            f"Invalid value in empi list '{path}'. expected>=0, actual={empi_min}"
        )

    empi_sum = np.sum(raw_data, axis=1)
    check_sum_is_one = np.vectorize(lambda x: True if math.isclose(x, 1.0) else False)
    check_result = np.where(check_sum_is_one(empi_sum) == False)
    if len(check_result[0]) > 0:
        invalid_sum = empi_sum[check_result[0]][0]
        invalid_row = raw_data[check_result[0]][0]
        raise ValueError(
            f"Invalid sum of rows in empi list '{path}'. expected=1.0, actual={invalid_sum} {invalid_row}"
        )

    return raw_data


def load_weight_list(path: str, num_schedule: int, num_outcome: int) -> np.ndarray:
    """Load weight list from a csv file.
    The csv file must satisfy the followings:

    - the file extension is `csv`.
    - number of columns is equal to ``num_outcome``.
    - number of rows is equal to ``num_schedule * num_outcome``.

    Parameters
    ----------
    path : str
        the csv file path to load.
    num_schedule : int
        number of schedule in the csv file.
    num_outcome : int
        number of outcome in the csv file.

    Returns
    -------
    np.ndarray
        weight list represented by ndarray of dtype ``np.float64``.
        its shape is ``(num_schedule, num_outcome, num_outcome)``.

    Raises
    ------
    ValueError
        the file extension is not `csv`.
    ValueError
        number of columns is not equal to ``num_outcome``.
    ValueError
        number of rows is not equal to  ``num_schedule * num_outcome``.
    """
    # check file extension
    check_file_extension(path)

    raw_data = np.loadtxt(path, delimiter=",", dtype=np.float64)

    # check data
    if raw_data.shape[1] != num_outcome:
        raise ValueError(
            f"Invalid number of columns in weight list '{path}'. expected={num_outcome}(num_outcome), actual={raw_data.shape[1]}"
        )
    if raw_data.shape[0] != num_schedule * num_outcome:
        raise ValueError(
            f"Invalid number of rows in weight list '{path}'. expected={num_schedule * num_outcome}(num_schedule * num_outcome), actual={raw_data.shape[0]}"
        )

    weight_list = np.reshape(raw_data, (num_schedule, num_outcome, num_outcome))
    return weight_list
