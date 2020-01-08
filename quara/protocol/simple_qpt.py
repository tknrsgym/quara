import logging

import matlab.engine
import numpy as np

from quara.engine.matlabengine import MatlabEngine

logger = logging.getLogger(__name__)


def load_state_list(path: str, dim: int, num_state: int) -> np.ndarray:
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
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.complex128)

    # check data
    if raw_data.shape[1] != dim:
        raise ValueError(
            f"Invalid number of columns in povm list '{path}'. expected={dim}(dim), actual={raw_data.shape[1]}"
        )
    if raw_data.shape[0] != num_povm * num_outcome * dim:
        raise ValueError(
            f"Invalid number of rows in povm list '{path}'. expected={num_povm * num_outcome * dim}(num_povm * num_outcome * dim), actual={raw_data.shape[0]}"
        )

    povm_list = np.reshape(raw_data, (num_povm, num_outcome, dim * dim))
    return povm_list


def load_schedule(path: str, num_state: int, num_povm: int) -> (int, np.ndarray):
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.uint16)
    num_schedule = raw_data.shape[0]

    # check data
    if raw_data.shape[1] != 2:
        raise ValueError(
            f"Invalid number of columns in schedule list '{path}'. expected=2, actual={raw_data.shape[1]}"
        )
    if np.max(raw_data[:, 0]) > num_state:
        raise ValueError(
            f"Invalid state in schedule list '{path}'. expected<={num_state}, actual={np.max(raw_data[:, 0])}"
        )
    if np.min(raw_data[:, 0]) >= 0:
        raise ValueError(
            f"Invalid state in schedule list '{path}'. expected>=0, actual={np.min(raw_data[:, 0])}"
        )
    if np.max(raw_data[:, 1]) > num_povm:
        raise ValueError(
            f"Invalid povm in schedule list '{path}'. expected<={num_povm}, actual={np.max(raw_data[:, 1])}"
        )
    if np.min(raw_data[:, 0]) >= 0:
        raise ValueError(
            f"Invalid povm in schedule list '{path}'. expected>=0, actual={np.min(raw_data[:, 1])}"
        )

    return num_schedule, raw_data


def load_empi_list(path: str, num_schedule: int, num_outcome: int) -> np.ndarray:
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

    return raw_data


def load_weight_list(path: str, num_schedule: int, num_outcome: int) -> np.ndarray:
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


def execute(settings: dict) -> np.ndarray:
    logger.debug("--- load schedule ---")
    num_schedule, schedule_np = load_schedule(
        settings["path_schedule"], settings["num_state"], settings["num_povm"]
    )
    logger.debug(f"num_schedule={num_schedule}")
    schedule_ml = matlab.uint64(schedule_np.tolist())
    logger.debug(schedule_ml)

    logger.debug("--- load empi list ---")
    empi_list_np = load_empi_list(settings["path_empi"], num_schedule)
    empi_list_ml = matlab.double(empi_list_np.tolist())
    logger.debug(empi_list_ml)

    logger.debug("--- load state list ---")
    state_list_np = load_state_list(
        settings["path_state"], settings["dim"], settings["num_state"]
    )
    state_list_ml = matlab.double(state_list_np.tolist(), is_complex=True)
    logger.debug(state_list_ml)

    logger.debug("--- load povm list ---")
    povm_list_np = load_povm_list(
        settings["path_povm"],
        settings["dim"],
        settings["num_povm"],
        settings["num_outcome"],
    )
    povm_list_ml = matlab.double(povm_list_np.tolist(), is_complex=True)
    logger.debug(povm_list_ml)

    logger.debug("--- load weight list ---")
    weight_list_np = load_weight_list(
        settings["path_weight"], num_schedule, settings["num_outcome"]
    )
    weight_list_ml = matlab.double(weight_list_np.tolist())
    logger.debug(weight_list_ml)

    with MatlabEngine() as engine:
        engine.check_pass_from_python_to_matlab(
            state_list_ml, nargout=0,
        )
