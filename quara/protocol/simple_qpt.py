import logging

import matlab.engine
import numpy as np

from quara.engine.matlabengine import MatlabEngine

logger = logging.getLogger(__name__)


def load_state_list(path: str, dim: int, num_state: int) -> np.ndarray:
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    state_list = np.reshape(raw_data, (num_state, dim * dim))
    return state_list


def load_povm_list(path: str, dim: int, num_povm: int, num_outcome: int) -> np.ndarray:
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    povm_list = np.reshape(raw_data, (num_povm, num_outcome, dim * dim))
    return povm_list


def load_schedule(path: str) -> (int, np.ndarray):
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.uint16)
    num_schedule, _ = raw_data.shape
    return num_schedule, raw_data


def load_empi_list(path: str, num_schedule: int = None) -> (int, np.ndarray):
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    _, num_outcome = raw_data.shape
    return num_outcome, raw_data


def load_weight_list(path: str, n_schedule: int, num_outcome: int) -> np.ndarray:
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    weight_list = np.reshape(raw_data, (n_schedule, num_outcome, num_outcome))
    return weight_list


def execute(settings: dict) -> np.ndarray:
    logger.debug("--- load schedule ---")
    num_schedule, schedule_np = load_schedule(settings["path_schedule"])
    logger.debug(f"num_schedule={num_schedule}")
    schedule_ml = matlab.uint64(schedule_np.tolist())
    logger.debug(schedule_ml)

    logger.debug("--- load empi list ---")
    num_outcome, empi_list_np = load_empi_list(settings["path_empi"], num_schedule)
    empi_list_ml = matlab.double(empi_list_np.tolist())
    logger.debug(f"num_outcome={num_outcome}")
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
