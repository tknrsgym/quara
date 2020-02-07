import logging
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import matlab

from quara.engine.matlabengine import MatlabEngine
from quara.protocol import simple_io
from quara.utils import number_util

logger = logging.getLogger(__name__)


def execute_from_csv(settings: dict) -> Tuple[np.ndarray, float]:
    """load data from csv files and execute lsQPT.
    
    Parameters
    ----------
    settings : dict
        following dictionary:

        - "dim": dimension of Hilbert space.
        - "num_state": number of state in the csv file.
        - "num_povm": number of povm in the csv file.
        - "num_outcome": number of outcome in the csv file.
        - "path_state": path of the state csv file.
        - "path_povm": path of the povm csv file.
        - "path_schedule": path of the schedule csv file.
        - "path_empi": path of the empi csv file.
        - "path_weight": path of the weight csv file.
        - "k": the length of an amplified gate sequence.
        - "matL0": the Hilbert-Schmidt representation of a Lindbladian in the computational basis.
        - "eps_logmat": a nonnegative parameter for numerical error tolerance.
    
    Returns
    -------
    Tuple[np.ndarray, float]
        see :func:`~quara.protocol.simple_lsqpt.execute`.
    """
    number_util.check_positive_number(settings["dim"], "dim")

    logger.debug("--- load state list ---")
    state_list = simple_io.load_state_list(
        settings["path_state"], settings["dim"], settings["num_state"]
    )
    logger.debug(state_list)

    logger.debug("--- load povm list ---")
    povm_list = simple_io.load_povm_list(
        settings["path_povm"],
        settings["dim"],
        settings["num_povm"],
        settings["num_outcome"],
    )
    logger.debug(povm_list)

    logger.debug("--- load schedule ---")
    num_schedule, schedule = simple_io.load_schedule(
        settings["path_schedule"], settings["num_state"], settings["num_povm"]
    )
    logger.debug(f"num_schedule={num_schedule}")
    logger.debug(schedule)

    logger.debug("--- load empi list ---")
    empi_list = simple_io.load_empi_list(
        settings["path_empi"], num_schedule, settings["num_outcome"]
    )
    logger.debug(empi_list)

    logger.debug("--- load weight list ---")
    weight_list = simple_io.load_weight_list(
        settings["path_weight"], num_schedule, settings["num_outcome"]
    )
    logger.debug(weight_list)

    number_util.check_positive_number(settings["k"], "k")

    logger.debug("--- load matL0 ---")
    matL0 = simple_io.load_matL0(settings["path_matL0"], settings["dim"])
    logger.debug(matL0)

    number_util.check_nonnegative_number(settings["eps_logmat"], "eps_logmat")

    eps_sedumi = 0.0  # matlab.double(0.0)
    int_verbose = 0  # matlab.uint8(1)
    choi, obj_value = execute(
        settings["dim"],
        state_list,
        povm_list,
        schedule,
        empi_list,
        weight_list,
        k,
        matL0,
        settings["eps_logmat"],
    )
    logger.debug("-----")
    logger.debug(f"choi={choi}")
    logger.debug(f"obj_value={obj_value}")

    return choi, obj_value


def execute(
    dim: int,
    state_list: np.ndarray,
    povm_list: np.ndarray,
    schedule: np.ndarray,
    empi_list: np.ndarray,
    weight_list: np.ndarray,
    k: int,
    matL0: np.ndarray,
    eps_logmat: float,
) -> Tuple[np.ndarray, float]:
    """execute lsQPT.
    
    Parameters
    ----------
    dim : int
        dimension of Hilbert space.
    state_list : np.ndarray
        list of density matrices.
        state list represented by ndarray of dtype ``np.complex128``.
        its shape is ``(num_state, dim * dim)``.
    povm_list : np.ndarray
        list of POVMs.
        povm list represented by ndarray of dtype ``np.complex128``.
        its shape is ``(num_povm, num_outcome, dim * dim)``.
    schedule : np.ndarray
        list of pair of labels for an input state and for measurement.
        schedule list represented by ndarray of dtype ``np.uint16``.
        its shape is ``(number of schedule, 2)``.
    empi_list : np.ndarray
        list of empirical distributions.
        empi list represented by ndarray of dtype ``np.float64``.
        its shape is ``(num_schedule, num_outcome)``.
    weight_list : np.ndarray
        list of weight matrices.
        weight list represented by ndarray of dtype ``np.float64``.
        its shape is ``(num_schedule, num_outcome, num_outcome)``.
    k : int
        the length of an amplified gate sequence. positive integer.
    matL0 : np.ndarray
        the Hilbert-Schmidt representation of a Lindbladian in the computational basis represented by ndarray of dtype ``np.complex128``.
        its shape is ``(dim * dim, dim * dim)``
    eps_logmat : float
        a nonnegative parameter for numerical error tolerance.
    
    Returns
    -------
    Tuple[np.ndarray, float]
        first value of tuple is a Choi matrix represented by ndarray of dtype ``np.complex128``.
        its shape is ``(dim * dim, dim * dim)``.
        second value of tuple is a weighted squared distance between optimized value and actual value.
    """
    state_list_ml = matlab.double(state_list.tolist(), is_complex=True)
    povm_list_ml = matlab.double(povm_list.tolist(), is_complex=True)
    schedule = schedule + 1
    schedule_ml = matlab.uint64(schedule.tolist())
    logger.debug(schedule_ml)
    empi_list_ml = matlab.double(empi_list.tolist())
    weight_list_ml = matlab.double(weight_list.tolist())
    matL0_ml = matlab.double(matL0.tolist(), is_complex=True)

    eps_sedumi = 0.0  # matlab.double(0.0)
    int_verbose = 0  # matlab.uint8(1)
    with MatlabEngine() as engine:
        choi_ml, wsd = engine.simple_lsqpt(
            float(dim),
            state_list_ml,
            povm_list_ml,
            schedule_ml,
            weight_list_ml,
            empi_list_ml,
            eps_sedumi,
            int_verbose,
            float(k),
            matL0_ml,
            eps_logmat,
            nargout=2,
        )
    choi_np = np.array(choi_ml, dtype=np.complex128)

    return choi_np, wsd
