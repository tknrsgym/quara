import numpy as np
from typing import List, Tuple, Union

from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.qoperation import QOperation
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.standard_povmt import StandardPovmt
from quara.protocol.qtomography.standard.standard_qpt import StandardQpt
from quara.protocol.qtomography.standard.standard_qmpt import StandardQmpt
from quara.interface.cvxpy.qtomography.standard.estimator import (
    CvxpyLossMinimizationEstimator,
)
from quara.interface.cvxpy.qtomography.standard.loss_function import (
    CvxpyRelativeEntropy,
)
from quara.interface.cvxpy.qtomography.standard.loss_function import (
    CvxpyLossFunctionOption,
    CvxpyRelativeEntropy,
    CvxpyUniformSquaredError,
    CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm,
)
from quara.interface.cvxpy.qtomography.standard.minimization_algorithm import (
    CvxpyMinimizationAlgorithm,
    CvxpyMinimizationAlgorithmOption,
)
from quara.interface.cvxpy.qtomography.standard.estimator import (
    CvxpyLossMinimizationEstimator,
)


def estimate_standard_qtomography_with_cvxpy(
    mode_system: str,
    num_system: int,
    type_qoperation: str,
    tester: Union[List[QOperation], List[List[QOperation]]],
    schedules: Union[List[List[Tuple]], str],
    empi_dists: List[Tuple[int, np.ndarray]],
    estimator_name: str,
    name_solver: str,
    num_outcomes: int = None,
) -> QOperation:
    # Valildate
    if type_qoperation in ["povm", "mprocess"]:
        if num_outcomes < 2:
            error_message = "If type_qoperation is 'povm' or 'mprocess', then num_outcomes must be greater than or equal to 2."
            raise ValueError(error_message)

    if type_qoperation == "state":
        estimated_qoperation = estimate_standard_qst_with_cvxpy(
            mode_system=mode_system,
            num_system=num_system,
            tester_povms=tester,
            empi_dists=empi_dists,
            name_solver=name_solver,
            estimator_name=estimator_name,
            schedules=schedules,
        )
    elif type_qoperation == "povm":
        estimated_qoperation = estimate_standard_povmt_with_cvxpy(
            mode_system=mode_system,
            num_system=num_system,
            tester_states=tester,
            empi_dists=empi_dists,
            name_solver=name_solver,
            estimator_name=estimator_name,
            schedules=schedules,
            num_outcomes=num_outcomes,
        )
    elif type_qoperation == "gate":
        states = [t for t in tester if type(t) == State]
        povms = [t for t in tester if type(t) == Povm]
        estimated_qoperation = estimate_standard_qpt_with_cvxpy(
            mode_system=mode_system,
            num_system=num_system,
            tester_states=states,
            tester_povms=povms,
            empi_dists=empi_dists,
            name_solver=name_solver,
            estimator_name=estimator_name,
            schedules=schedules,
        )
    elif type_qoperation == "mprocess":
        states = [t for t in tester if type(t) == State]
        povms = [t for t in tester if type(t) == Povm]
        estimated_qoperation = estimate_standard_qmpt_with_cvxpy(
            mode_system=mode_system,
            num_system=num_system,
            tester_states=states,
            tester_povms=povms,
            empi_dists=empi_dists,
            name_solver=name_solver,
            estimator_name=estimator_name,
            schedules=schedules,
            num_outcomes=num_outcomes,
        )
    else:
        error_message = "expected_type must be 'state', 'povm', 'gate', or 'mprocess'."
        raise ValueError(error_message)
    return estimated_qoperation


def _get_estimator_and_options(estimator_name: str, name_solver: str) -> dict:
    expected_solver = ["mosek"]
    if name_solver not in expected_solver:
        error_message = "name_solver must be 'mosek'."
        raise ValueError(error_message)

    estimator = CvxpyLossMinimizationEstimator()
    loss_option = CvxpyLossFunctionOption()
    algo = CvxpyMinimizationAlgorithm()
    algo_option = CvxpyMinimizationAlgorithmOption(name_solver=name_solver)

    if estimator_name == "maximum-likelihood":
        loss = CvxpyRelativeEntropy()
    elif estimator_name == "approximate-maximum-likelihood":
        loss = CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm()
    elif estimator_name == "least-squares":
        loss = CvxpyUniformSquaredError()
    else:
        error_message = "estimator_name must be 'maximum-likelihood', 'approximate-maximum-likelihood', or 'least-squares'."
        raise ValueError(error_message)

    data_dict = dict(
        estimator=estimator,
        loss=loss,
        loss_option=loss_option,
        algo=algo,
        algo_option=algo_option,
    )
    return data_dict


def estimate_standard_qst_with_cvxpy(
    mode_system: str,
    num_system: int,
    tester_povms: List[List[np.ndarray]],
    empi_dists: np.ndarray,
    name_solver: str,
    estimator_name: str,
    schedules: Union[List[List[Tuple]], str],
) -> np.ndarray:

    estimator_data = _get_estimator_and_options(estimator_name, name_solver)

    sqt = StandardQst(
        povms=tester_povms, schedules=schedules, on_para_eq_constraint=True
    )
    estimated_qoperation = estimator_data["estimator"].calc_estimate(
        qtomography=sqt,
        empi_dists=empi_dists,
        loss=estimator_data["loss"],
        loss_option=estimator_data["loss_option"],
        algo=estimator_data["algo"],
        algo_option=estimator_data["algo_option"],
        is_computation_time_required=True,
    )

    return estimated_qoperation


def estimate_standard_povmt_with_cvxpy(
    mode_system: str,
    num_system: int,
    tester_states: List[List[np.ndarray]],
    empi_dists: np.ndarray,
    name_solver: str,
    estimator_name: str,
    schedules: Union[List[List[Tuple]], str],
    num_outcomes: int,
) -> np.ndarray:

    estimator_data = _get_estimator_and_options(estimator_name, name_solver)

    sqt = StandardPovmt(
        states=tester_states,
        num_outcomes=num_outcomes,
        on_para_eq_constraint=True,
        schedules=schedules,
    )
    estimated_qoperation = estimator_data["estimator"].calc_estimate(
        qtomography=sqt,
        empi_dists=empi_dists,
        loss=estimator_data["loss"],
        loss_option=estimator_data["loss_option"],
        algo=estimator_data["algo"],
        algo_option=estimator_data["algo_option"],
        is_computation_time_required=True,
    )

    return estimated_qoperation


def estimate_standard_qpt_with_cvxpy(
    mode_system: str,
    num_system: int,
    tester_states: List[List[np.ndarray]],
    tester_povms: List[List[np.ndarray]],
    empi_dists: np.ndarray,
    name_solver: str,
    estimator_name: str,
    schedules: Union[List[List[Tuple]], str],
) -> np.ndarray:

    estimator_data = _get_estimator_and_options(estimator_name, name_solver)

    sqt = StandardQpt(
        states=tester_states,
        povms=tester_povms,
        on_para_eq_constraint=True,
        schedules=schedules,
    )
    estimated_qoperation = estimator_data["estimator"].calc_estimate(
        qtomography=sqt,
        empi_dists=empi_dists,
        loss=estimator_data["loss"],
        loss_option=estimator_data["loss_option"],
        algo=estimator_data["algo"],
        algo_option=estimator_data["algo_option"],
        is_computation_time_required=True,
    )

    return estimated_qoperation


def estimate_standard_qmpt_with_cvxpy(
    mode_system: str,
    num_system: int,
    tester_states: List[List[np.ndarray]],
    tester_povms: List[List[np.ndarray]],
    empi_dists: np.ndarray,
    name_solver: str,
    estimator_name: str,
    schedules: Union[List[List[Tuple]], str],
    num_outcomes: int,
) -> np.ndarray:

    estimator_data = _get_estimator_and_options(estimator_name, name_solver)

    sqt = StandardQmpt(
        states=tester_states,
        povms=tester_povms,
        num_outcomes=num_outcomes,
        on_para_eq_constraint=True,
        schedules=schedules,
    )
    estimated_qoperation = estimator_data["estimator"].calc_estimate(
        qtomography=sqt,
        empi_dists=empi_dists,
        loss=estimator_data["loss"],
        loss_option=estimator_data["loss_option"],
        algo=estimator_data["algo"],
        algo_option=estimator_data["algo_option"],
        is_computation_time_required=True,
    )

    return estimated_qoperation
