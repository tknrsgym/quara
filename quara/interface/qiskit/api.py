from quara.objects import state
from _pytest.monkeypatch import V
import numpy as np
from numpy import ndarray, result_type
from typing import List, Tuple, Union

from quara.interface.qiskit.conversion import (
    convert_empi_dists_qiskit_to_quara,
    convert_state_qiskit_to_quara,
    convert_state_quara_to_qiskit,
    convert_povm_qiskit_to_quara,
    convert_povm_quara_to_qiskit,
)

from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.qoperation import QOperation
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.standard_povmt import StandardPovmt
from quara.protocol.qtomography.standard.standard_qpt import StandardQpt
from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
)
from quara.minimization_algorithm.projected_gradient_descent_backtracking import (
    ProjectedGradientDescentBacktracking,
    ProjectedGradientDescentBacktrackingOption,
)
from quara.loss_function.weighted_probability_based_squared_error import (
    WeightedProbabilityBasedSquaredError,
    WeightedProbabilityBasedSquaredErrorOption,
)


def least_squares_estimator_wrapper(
    qtomography: StandardQTomography,
    empi_dists: List[Tuple[int, ndarray]],
    mode_weight: str,
) -> QOperation:
    estimator = LossMinimizationEstimator()
    loss = WeightedProbabilityBasedSquaredError()
    loss_option = WeightedProbabilityBasedSquaredErrorOption(mode_weight)
    algo = ProjectedGradientDescentBacktracking()
    algo_option = ProjectedGradientDescentBacktrackingOption()
    result = estimator.calc_estimate(
        qtomography=qtomography,
        empi_dists=empi_dists,
        loss=loss,
        loss_option=loss_option,
        algo=algo,
        algo_option=algo_option,
        is_computation_time_required=True,
    )
    return result.estimated_qoperation


def estimate_standard_qst_from_qiskit(
    mode_system: str,
    num_system: int,
    tester_povms: List[List[np.ndarray]],
    empi_dists: np.ndarray,
    shots: Union[List[int], int],
    label: List[int],
    estimator_name: str,
    schedules: Union[List[List[Tuple]], str],
) -> np.ndarray:

    c_sys = generate_composite_system(mode_system, num_system)
    tester_povms_quara = []
    for qiskit_povm in tester_povms:
        tester_povms_quara.append(convert_povm_qiskit_to_quara(qiskit_povm, c_sys))

    empi_dists_quara = convert_empi_dists_qiskit_to_quara(empi_dists, shots, label)
    qst = StandardQst(tester_povms_quara, schedules=schedules)
    if estimator_name == "linear":
        estimator = LinearEstimator()
        result = estimator.calc_estimate(qtomography=qst, empi_dists=empi_dists_quara)
        estimated_state = result.estimated_qoperation
    elif estimator_name == "least_squares":
        estimated_state = least_squares_estimator_wrapper(
            qst, empi_dists_quara, "identity"
        )
    else:
        raise ValueError("estimator_name is invalid")
    return convert_state_quara_to_qiskit(estimated_state)


def estimate_standard_qpt_from_qiskit(
    mode_system: str,
    num_system: int,
    tester_states: List[List[np.ndarray]],
    tester_povms: List[List[np.ndarray]],
    empi_dists: np.ndarray,
    shots: Union[List[int], int],
    label: List[int],
    estimator_name: str,
    schedules: Union[List[List[Tuple]], str],
) -> np.ndarray:

    c_sys = generate_composite_system(mode_system, num_system)
    tester_states_quara = []
    for qiskit_state in tester_states:
        tester_states_quara.append(convert_state_qiskit_to_quara(qiskit_state, c_sys))
    tester_povms_quara = []
    for qiskit_povms in tester_povms:
        tester_povms_quara.append(convert_povm_qiskit_to_quara(qiskit_povms, c_sys))
    empi_dists_quara = convert_empi_dists_qiskit_to_quara(empi_dists, shots, label)
    qpt = StandardQpt(
        states=tester_states_quara,
        povms=tester_povms_quara,
        on_para_eq_constraint=True,
        schedules=schedules,
    )
    if estimator_name == "linear":
        estimator = LinearEstimator()
        result = estimator.calc_estimate(
            qtomography=qpt,
            empi_dists=empi_dists_quara,
            is_computation_time_required=True,
        )
        estimate_gate = result.estimated_qoperation
    elif estimator_name == "least_squares":
        estimate_gate = least_squares_estimator_wrapper(
            qpt, empi_dists_quara, "identity"
        )
    else:
        raise ValueError("estimator is invalid")
    return convert_povm_quara_to_qiskit(estimate_gate)


def estimate_standard_povmt_from_qiskit(
    mode_system: str,
    num_system: int,
    tester_states: List[List[np.ndarray]],
    empi_dists: np.ndarray,
    shots: Union[List[int], int],
    label: List[int],
    num_outcomes: int,
    estimator_name: str,
    schedules: Union[List[List[Tuple]], str],
) -> np.ndarray:

    c_sys = generate_composite_system(mode_system, num_system)
    tester_states_quara = []
    for qiskit_state in tester_states:
        tester_states_quara.append(convert_state_qiskit_to_quara(qiskit_state, c_sys))
    empi_dists_quara = convert_empi_dists_qiskit_to_quara(empi_dists, shots, label)
    povmt = StandardPovmt(
        states=tester_states_quara,
        num_outcomes=num_outcomes,
        on_para_eq_constraint=True,
        schedules=schedules,
    )
    if estimator_name == "linear":
        estimator = LinearEstimator()
        result = estimator.calc_estimate(
            qtomography=povmt,
            empi_dists=empi_dists_quara,
            is_computation_time_required=True,
        )
        estimate_povm = result.estimated_qoperation
    elif estimator_name == "least_squares":
        estimate_povm = least_squares_estimator_wrapper(
            povmt, empi_dists_quara, "identity"
        )
    else:
        raise ValueError("estimator is invalid")
    return convert_povm_quara_to_qiskit(estimate_povm)


def generate_empi_dists_from_qutip_state(
    mode_system: str,
    num_system: int,
    true_state: List[List[np.ndarray]],
    tester_povms: List[List[np.ndarray]],
    num_sum: int,
    seed: int,
    schedules: Union[List[List[Tuple]], str],
) -> List[Tuple[int, ndarray]]:

    c_sys = generate_composite_system(mode_system, num_system)
    tester_povms_quara = []
    for qiskit_povm in tester_povms:
        tester_povms_quara.append(convert_povm_qiskit_to_quara(qiskit_povm, c_sys))
    true_state_quara = convert_state_qiskit_to_quara(true_state, c_sys)

    qst = StandardQst(
        tester_povms_quara, is_physicality_required=True, schedules=schedules, seed=seed
    )
    return qst.generate_empi_dists(state=true_state_quara, num_sum=num_sum)
