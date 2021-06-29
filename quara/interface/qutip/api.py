from typing import List, Tuple, Union

from numpy import ndarray
from qutip import Qobj

from quara.interface.qutip.conversion import (
    convert_gate_quara_to_qutip,
    convert_gate_qutip_to_quara,
    convert_state_quara_to_qutip,
    convert_state_qutip_to_quara,
    convert_povm_quara_to_qutip,
    convert_povm_qutip_to_quara,
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


def estimate_standard_qst_from_qutip(
    mode_system: str,
    num_system: int,
    tester_povms: List[List[Qobj]],
    empi_dists: List[Tuple[int, ndarray]],
    estimator_name: str,
    schedules: Union[List[List[Tuple]], str],
) -> Qobj:
    """Calculates estimate variables and returns an estimate of a quantum state by executing quantum state tomography using tester POVMs and empirical distributions.

    Parameters
    ----------
    mode_system: str
        "qubit" or "qutrit"
    num_system: int
        number of qubits or qutrits
    tester_povms: List[List[Qobj]]
        testers of QST.
    empi_dists: List[Tuple[int, np.ndarray]]
        empirical distributions to calculate estimates of variables.
    estimator_name: str
        "linear" or "least_squares".
    schedules: Union[List[List[Tuple]], str]
        schedule of the experiment.

    Returns
    -------
    Qobj
        estimated result of the quantum state.

    Raises
    ------
    ValueError
        ``estimator_name`` is invalid.
    """
    c_sys = generate_composite_system(mode_system, num_system)
    tester_povms_quara = []
    for qutip_povm in tester_povms:
        tester_povms_quara.append(convert_povm_qutip_to_quara(qutip_povm, c_sys))

    qst = StandardQst(tester_povms_quara, schedules=schedules)
    if estimator_name == "linear":
        estimator = LinearEstimator()
        result = estimator.calc_estimate(
            qtomography=qst, empi_dists=empi_dists, is_computation_time_required=True
        )
        estimated_state = result.estimated_qoperation
    elif estimator_name == "least_squares":
        estimated_state = least_squares_estimator_wrapper(qst, empi_dists, "identity")
    else:
        raise ValueError("estimator_name is invalid")
    return convert_state_quara_to_qutip(estimated_state)


def estimate_standard_povmt_from_qutip(
    mode_system: str,
    num_system: int,
    tester_states: List[Qobj],
    num_outcomes: int,
    empi_dists: List[Tuple[int, ndarray]],
    estimator_name: str,
    schedules: Union[List[List[Tuple]], str],
) -> Qobj:
    """Calculates estimate variables and returns an estimate of a quantum state by executing POVM tomography using tester states and empirical distributions.

    Parameters
    ----------
    mode_system: str
        "qubit" or "qutrit"
    num_system: int
        number of qubits or qutrits
    tester_states: List[Qobj]
        testers of POVMT.
    num_outcomes: int
        number of outcome values of the POVM that will be estimated.
    empi_dists: List[Tuple[int, np.ndarray]]
        empirical distributions to calculate estimates of variables.
    estimator_name: str
        "linear" or "least_squares".
    schedules: Union[List[List[Tuple]], str]
        schedule of the experiment.

    Returns
    -------
    Qobj
        estimated result of the POVM.

    Raises
    ------
    ValueError
        ``estimator_name`` is invalid.
    """
    c_sys = generate_composite_system(mode_system, num_system)
    tester_states_quara = []
    for qutip_state in tester_states:
        tester_states_quara.append(convert_state_qutip_to_quara(qutip_state, c_sys))
    povmt = StandardPovmt(
        tester_states_quara,
        num_outcomes=num_outcomes,
        on_para_eq_constraint=True,
        schedules=schedules,
    )
    if estimator_name == "linear":
        estimator = LinearEstimator()
        result = estimator.calc_estimate(
            qtomography=povmt, empi_dists=empi_dists, is_computation_time_required=True
        )
        estimated_povm = result.estimated_qoperation
    elif estimator_name == "least_squares":
        estimated_povm = least_squares_estimator_wrapper(povmt, empi_dists, "identity")
    else:
        raise ValueError("estimator_name is invalid")
    return convert_povm_quara_to_qutip(estimated_povm)


def estimate_standard_qpt_from_qutip(
    mode_system: str,
    num_system: int,
    tester_states: List[Qobj],
    tester_povms: List[Qobj],
    empi_dists: List[Tuple[int, List[int]]],
    estimator_name: str,
    schedules: Union[List[List[Tuple]], str],
) -> Qobj:
    """Calculates estimate variables and returns an estimate of a quantum gate by executing quantum process tomography using tester states, tester POVMs and empirical distributions.

    Parameters
    ----------
    mode_system: str
        "qubit" or "qutrit"
    num_system: int
        number of qubits or qutrits
    tester_states: List[Qobj]
        tester states of QPT.
    tester_povms: List[List[Qobj]]
        tester POVMs of QPT.
    empi_dists: List[Tuple[int, np.ndarray]]
        empirical distributions to calculate estimates of variables.
    estimator_name: str
        "linear" or "least_squares".
    schedules: Union[List[List[Tuple]], str]
        schedule of the experiment.

    Returns
    -------
    Qobj
        estimated result of the quantum gate in super representation.

    Raises
    ------
    ValueError
        ``estimator_name`` is invalid.
    """
    c_sys = generate_composite_system(mode_system, num_system)
    tester_states_quara = []
    for qutip_state in tester_states:
        tester_states_quara.append(convert_state_qutip_to_quara(qutip_state, c_sys))
    tester_povms_quara = []
    for qutip_povm in tester_povms:
        tester_povms_quara.append(convert_povm_qutip_to_quara(qutip_povm, c_sys))
    qpt = StandardQpt(
        states=tester_states_quara,
        povms=tester_povms_quara,
        on_para_eq_constraint=True,
        schedules=schedules,
    )
    if estimator_name == "linear":
        estimator = LinearEstimator()
        result = estimator.calc_estimate(
            qtomography=qpt, empi_dists=empi_dists, is_computation_time_required=True
        )
        estimated_gate = result.estimated_qoperation
    elif estimator_name == "least_squares":
        estimated_gate = least_squares_estimator_wrapper(qpt, empi_dists, "identity")
    else:
        raise ValueError("estimator_name is invalid")
    return convert_gate_quara_to_qutip(estimated_gate)


def generate_empi_dists_from_qutip_state(
    mode_system: str,
    num_system: int,
    true_state: Qobj,
    tester_povms: List[List[Qobj]],
    num_sum: int,
    seed: int,
    schedules: Union[List[List[Tuple]], str],
) -> List[Tuple[int, ndarray]]:
    """Generate empirical distributions using the data generated from probability distribution of specified schedules.

    Parameters
    ----------
    mode_system: str
        "qubit" or "qutrit"
    num_system: int
        number of qubits or qutrits
    true_state: Qobj
        true quantum state.
    tester_povms: List[List[Qobj]]
        testers of QST.
    num_sum: int
        number of observations per POVM.
    seed: int
        seed number for generating random number.
    schedules: Union[List[List[Tuple]], str]
        schedule of the experiment.

    Returns
    -------
    List[Tuple[int, ndarray]]
        Empirical distribution which is a List of empirical probabilities.
    """
    c_sys = generate_composite_system(mode_system, num_system)
    tester_povms_quara = []
    for qutip_povm in tester_povms:
        tester_povms_quara.append(convert_povm_qutip_to_quara(qutip_povm, c_sys))
    true_state_quara = convert_state_qutip_to_quara(true_state, c_sys)

    qst = StandardQst(
        tester_povms_quara, on_para_eq_constraint=True, schedules=schedules, seed=seed
    )
    return qst.generate_empi_dists(state=true_state_quara, num_sum=num_sum)


def generate_empi_dists_from_qutip_povm(
    mode_system: str,
    num_system: int,
    true_povm: List[Qobj],
    tester_states: List[Qobj],
    num_sum: int,
    seed: int,
    schedules: Union[List[List[Tuple]], str],
) -> List[Tuple[int, ndarray]]:
    """Generate empirical distributions using the data generated from probability distribution of specified schedules.

    Parameters
    ----------
    mode_system: str
        "qubit" or "qutrit"
    num_system: int
        number of qubits or qutrits
    true_povm: Qobj
        true POVM.
    tester_states: List[List[Qobj]]
        testers of POVMT.
    num_sum: int
        number of observations per state.
    seed: int
        seed number for generating random number.
    schedules: Union[List[List[Tuple]], str]
        schedule of the experiment.

    Returns
    -------
    List[Tuple[int, ndarray]]
        Empirical distribution which is a List of empirical probabilities.
    """
    c_sys = generate_composite_system(mode_system, num_system)
    tester_states_quara = []
    for qutip_state in tester_states:
        tester_states_quara.append(convert_state_qutip_to_quara(qutip_state, c_sys))
    true_povm_quara = convert_povm_qutip_to_quara(true_povm, c_sys)

    povmt = StandardPovmt(
        tester_states_quara,
        num_outcomes=true_povm_quara.num_outcomes,
        on_para_eq_constraint=True,
        schedules=schedules,
        seed=seed,
    )
    return povmt.generate_empi_dists(povm=true_povm_quara, num_sum=num_sum)


def generate_empi_dists_from_qutip_gate(
    mode_system: str,
    num_system: int,
    true_gate: Qobj,
    tester_states: List[Qobj],
    tester_povms: List[List[Qobj]],
    num_sum: int,
    seed: int,
    schedules: Union[List[List[Tuple]], str],
) -> List[Tuple[int, ndarray]]:
    """Generate empirical distributions using the data generated from probability distribution of specified schedules.

    Parameters
    ----------
    mode_system: str
        "qubit" or "qutrit"
    num_system: int
        number of qubits or qutrits
    true_state: Qobj
        true quantum state.
    tester_states: List[List[Qobj]]
        tester states of QPT.
    tester_povms: List[List[Qobj]]
        tester POVMs of QPT.
    num_sum: int
        number of observations per POVM.
    seed: int
        seed number for generating random number.
    schedules: Union[List[List[Tuple]], str]
        schedule of the experiment.

    Returns
    -------
    List[Tuple[int, ndarray]]
        Empirical distribution which is a List of empirical probabilities.
    """
    c_sys = generate_composite_system(mode_system, num_system)
    tester_states_quara = []
    for qutip_state in tester_states:
        tester_states_quara.append(convert_state_qutip_to_quara(qutip_state, c_sys))
    tester_povms_quara = []
    for qutip_povm in tester_povms:
        tester_povms_quara.append(convert_povm_qutip_to_quara(qutip_povm, c_sys))
    true_gate_quara = convert_gate_qutip_to_quara(true_gate, c_sys)

    qpt = StandardQpt(
        tester_states_quara,
        tester_povms_quara,
        on_para_eq_constraint=True,
        schedules=schedules,
        seed=seed,
    )
    return qpt.generate_empi_dists(gate=true_gate_quara, num_sum=num_sum)