from cgi import test
from multiprocessing.sharedctypes import Value
import numpy as np
from typing import List, Tuple, Union

from quara.objects.state import State
from quara.objects.povm import Povm
from quara.interface.qiskit.conversion import (
    convert_empi_dists_qiskit_to_quara,
    convert_empi_dists_quara_to_qiskit,
    convert_empi_dists_quara_to_qiskit_shots,
    convert_gate_qiskit_to_quara,
    convert_gate_quara_to_qiskit,
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
from quara.protocol.qtomography.standard.standard_qmpt import StandardQmpt
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

from quara.interface.cvxpy.qtomography.standard.estimator import (
    CvxpyLossMinimizationEstimator,
)
from quara.interface.cvxpy.qtomography.standard.loss_function import (
    CvxpyRelativeEntropy,
)


from quara.interface.cvxpy.qtomography.standard.loss_function import (
    CvxpyLossFunction,
    CvxpyLossFunctionOption,
    CvxpyRelativeEntropy,
    CvxpyUniformSquaredError,
    # CvxpyApproximateRelativeEntropyWithoutZeroProbabilityTerm,
    CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm,
    # CvxpyApproximateRelativeEntropyWithZeroProbabilityTermSquared,
)

# from quara.protocol.qtomography.standard.preprocessing import (
#     # combine_nums_prob_dists,
# )
from quara.interface.cvxpy.qtomography.standard.minimization_algorithm import (
    CvxpyMinimizationAlgorithm,
    CvxpyMinimizationAlgorithmOption,
)
from quara.interface.cvxpy.qtomography.standard.estimator import (
    CvxpyLossMinimizationEstimator,
)

# def least_squares_estimator_wrapper(
#     qtomography: StandardQTomography,
#     empi_dists: List[Tuple[int, np.ndarray]],
#     mode_weight: str,
# ) -> QOperation:
#     estimator = LossMinimizationEstimator()
#     loss = WeightedProbabilityBasedSquaredError()
#     loss_option = WeightedProbabilityBasedSquaredErrorOption(mode_weight)
#     algo = ProjectedGradientDescentBacktracking()
#     algo_option = ProjectedGradientDescentBacktrackingOption()
#     result = estimator.calc_estimate(
#         qtomography=qtomography,
#         empi_dists=empi_dists,
#         loss=loss,
#         loss_option=loss_option,
#         algo=algo,
#         algo_option=algo_option,
#         is_computation_time_required=True,
#     )
#     return result.estimated_qoperation


# def estimate_standard_qst_from_qiskit(
#     mode_system: str,
#     num_system: int,
#     tester_povms: List[List[np.ndarray]],
#     empi_dists: np.ndarray,
#     shots: Union[List[int], int],
#     label: List[int],
#     estimator_name: str,
#     schedules: Union[List[List[Tuple]], str],
# ) -> np.ndarray:

#     """Calculates estimate variables and returns an estimate of a quantum state by executing quantum state tomography of quara using tester POVMs and  empirical distirbutions.

#     Parameters
#     ----------
#     mode_system: str
#         "qubit"
#     num_system: int
#         number of qubits
#     tester_povms:List[List[np.ndarray]]
#         testers of QST
#     empi_dists:np.ndarray
#         empirical distribution to calculate estimators of variables
#     shots: int
#         sum number of shot
#     label: List[int]
#         label for kind of one POVM
#     estimator_name: str
#         "linear" or "least_squares"
#     schedules: Union[List[List[Tuple]],str]
#         schedule of the experiment

#     Returns
#     ---------
#     np.ndarray
#         estimated result of the quantum state

#     Raises
#     -------
#     ValueError
#         ``estimator name`` is invalid

#     """

#     c_sys = generate_composite_system(mode_system, num_system)
#     tester_povms_quara = []
#     for qiskit_povm in tester_povms:
#         tester_povms_quara.append(convert_povm_qiskit_to_quara(qiskit_povm, c_sys))

#     empi_dists_quara = convert_empi_dists_qiskit_to_quara(empi_dists, shots, label)
#     qst = StandardQst(tester_povms_quara, schedules=schedules)
#     if estimator_name == "linear":
#         estimator = LinearEstimator()
#         result = estimator.calc_estimate(qtomography=qst, empi_dists=empi_dists_quara)
#         estimated_state = result.estimated_qoperation
#     elif estimator_name == "least_squares":
#         estimated_state = least_squares_estimator_wrapper(
#             qst, empi_dists_quara, "identity"
#         )
#     else:
#         raise ValueError("estimator_name is invalid")
#     return convert_state_quara_to_qiskit(estimated_state)


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
    # type_qoperation:  <- “state”, “povm”, “gate”, “mprocess”
    # estimator_name: “maximum-likelihood”, “approximate-maximum-likelihood”, “least-squares”
    # name_solver: , <- “mosek” 後から追加される可能性あり
    # num_outcomes:  <- “povm”, “mprocess”の場合は2以上. “state”と“gate”の場合はNone
    # Valildate
    expected_estimator = [
        "maximum-likelihood",
        "approximate-maximum-likelihood",
        "least-squares",
    ]
    if estimator_name not in expected_estimator:
        error_message = "estimator_name must be 'maximum-likelihood', 'approximate-maximum-likelihood', or 'least-squares'."
        raise ValueError(error_message)
    expected_solver = ["mosek"]
    if name_solver not in expected_solver:
        error_message = "name_solver must be 'mosek'."
        raise ValueError(error_message)
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


def estimate_standard_qst_with_cvxpy(
    mode_system: str,
    num_system: int,
    tester_povms: List[List[np.ndarray]],
    empi_dists: np.ndarray,
    name_solver: str,
    estimator_name: str,
    schedules: Union[List[List[Tuple]], str],
) -> np.ndarray:

    estimator = CvxpyRelativeEntropy()
    # TODO: 仮実装
    loss = CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm()
    loss_option = CvxpyLossFunctionOption()
    algo = CvxpyMinimizationAlgorithm()
    algo_option = CvxpyMinimizationAlgorithmOption(name_solver=name_solver)
    estimator = CvxpyLossMinimizationEstimator()

    sqt = StandardQst(
        povms=tester_povms, schedules=schedules, on_para_eq_constraint=True
    )
    estimated_qoperation = estimator.calc_estimate(
        qtomography=sqt,
        empi_dists=empi_dists,
        loss=loss,
        loss_option=loss_option,
        algo=algo,
        algo_option=algo_option,
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

    estimator = CvxpyRelativeEntropy()
    # TODO: 仮実装
    loss = CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm()
    loss_option = CvxpyLossFunctionOption()
    algo = CvxpyMinimizationAlgorithm()
    algo_option = CvxpyMinimizationAlgorithmOption(name_solver=name_solver)
    estimator = CvxpyLossMinimizationEstimator()

    sqt = StandardPovmt(
        states=tester_states,
        num_outcomes=num_outcomes,
        on_para_eq_constraint=True,
        schedules=schedules,
    )
    estimated_qoperation = estimator.calc_estimate(
        qtomography=sqt,
        empi_dists=empi_dists,
        loss=loss,
        loss_option=loss_option,
        algo=algo,
        algo_option=algo_option,
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

    estimator = CvxpyRelativeEntropy()
    # TODO: 仮実装
    loss = CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm()
    loss_option = CvxpyLossFunctionOption()
    algo = CvxpyMinimizationAlgorithm()
    algo_option = CvxpyMinimizationAlgorithmOption(name_solver=name_solver)
    estimator = CvxpyLossMinimizationEstimator()

    sqt = StandardQpt(
        states=tester_states,
        povms=tester_povms,
        on_para_eq_constraint=True,
        schedules=schedules,
    )
    estimated_qoperation = estimator.calc_estimate(
        qtomography=sqt,
        empi_dists=empi_dists,
        loss=loss,
        loss_option=loss_option,
        algo=algo,
        algo_option=algo_option,
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

    estimator = CvxpyRelativeEntropy()
    # TODO: 仮実装
    loss = CvxpyApproximateRelativeEntropyWithZeroProbabilityTerm()
    loss_option = CvxpyLossFunctionOption()
    algo = CvxpyMinimizationAlgorithm()
    algo_option = CvxpyMinimizationAlgorithmOption(name_solver=name_solver)
    estimator = CvxpyLossMinimizationEstimator()

    sqt = StandardQmpt(
        states=tester_states,
        povms=tester_povms,
        num_outcomes=num_outcomes,
        on_para_eq_constraint=True,
        schedules=schedules,
    )
    estimated_qoperation = estimator.calc_estimate(
        qtomography=sqt,
        empi_dists=empi_dists,
        loss=loss,
        loss_option=loss_option,
        algo=algo,
        algo_option=algo_option,
        is_computation_time_required=True,
    )

    return estimated_qoperation


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

    """Calculates estimate variables and returns an estimate of a quantum gate by executing quantum process tomography of quara using tester states, POVMs and  empirical distirbutions.

    Parameters
    ----------
    mode_system: str
        "qubit"
    num_system: int
        number of qubits
    tester_states:List[List[np.ndarray]]
        testers of QPT
    tester_povms:List[List[np.ndarray]]
        testers of QPT
    empi_dists:np.ndarray
        empirical distribution to calculate estimators of variables
    shots: int
        sum number of shot
    label: List[int]
        label for kind of one POVM
    estimator_name: str
        "linear" or "least_squares"
    schedules: Union[List[List[Tuple]],str]
        schedule of the experiment

    Returns
    ---------
    np.ndarray
        estimated result of the quantum gate

    Raises
    -------
    ValueError
        ``estimator name`` is invalid

    """

    dim = 2 ** num_system
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
    return convert_gate_quara_to_qiskit(estimate_gate, dim)


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

    """Calculates estimate variables and returns an estimate of a quantum povm by executing quantum POVM tomography of quara using tester POVMs and  empirical distirbutions.

    Parameters
    ----------
    mode_system: str
        "qubit"
    num_system: int
        number of qubits
    tester_states:List[List[np.ndarray]]
        testers of POVMT
    num_outcomes: int
        number of outcome values of the POVM that will be estimated
    empi_dists:np.ndarray
        empirical distribution to calculate estimators of variables
    shots: int
        sum number of shot
    label: List[int]
        label for kind of one POVM
    estimator_name: str
        "linear" or "least_squares"
    schedules: Union[List[List[Tuple]],str]
        schedule of the experiment

    Returns
    ---------
    np.ndarray
        estimated result of the quantum POVM

    Raises
    -------
    ValueError
        ``estimator name`` is invalid

    """

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


def generate_empi_dists_from_qiskit_state(
    mode_system: str,
    num_system: int,
    true_state: List[List[np.ndarray]],
    tester_povms: List[List[np.ndarray]],
    num_sum: int,
    seed: int,
    schedules: Union[List[List[Tuple]], str],
) -> List[Tuple[int, np.ndarray]]:

    """Generate empirical distributions using the data generated from probability distributions of specified schedules.

    Parameters
    ----------
    mode_system: str
        "qubit"
    num_system: int
        number of qubits
    true_state: List[List[np.ndarray]]
        true quantum state
    tester_povms: List[List[np.ndarray]]
        testers of QST
    num_sum: int
        number of observations per POVM
    seed: int
        seed number for generating random number
    schedules: Union[List[List[Tuple]],str]
        schedule of the experiment

    Returns
    ---------
    List[Tuple[int, np.ndarray]]
        empirical disribution which is a List of empirical probabilities.
    """

    c_sys = generate_composite_system(mode_system, num_system)
    tester_povms_quara = []
    for qiskit_povm in tester_povms:
        tester_povms_quara.append(convert_povm_qiskit_to_quara(qiskit_povm, c_sys))
    true_state_quara = convert_state_qiskit_to_quara(true_state, c_sys)

    qst = StandardQst(
        tester_povms_quara,
        on_para_eq_constraint=True,
        schedules=schedules,
        seed_data=seed,
    )
    return qst.generate_empi_dists(state=true_state_quara, num_sum=num_sum)


def generate_empi_dists_from_qiskit_povm(
    mode_system: str,
    num_system: int,
    true_povm: List[List[np.ndarray]],
    tester_states: List[List[np.ndarray]],
    num_sum: int,
    seed: int,
    schedules: Union[List[List[Tuple]], str],
) -> List[Tuple[int, np.ndarray]]:

    """Generate empirical distributions using the data generated from probability distributions of specified schedules.

    Parameters
    ----------
    mode_system: str
        "qubit"
    num_system: int
        number of qubits
    tester_states: List[List[np.ndarray]]
        testers of POVMT
    tester_povms: List[List[np.ndarray]]
        true POVM
    num_sum: int
        number of observations per state
    seed: int
        seed number for generating random number
    schedules: Union[List[List[Tuple]],str]
        schedule of the experiment

    Returns
    ---------
    List[Tuple[int, np.ndarray]]
        empirical disribution which is a List of empirical probabilities.
    """

    c_sys = generate_composite_system(mode_system, num_system)
    tester_states_quara = []
    for qiskit_state in tester_states:
        tester_states_quara.append(convert_state_qiskit_to_quara(qiskit_state, c_sys))
    true_povm_quara = convert_povm_qiskit_to_quara(true_povm, c_sys)

    povmt = StandardPovmt(
        tester_states_quara,
        num_outcomes=true_povm_quara.num_outcomes,
        on_para_eq_constraint=True,
        schedules=schedules,
        seed_data=seed,
    )
    return povmt.generate_empi_dists(povm=true_povm_quara, num_sum=num_sum)


def generate_empi_dists_from_qiskit_gate(
    mode_system: str,
    num_system: int,
    true_gate: np.ndarray,
    tester_states: List[List[np.ndarray]],
    tester_povms: List[List[np.ndarray]],
    num_sum: int,
    seed: int,
    schedules: Union[List[List[Tuple]], str],
) -> List[Tuple[int, np.ndarray]]:

    """Generate empirical distributions using the data generated from probability distributions of specified schedules.

    Parameters
    ----------
    mode_system: str
        "qubit"
    num_system: int
        number of qubits
    tester_states: List[List[np.ndarray]]
        testers of QPT
    tester_povms: List[List[np.ndarray]]
        testers of QPT
    num_sum: int
        number of observations per POVM
    seed: int
        seed number for generating random number
    schedules: Union[List[List[Tuple]],str]
        schedule of the experiment

    Returns
    ---------
    List[Tuple[int, np.ndarray]]
        empirical disribution which is a List of empirical probabilities.
    """

    dim = 2 ** num_system
    c_sys = generate_composite_system(mode_system, num_system)
    tester_states_quara = []
    for qiskit_state in tester_states:
        tester_states_quara.append(convert_state_qiskit_to_quara(qiskit_state, c_sys))
    tester_povms_quara = []
    for qiskit_povm in tester_povms:
        tester_povms_quara.append(convert_povm_qiskit_to_quara(qiskit_povm, c_sys))
    true_gate_quara = convert_gate_qiskit_to_quara(true_gate, c_sys, dim)

    qpt = StandardQpt(
        tester_states_quara,
        tester_povms_quara,
        on_para_eq_constraint=True,
        schedules=schedules,
        seed_data=seed,
    )
    return qpt.generate_empi_dists(gate=true_gate_quara, num_sum=num_sum)


def generate_empi_dists_from_quara(
    empi_dists_quara: List[Tuple[int, np.ndarray]]
) -> Tuple[List[int], np.ndarray]:

    """Generate empirical distributions in qiskit and label from empirica distribution of quara.

    Parameters
    ----------
    empi_dists_quara: List[Tuple[int, np.ndarray]]

    Returns
    ---------
    Tuple[List[int], np.ndarray]
        Tuple of label and empirical distribution.
    """

    empi_dists = convert_empi_dists_quara_to_qiskit(empi_dists_quara)
    label = []
    for i in empi_dists_quara:
        label.append(np.size(i[1]))
    return label, empi_dists
