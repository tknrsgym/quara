from typing import List, Tuple

import numpy as np
import numpy.testing as npt
import pytest

import quara.objects.qoperation_typical as qt
from quara.objects.composite_system_typical import generate_composite_system

from quara.loss_function.weighted_probability_based_squared_error import (
    WeightedProbabilityBasedSquaredError,
    WeightedProbabilityBasedSquaredErrorOption,
)
from quara.loss_function.weighted_relative_entropy import (
    WeightedRelativeEntropy,
    WeightedRelativeEntropyOption,
)
from quara.minimization_algorithm.projected_gradient_descent_backtracking import (
    ProjectedGradientDescentBacktracking,
    ProjectedGradientDescentBacktrackingOption,
)
from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
)
from quara.protocol.qtomography.standard.projected_linear_estimator import (
    ProjectedLinearEstimator,
)
from quara.simulation.standard_qtomography_simulation import NoiseSetting, TestSetting
from quara.simulation.standard_qtomography_simulation_flow import (
    execute_simulation_test_settings,
)


def generate_common_setting():
    # Generate settings for simulation
    case_names = [
        "Linear (True)",
        "Linear (False)",
        "ProjectedLinear (True)",
        "ProjectedLinear (False)",
        "Maximum-Likelihood (True)",
        "Maximum-Likelihood (False)",
        "Least Squares (True)",
        "Least Squares (False)",
    ]

    parametrizations = [True, False, True, False, True, False, True, False]

    estimators = [
        LinearEstimator(),
        LinearEstimator(),
        ProjectedLinearEstimator(),
        ProjectedLinearEstimator(),
        LossMinimizationEstimator(),
        LossMinimizationEstimator(),
        LossMinimizationEstimator(),
        LossMinimizationEstimator(),
    ]

    loss_list = [
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (WeightedRelativeEntropy(), WeightedRelativeEntropyOption("identity")),
        (WeightedRelativeEntropy(), WeightedRelativeEntropyOption("identity")),
        (
            WeightedProbabilityBasedSquaredError(),
            WeightedProbabilityBasedSquaredErrorOption("identity"),
        ),
        (
            WeightedProbabilityBasedSquaredError(),
            WeightedProbabilityBasedSquaredErrorOption("identity"),
        ),
    ]

    def generate_pgdb_algo_option():
        return ProjectedGradientDescentBacktrackingOption(
            mode_stopping_criterion_gradient_descent="sum_absolute_difference_variable",
            num_history_stopping_criterion_gradient_descent=1,
        )

    algo_list = [
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (ProjectedGradientDescentBacktracking(), generate_pgdb_algo_option()),
        (ProjectedGradientDescentBacktracking(), generate_pgdb_algo_option()),
        (ProjectedGradientDescentBacktracking(), generate_pgdb_algo_option()),
        (ProjectedGradientDescentBacktracking(), generate_pgdb_algo_option()),
    ]

    return case_names, parametrizations, estimators, loss_list, algo_list


def execute(
    mode: str,
    n_qubit: int,
    true_objects: List[str],
    tester_names: List[Tuple[str, str]],
    noise_method: str,
    noise_para: dict,
    n_sample: int,
    n_rep: int,
    num_data: List[int],
    seed: int,
    output_root_dir: str,
):

    c_sys = generate_composite_system(mode, n_qubit)
    (
        case_names,
        parametrizations,
        estimators,
        loss_list,
        algo_list,
    ) = generate_common_setting()

    test_settings = []
    for true_object in true_objects:
        # Generate TestSetting 0: random_effective_lindbladian
        # True Object
        true_object_noise_setting = NoiseSetting(
            qoperation_base=("state", true_object),
            method=noise_method,
            para=noise_para,
        )

        # Tester Object
        tester_object_noise_settings = [
            NoiseSetting(
                qoperation_base=name,
                method=noise_method,
                para=noise_para,
            )
            for name in tester_names
        ]

        # Test Setting
        test_setting = TestSetting(
            true_object=true_object_noise_setting,
            tester_objects=tester_object_noise_settings,
            seed=seed,
            n_sample=n_sample,
            n_rep=n_rep,
            num_data=num_data,
            schedules="all",
            case_names=case_names,
            estimators=estimators,
            algo_list=algo_list,
            loss_list=loss_list,
            parametrizations=parametrizations,
            c_sys=c_sys,
        )
        test_settings.append(test_setting)

    all_results = execute_simulation_test_settings(test_settings, output_root_dir)
    return all_results


def execute_qst_1qubit():
    setting = {
        "mode": "qubit",
        "n_qubit": 1,
        "true_objects": ["z0", "z1", "x0", "a"],
        "tester_names": [("povm", name) for name in ["x", "y", "z"]],
        "noise_method": "random_effective_lindbladian",
        "noise_para": {
            "lindbladian_base": "identity",
            "strength_h_part": 0.1,
            "strength_k_part": 0.1,
        },
        # "noise_method": "depolarized",
        # "noise_para": {
        #    "error_rate": 0.1,
        # },
        "n_sample": 1,
        "n_rep": 1,
        "num_data": [1000, 10000],
        "seed": 777,
        "output_root_dir": "result_random_qst_1qubit",
    }
    execute(**setting)
