import datetime
from itertools import product
from typing import List, Tuple

import numpy as np
import numpy.testing as npt

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
from quara.simulation.standard_qtomography_simulation import (
    EstimatorTestSetting,
    NoiseSetting,
)
from quara.simulation.standard_qtomography_simulation_flow import (
    execute_simulation_test_settings,
)

output_root_dir_prefix = ""


def get_current_time_string():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")


def generate_common_setting():
    # Generate settings for simulation
    case_names = [
        "Linear (True)",
        # "Linear (False)",
        "ProjectedLinear (True)",
        # "ProjectedLinear (False)",
        "Maximum-Likelihood (True)",
        # "Maximum-Likelihood (False)",
        "Least Squares (True)",
        # "Least Squares (False)",
    ]

    parametrizations = [
        True,
        # False,
        True,
        # False,
        True,
        # False,
        True,
        # False
    ]

    estimators = [
        LinearEstimator(),
        # LinearEstimator(),
        ProjectedLinearEstimator(),
        # ProjectedLinearEstimator(),
        LossMinimizationEstimator(),
        # LossMinimizationEstimator(),
        LossMinimizationEstimator(),
        # LossMinimizationEstimator(),
    ]

    loss_list = [
        (None, None),
        # (None, None),
        (None, None),
        # (None, None),
        (WeightedRelativeEntropy(), WeightedRelativeEntropyOption("identity")),
        # (WeightedRelativeEntropy(), WeightedRelativeEntropyOption("identity")),
        (
            WeightedProbabilityBasedSquaredError(),
            WeightedProbabilityBasedSquaredErrorOption("identity"),
        ),
        # (
        #    WeightedProbabilityBasedSquaredError(),
        #    WeightedProbabilityBasedSquaredErrorOption("identity"),
        # ),
    ]

    def generate_pgdb_algo_option():
        return ProjectedGradientDescentBacktrackingOption(
            mode_stopping_criterion_gradient_descent="sum_absolute_difference_variable",
            num_history_stopping_criterion_gradient_descent=1,
            eps=1e-9,
        )

    algo_list = [
        (None, None),
        # (None, None),
        (None, None),
        # (None, None),
        (ProjectedGradientDescentBacktracking(), generate_pgdb_algo_option()),
        # (ProjectedGradientDescentBacktracking(), generate_pgdb_algo_option()),
        (ProjectedGradientDescentBacktracking(), generate_pgdb_algo_option()),
        # (ProjectedGradientDescentBacktracking(), generate_pgdb_algo_option()),
    ]

    eps_proj_physical_list = [1e-5] * len(case_names)

    return (
        case_names,
        parametrizations,
        estimators,
        loss_list,
        algo_list,
        eps_proj_physical_list,
    )


def execute(
    mode: str,
    n_qubit: int,
    tomography_type: str,
    true_objects: List[str],
    tester_names: List[Tuple[str, str]],
    noise_method: str,
    noise_para: dict,
    n_sample: int,
    n_rep: int,
    num_data: List[int],
    seed_qoperation: int,
    seed_data: int,
    output_root_dir: str,
    true_object_ids: List[int] = None,
    tester_ids: List[int] = None,
):
    c_sys = generate_composite_system(mode, n_qubit)
    (
        case_names,
        parametrizations,
        estimators,
        loss_list,
        algo_list,
        eps_proj_physical_list,
    ) = generate_common_setting()

    test_settings = []
    for index, true_object in enumerate(true_objects):
        # Generate EstimatorTestSetting 0: random_effective_lindbladian
        # True Object
        true_object_noise_setting = NoiseSetting(
            qoperation_base=(tomography_type, true_object),
            method=noise_method,
            para=noise_para,
            ids=true_object_ids[index] if true_object_ids else None,
        )

        # Tester Object
        tester_object_noise_settings = [
            NoiseSetting(
                qoperation_base=name,
                method=noise_method,
                para=noise_para,
                ids=tester_ids[index] if tester_ids else None,
            )
            for name in tester_names
        ]

        # Test Setting
        test_setting = EstimatorTestSetting(
            true_object=true_object_noise_setting,
            tester_objects=tester_object_noise_settings,
            seed_qoperation=seed_qoperation,
            seed_data=seed_data,
            n_sample=n_sample,
            n_rep=n_rep,
            num_data=num_data,
            schedules="all",
            case_names=case_names,
            estimators=estimators,
            eps_proj_physical_list=eps_proj_physical_list,
            algo_list=algo_list,
            loss_list=loss_list,
            parametrizations=parametrizations,
            c_sys=c_sys,
        )
        test_settings.append(test_setting)

    all_results = execute_simulation_test_settings(
        test_settings, output_root_dir, pdf_mode="none"
    )
    return all_results


def execute_qst_1qubit():
    setting = {
        "mode": "qubit",
        "n_qubit": 1,
        "tomography_type": "state",
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
        "seed_qoperation": 888,
        "seed_data": 777,
        "output_root_dir": output_root_dir_prefix
        + "result_random_qst_1qubit-"
        + get_current_time_string(),
    }
    execute(**setting)


def execute_qst_2qubit():
    setting = {
        "mode": "qubit",
        "n_qubit": 2,
        "tomography_type": "state",
        "true_objects": ["z0_z0", "z0_x0", "z0_a", "bell_psi_minus"],
        "tester_names": [
            ("povm", f"{a}_{b}") for a, b in product(["x", "y", "z"], repeat=2)
        ],
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
        "seed_qoperation": 888,
        "seed_data": 777,
        "output_root_dir": output_root_dir_prefix
        + "result_random_qst_2qubit-"
        + get_current_time_string(),
    }
    execute(**setting)


def execute_qst_3qubit():
    setting = {
        "mode": "qubit",
        "n_qubit": 3,
        "tomography_type": "state",
        "true_objects": ["z0_z0_z0", "ghz", "werner"],
        "tester_names": [
            ("povm", f"{a}_{b}_{c}") for a, b, c in product(["x", "y", "z"], repeat=3)
        ],
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
        "seed_qoperation": 888,
        "seed_data": 777,
        "output_root_dir": output_root_dir_prefix
        + "result_random_qst_3qubit-"
        + get_current_time_string(),
    }
    execute(**setting)


def execute_qst_1qutrit():
    setting = {
        "mode": "qutrit",
        "n_qubit": 1,
        "tomography_type": "state",
        "true_objects": ["01z0", "02z1", "0_1_2_superposition"],
        "tester_names": [
            ("povm", name)
            for name in ["01x3", "01y3", "z3", "12x3", "12y3", "02x3", "02y3"]
        ],
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
        "seed_qoperation": 888,
        "seed_data": 777,
        "output_root_dir": output_root_dir_prefix
        + "result_random_qst_1qutrit-"
        + get_current_time_string(),
    }
    execute(**setting)


def execute_qst_2qutrit():
    setting = {
        "mode": "qutrit",
        "n_qubit": 2,
        "tomography_type": "state",
        "true_objects": ["01z0_01z0", "00_11_22_superposition"],
        "tester_names": [
            ("povm", f"{a}_{b}")
            for a, b in product(
                ["01x3", "01y3", "z3", "12x3", "12y3", "02x3", "02y3"], repeat=2
            )
        ],
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
        "seed_qoperation": 888,
        "seed_data": 777,
        "output_root_dir": output_root_dir_prefix
        + "result_random_qst_2qutrit-"
        + get_current_time_string(),
    }
    execute(**setting)


def execute_povmt_1qubit():
    setting = {
        "mode": "qubit",
        "n_qubit": 1,
        "tomography_type": "povm",
        "true_objects": ["z", "x"],
        "tester_names": [("state", name) for name in ["x0", "y0", "z0", "z1"]],
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
        "seed_qoperation": 888,
        "seed_data": 777,
        "output_root_dir": output_root_dir_prefix
        + "result_random_povmt_1qubit-"
        + get_current_time_string(),
    }
    execute(**setting)


def execute_povmt_2qubit():
    setting = {
        "mode": "qubit",
        "n_qubit": 2,
        "tomography_type": "povm",
        "true_objects": ["z_z", "bell"],
        "tester_names": [
            ("state", f"{a}_{b}")
            for a, b in product(["x0", "y0", "z0", "z1"], repeat=2)
        ],
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
        "seed_qoperation": 888,
        "seed_data": 777,
        "output_root_dir": output_root_dir_prefix
        + "result_random_povmt_2qubit-"
        + get_current_time_string(),
    }
    execute(**setting)


def execute_povmt_3qubit():
    setting = {
        "mode": "qubit",
        "n_qubit": 3,
        "tomography_type": "povm",
        "true_objects": ["z_z_z"],
        "tester_names": [
            ("state", f"{a}_{b}_{c}")
            for a, b, c in product(["x0", "y0", "z0", "z1"], repeat=3)
        ],
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
        "seed_qoperation": 888,
        "seed_data": 777,
        "output_root_dir": output_root_dir_prefix
        + "result_random_povmt_3qubit-"
        + get_current_time_string(),
    }
    execute(**setting)


def execute_povmt_1qutrit():
    setting = {
        "mode": "qutrit",
        "n_qubit": 1,
        "tomography_type": "povm",
        "true_objects": ["z3", "z2"],
        "tester_names": [
            ("state", name)
            for name in [
                "01z0",
                "12z0",
                "02z1",
                "01x0",
                "01y0",
                "12x0",
                "12y0",
                "02x0",
                "02y0",
            ]
        ],
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
        "seed_qoperation": 888,
        "seed_data": 777,
        "output_root_dir": output_root_dir_prefix
        + "result_random_povmt_1qutrit-"
        + get_current_time_string(),
    }
    execute(**setting)


def execute_povmt_2qutrit():
    setting = {
        "mode": "qutrit",
        "n_qubit": 2,
        "tomography_type": "povm",
        "true_objects": ["z3_z3", "z2_z2"],
        "tester_names": [
            ("state", f"{a}_{b}")
            for a, b in product(
                [
                    "01z0",
                    "12z0",
                    "02z1",
                    "01x0",
                    "01y0",
                    "12x0",
                    "12y0",
                    "02x0",
                    "02y0",
                ],
                repeat=2,
            )
        ],
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
        "seed_qoperation": 888,
        "seed_data": 777,
        "output_root_dir": output_root_dir_prefix
        + "result_random_povmt_2qutrit-"
        + get_current_time_string(),
    }
    execute(**setting)


def execute_qpt_1qubit():
    setting = {
        "mode": "qubit",
        "n_qubit": 1,
        "tomography_type": "gate",
        "true_objects": [
            "identity",
            "x90",
            "y90",
            "z90",
            "x180",
            "y180",
            "z180",
            "hadamard",
            "phase",
            "phase_daggered",
            "piover8",
            "piover8_daggered",
        ],
        "tester_names": [("state", name) for name in ["x0", "y0", "z0", "z1"]]
        + [("povm", name) for name in ["x", "y", "z"]],
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
        "seed_qoperation": 888,
        "seed_data": 777,
        "output_root_dir": output_root_dir_prefix
        + "result_random_qpt_1qubit-"
        + get_current_time_string(),
    }
    execute(**setting)


def execute_qpt_2qubit():
    setting = {
        "mode": "qubit",
        "n_qubit": 2,
        "tomography_type": "gate",
        "true_objects": ["identity", "zx90"],
        "true_object_ids": [None, [0, 1]],
        "tester_names": [
            ("state", f"{a}_{b}")
            for a, b in product(["x0", "y0", "z0", "z1"], repeat=2)
        ]
        + [("povm", f"{a}_{b}") for a, b in product(["x", "y", "z"], repeat=2)],
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
        "seed_qoperation": 888,
        "seed_data": 777,
        "output_root_dir": output_root_dir_prefix
        + "result_random_qpt_2qubit-"
        + get_current_time_string(),
    }
    execute(**setting)


def execute_qpt_3qubit():
    setting = {
        "mode": "qubit",
        "n_qubit": 3,
        "tomography_type": "gate",
        "true_objects": ["identity", "toffoli", "fredkin"],
        "true_object_ids": [None, [0, 1, 2], [0, 1, 2]],
        "tester_names": [
            ("state", f"{a}_{b}_{c}")
            for a, b, c in product(["x0", "y0", "z0", "z1"], repeat=3)
        ]
        + [("povm", f"{a}_{b}_{c}") for a, b, c in product(["x", "y", "z"], repeat=3)],
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
        "seed_qoperation": 888,
        "seed_data": 777,
        "output_root_dir": output_root_dir_prefix
        + "result_random_qpt_3qubit-"
        + get_current_time_string(),
    }
    execute(**setting)


def execute_qpt_1qutrit():
    setting = {
        "mode": "qutrit",
        "n_qubit": 1,
        "tomography_type": "gate",
        "true_objects": ["identity"],
        "tester_names": [
            ("state", name)
            for name in [
                "01z0",
                "12z0",
                "02z1",
                "01x0",
                "01y0",
                "12x0",
                "12y0",
                "02x0",
                "02y0",
            ]
        ]
        + [
            ("povm", name)
            for name in ["01x3", "01y3", "z3", "12x3", "12y3", "02x3", "02y3"]
        ],
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
        "seed_qoperation": 888,
        "seed_data": 777,
        "output_root_dir": output_root_dir_prefix
        + "result_random_qpt_1qutrit-"
        + get_current_time_string(),
    }
    execute(**setting)


def execute_qpt_2qutrit():
    setting = {
        "mode": "qutrit",
        "n_qubit": 2,
        "tomography_type": "gate",
        "true_objects": ["identity"],
        "tester_names": [
            ("state", f"{a}_{b}")
            for a, b in product(
                [
                    "01z0",
                    "12z0",
                    "02z1",
                    "01x0",
                    "01y0",
                    "12x0",
                    "12y0",
                    "02x0",
                    "02y0",
                ],
                repeat=2,
            )
        ]
        + [
            ("povm", f"{a}_{b}")
            for a, b in product(
                ["01x3", "01y3", "z3", "12x3", "12y3", "02x3", "02y3"], repeat=2
            )
        ],
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
        "seed_qoperation": 888,
        "seed_data": 777,
        "output_root_dir": output_root_dir_prefix
        + "result_random_qpt_2qutrit-"
        + get_current_time_string(),
    }
    execute(**setting)
