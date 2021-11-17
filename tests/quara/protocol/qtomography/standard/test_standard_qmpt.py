import itertools

import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system import CompositeSystem
from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.mprocess import MProcess
from quara.objects.povm import (
    get_x_povm,
    get_y_povm,
    get_z_povm,
)
from quara.objects.qoperation_typical import (
    generate_qoperation,
    generate_qoperation_object,
)
from quara.objects.tester_typical import (
    generate_tester_states,
    generate_tester_povms,
)
from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
from quara.protocol.qtomography.standard.standard_qmpt import (
    cqpt_to_cqmpt,
    StandardQmpt,
)
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
)
from quara.loss_function.standard_qtomography_based_weighted_relative_entropy import (
    StandardQTomographyBasedWeightedRelativeEntropy,
    StandardQTomographyBasedWeightedRelativeEntropyOption,
)
from quara.loss_function.standard_qtomography_based_weighted_probability_based_squared_error import (
    StandardQTomographyBasedWeightedProbabilityBasedSquaredError,
    StandardQTomographyBasedWeightedProbabilityBasedSquaredErrorOption,
)
from quara.minimization_algorithm.projected_gradient_descent_backtracking import (
    ProjectedGradientDescentBacktracking,
    ProjectedGradientDescentBacktrackingOption,
)


class TestStandardQmpt:
    def test_testers(self):
        # Arrange
        num_qubits = 1
        c_sys = generate_composite_system(mode="qubit", num=num_qubits)

        # Tester Objects
        state_names = ["x0", "y0", "z0", "z1"]
        povm_names = ["x", "y", "z"]

        tester_states = [
            generate_qoperation_object(
                mode="state", object_name="state", name=name, c_sys=c_sys
            )
            for name in state_names
        ]
        tester_povms = [
            generate_qoperation_object(
                mode="povm", object_name="povm", name=name, c_sys=c_sys
            )
            for name in povm_names
        ]

        # True Object
        true_object = generate_qoperation(mode="mprocess", name="x-type1", c_sys=c_sys)

        # Qmpt
        qmpt = StandardQmpt(
            states=tester_states,
            povms=tester_povms,
            num_outcomes=true_object.num_outcomes,
            on_para_eq_constraint=True,
            schedules="all",
        )

        assert len(qmpt.testers) == 7

    def test_is_valid_experiment(self):
        # Arrange
        num_qubits = 1
        c_sys = generate_composite_system(mode="qubit", num=num_qubits)

        # Tester Objects
        state_names = ["x0", "y0", "z0", "z1"]
        povm_names = ["x", "y", "z"]

        tester_states = [
            generate_qoperation_object(
                mode="state", object_name="state", name=name, c_sys=c_sys
            )
            for name in state_names
        ]
        tester_povms = [
            generate_qoperation_object(
                mode="povm", object_name="povm", name=name, c_sys=c_sys
            )
            for name in povm_names
        ]

        # True Object
        true_object = generate_qoperation(mode="mprocess", name="x-type1", c_sys=c_sys)

        # Qmpt
        qmpt = StandardQmpt(
            states=tester_states,
            povms=tester_povms,
            num_outcomes=true_object.num_outcomes,
            on_para_eq_constraint=True,
            schedules="all",
        )

        # is_valid_experiment == True
        assert qmpt.is_valid_experiment() == True

        # is_valid_experiment == False
        e_sys0 = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys0 = CompositeSystem([e_sys0])
        e_sys1 = ElementalSystem(1, get_normalized_pauli_basis())
        c_sys1 = CompositeSystem([e_sys1])

        povm_x = get_x_povm(c_sys1)
        povm_y = get_y_povm(c_sys0)
        povm_z = get_z_povm(c_sys0)
        povms = [povm_x, povm_y, povm_z]

        qmpt.experiment.povms = povms
        assert qmpt.is_valid_experiment() == False


def test_cqpt_to_cqmpt():
    # Case 1: on_para_eq_constraint=False
    # Arrange
    c_qpt = np.array(list(range(1, 17)))
    dim = 2
    m = 3

    # Act
    actual_a_qmpt, actual_b_qmpt = cqpt_to_cqmpt(
        c_qpt=c_qpt, dim=dim, m_mprocess=m, on_para_eq_constraint=False
    )

    # Assert
    expected_a_qmpt = np.array(
        [
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
            ],
        ]
    )
    npt.assert_almost_equal(actual_a_qmpt, expected_a_qmpt, decimal=15)

    expected_b_qmpt = np.array([0, 0, 0.0])
    npt.assert_almost_equal(actual_b_qmpt, expected_b_qmpt, decimal=15)

    # Case 2: on_para_eq_constraint=True
    # Act
    actual_a_qmpt, actual_b_qmpt = cqpt_to_cqmpt(
        c_qpt=c_qpt, dim=dim, m_mprocess=m, on_para_eq_constraint=True
    )
    # Assert
    expected_a_qmpt = np.array(
        [
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
            ],
            [
                -1,
                -2,
                -3,
                -4,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -1,
                -2,
                -3,
                -4,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16.0,
            ],
        ]
    )
    npt.assert_almost_equal(actual_a_qmpt, expected_a_qmpt, decimal=15)
    expected_b_qmpt = np.array([0, 0, 1.0])
    npt.assert_almost_equal(actual_b_qmpt, expected_b_qmpt, decimal=15)


def test_set_coeffs():
    c_sys = generate_composite_system(mode="qubit", num=1, ids_esys=[1])

    # Tester Objects
    tester_states = [
        generate_qoperation_object(
            mode="state", object_name="state", name=name, c_sys=c_sys
        )
        for name in ["x0", "y0", "z0", "z1"]
    ]
    tester_povms = [
        generate_qoperation_object(
            mode="povm", object_name="povm", name=name, c_sys=c_sys
        )
        for name in ["x", "y", "z"]
    ]

    # Case 1: on_para_eq_constarint = True
    on_para_eq_constraint = True
    num_outcomes = 2
    actual = StandardQmpt(
        tester_states,
        tester_povms,
        num_outcomes=num_outcomes,
        on_para_eq_constraint=on_para_eq_constraint,
        seed_data=7,
    )

    # Assert
    assert actual.calc_matA().shape == (48, 28)
    assert actual.calc_vecB().shape == (48,)

    # Case 1: on_para_eq_constarint = False
    on_para_eq_constraint = False
    num_outcomes = 2
    actual = StandardQmpt(
        tester_states,
        tester_povms,
        num_outcomes=num_outcomes,
        on_para_eq_constraint=on_para_eq_constraint,
        seed_data=7,
    )

    # Assert
    assert actual.calc_matA().shape == (48, 32)
    assert actual.calc_vecB().shape == (48,)


def calc_prob_dist_with_experiment(source_qmpt, qope):
    tmp_experiment = source_qmpt._experiment.copy()
    for schedule_index in range(len(tmp_experiment.schedules)):
        target_index = source_qmpt._get_target_index(tmp_experiment, schedule_index)
        tmp_experiment.mprocesses[target_index] = qope
    return tmp_experiment.calc_prob_dists()


@pytest.mark.parametrize(("on_para_eq_constraint"), [(True), (False)])
def test_compare_prob_dist_1qubit(on_para_eq_constraint: bool):
    # Arrange
    c_sys = generate_composite_system(mode="qubit", num=1, ids_esys=[1])

    # Tester Objects
    state_names = ["x0", "y0", "z0", "z1"]
    povm_names = ["x", "y", "z"]

    tester_states = [
        generate_qoperation_object(
            mode="state", object_name="state", name=name, c_sys=c_sys
        )
        for name in state_names
    ]
    tester_povms = [
        generate_qoperation_object(
            mode="povm", object_name="povm", name=name, c_sys=c_sys
        )
        for name in povm_names
    ]

    # Qmpt
    num_outcomes = 2
    qmpt = StandardQmpt(
        tester_states,
        tester_povms,
        num_outcomes=num_outcomes,
        on_para_eq_constraint=on_para_eq_constraint,
        seed_data=7,
    )

    # TrueObject
    true_object_name = "x-type1"
    true_object = generate_qoperation_object(
        mode="mprocess", object_name="mprocess", name=true_object_name, c_sys=c_sys
    )
    if on_para_eq_constraint is False:
        true_object = MProcess(
            hss=true_object.hss, on_para_eq_constraint=False, c_sys=c_sys
        )

    # Act
    actual_list = qmpt.calc_prob_dists(true_object)

    # Assert
    expected_list = calc_prob_dist_with_experiment(qmpt, true_object)

    for actual, expected in zip(actual_list, expected_list):
        npt.assert_almost_equal(actual, expected, decimal=15)


@pytest.mark.qmpt_twoqubit
@pytest.mark.parametrize(("on_para_eq_constraint"), [(True), (False)])
def test_compare_prob_dist_2qubit(on_para_eq_constraint: bool):
    c_sys = generate_composite_system(mode="qubit", num=2, ids_esys=[1, 2])

    # Tester Objects
    state_names = ["x0", "y0", "z0", "z1"]
    povm_names = ["x", "y", "z"]

    tester_states = [
        generate_qoperation_object(
            mode="state", object_name="state", name=f"{a}_{b}", c_sys=c_sys
        )
        for a, b in itertools.product(state_names, state_names)
    ]
    tester_povms = [
        generate_qoperation_object(
            mode="povm", object_name="povm", name=f"{a}_{b}", c_sys=c_sys
        )
        for a, b in itertools.product(povm_names, povm_names)
    ]

    # True Object
    true_object_name = "x-type1_x-type1"
    true_object = generate_qoperation_object(
        mode="mprocess", object_name="mprocess", name=true_object_name, c_sys=c_sys
    )
    if on_para_eq_constraint is False:
        true_object = MProcess(
            hss=true_object.hss, on_para_eq_constraint=False, c_sys=c_sys
        )

    # StandardQmpt
    num_outcomes = true_object.num_outcomes  # 4
    qmpt = StandardQmpt(
        tester_states,
        tester_povms,
        num_outcomes=num_outcomes,
        on_para_eq_constraint=on_para_eq_constraint,
        seed_data=7,
    )

    # Act
    actual_list = qmpt.calc_prob_dists(true_object)

    # Assert
    expected_list = calc_prob_dist_with_experiment(qmpt, true_object)

    for actual, expected in zip(actual_list, expected_list):
        # If decimal is set to 15, the test will fail.
        npt.assert_almost_equal(actual, expected, decimal=14)


@pytest.mark.qmpt_onequtrit
@pytest.mark.parametrize(("on_para_eq_constraint"), [(True), (False)])
def test_compare_prob_dist_1qutrit(on_para_eq_constraint: bool):
    c_sys = generate_composite_system(mode="qutrit", num=1, ids_esys=[1])

    # Tester Objects
    state_names = [
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
    povm_names = ["01x3", "01y3", "z3", "12x3", "12y3", "02x3", "02y3"]

    tester_states = [
        generate_qoperation_object(
            mode="state", object_name="state", name=name, c_sys=c_sys
        )
        for name in state_names
    ]
    tester_povms = [
        generate_qoperation_object(
            mode="povm", object_name="povm", name=name, c_sys=c_sys
        )
        for name in povm_names
    ]

    num_outcomes = 3
    qmpt = StandardQmpt(
        tester_states,
        tester_povms,
        num_outcomes=num_outcomes,
        on_para_eq_constraint=on_para_eq_constraint,
        seed_data=7,
    )

    # True Object
    true_object_name = "z3-type1"
    true_object = generate_qoperation_object(
        mode="mprocess", object_name="mprocess", name=true_object_name, c_sys=c_sys
    )
    if on_para_eq_constraint is False:
        true_object = MProcess(
            hss=true_object.hss, on_para_eq_constraint=False, c_sys=c_sys
        )
    # Act
    actual_list = qmpt.calc_prob_dists(true_object)

    # Assert
    expected_list = calc_prob_dist_with_experiment(qmpt, true_object)
    for actual, expected in zip(actual_list, expected_list):
        npt.assert_almost_equal(actual, expected, decimal=15)


@pytest.mark.parametrize(
    ("true_object_name", "on_para_eq_constraint"),
    [("z-type1", True), ("z-type1", False)],
)
def test_calc_estimate_LinearEstimator_1qubit(
    true_object_name: str, on_para_eq_constraint: bool
):
    # Arrange
    num_qubits = 1
    c_sys = generate_composite_system(mode="qubit", num=num_qubits)

    # Tester Objects
    state_names = ["x0", "y0", "z0", "z1"]
    povm_names = ["x", "y", "z"]

    tester_states = [
        generate_qoperation_object(
            mode="state", object_name="state", name=name, c_sys=c_sys
        )
        for name in state_names
    ]
    tester_povms = [
        generate_qoperation_object(
            mode="povm", object_name="povm", name=name, c_sys=c_sys
        )
        for name in povm_names
    ]

    # True Object
    true_object = generate_qoperation(
        mode="mprocess", name=true_object_name, c_sys=c_sys
    )
    if on_para_eq_constraint is False:
        true_object = MProcess(
            hss=true_object.hss, on_para_eq_constraint=False, c_sys=c_sys
        )

    # Qmpt
    qmpt = StandardQmpt(
        states=tester_states,
        povms=tester_povms,
        num_outcomes=true_object.num_outcomes,
        on_para_eq_constraint=on_para_eq_constraint,
        schedules="all",
    )

    # empi_dists
    prob_dists = qmpt.calc_prob_dists(true_object)
    empi_dists = [(10, prob_dist) for prob_dist in prob_dists]

    # Estimator
    estimator = LinearEstimator()

    # Act
    result = estimator.calc_estimate(
        qtomography=qmpt, empi_dists=empi_dists, is_computation_time_required=True
    )
    actual = result.estimated_qoperation

    # Assert
    for a, e in zip(actual.hss, true_object.hss):
        npt.assert_almost_equal(a, e, decimal=15)


@pytest.mark.qmpt_twoqubit
@pytest.mark.parametrize(
    ("true_object_name", "on_para_eq_constraint"),
    [
        ("x-type1_x-type1", True),
        ("x-type1_x-type1", False),
        ("bell-type1", True),
        ("bell-type1", False),
    ],
)
def test_calc_estimate_LinearEstimator_2qubit(
    true_object_name: str, on_para_eq_constraint: bool
):
    # Arrange
    num_qubits = 2
    c_sys = generate_composite_system(mode="qubit", num=num_qubits)

    # Tester Objects
    state_names = ["x0", "y0", "z0", "z1"]
    povm_names = ["x", "y", "z"]

    tester_states = [
        generate_qoperation_object(
            mode="state", object_name="state", name=f"{a}_{b}", c_sys=c_sys
        )
        for a, b in itertools.product(state_names, repeat=num_qubits)
    ]
    tester_povms = [
        generate_qoperation_object(
            mode="povm", object_name="povm", name=f"{a}_{b}", c_sys=c_sys
        )
        for a, b in itertools.product(povm_names, repeat=num_qubits)
    ]

    # True Object
    true_object = generate_qoperation(
        mode="mprocess", name=true_object_name, c_sys=c_sys
    )
    if on_para_eq_constraint is False:
        true_object = MProcess(
            hss=true_object.hss, on_para_eq_constraint=False, c_sys=c_sys
        )

    # Qmpt
    qmpt = StandardQmpt(
        states=tester_states,
        povms=tester_povms,
        num_outcomes=true_object.num_outcomes,
        on_para_eq_constraint=on_para_eq_constraint,
        schedules="all",
    )

    # empi_dists
    prob_dists = qmpt.calc_prob_dists(true_object)
    empi_dists = [(10, prob_dist) for prob_dist in prob_dists]

    # Estimator
    estimator = LinearEstimator()

    # Act
    result = estimator.calc_estimate(
        qtomography=qmpt, empi_dists=empi_dists, is_computation_time_required=True
    )
    actual = result.estimated_qoperation

    # Assert
    for a, e in zip(actual.hss, true_object.hss):
        npt.assert_almost_equal(a, e, decimal=14)


@pytest.mark.qmpt_onequtrit
@pytest.mark.parametrize(
    ("true_object_name", "on_para_eq_constraint"),
    [("z3-type1", True), ("z3-type1", False), ("z2-type1", True), ("z2-type1", False)],
)
def test_calc_estimate_LinearEstimator_1qutrit(
    true_object_name: str, on_para_eq_constraint: bool
):
    # Arrange
    num_qubits = 1
    c_sys = generate_composite_system(mode="qutrit", num=num_qubits)

    # Tester Objects
    state_names = [
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
    povm_names = ["01x3", "01y3", "z3", "12x3", "12y3", "02x3", "02y3"]

    tester_states = [
        generate_qoperation_object(
            mode="state", object_name="state", name=name, c_sys=c_sys
        )
        for name in state_names
    ]
    tester_povms = [
        generate_qoperation_object(
            mode="povm", object_name="povm", name=name, c_sys=c_sys
        )
        for name in povm_names
    ]

    # True Object
    true_object = generate_qoperation(
        mode="mprocess", name=true_object_name, c_sys=c_sys
    )
    if on_para_eq_constraint is False:
        true_object = MProcess(
            hss=true_object.hss, on_para_eq_constraint=False, c_sys=c_sys
        )

    # Qmpt
    qmpt = StandardQmpt(
        states=tester_states,
        povms=tester_povms,
        num_outcomes=true_object.num_outcomes,
        on_para_eq_constraint=on_para_eq_constraint,
        schedules="all",
    )

    # empi_dists
    prob_dists = qmpt.calc_prob_dists(true_object)
    empi_dists = [(10, prob_dist) for prob_dist in prob_dists]

    # Estimator
    estimator = LinearEstimator()

    # Act
    result = estimator.calc_estimate(
        qtomography=qmpt, empi_dists=empi_dists, is_computation_time_required=True
    )
    actual = result.estimated_qoperation

    # Assert
    for a, e in zip(actual.hss, true_object.hss):
        npt.assert_almost_equal(a, e, decimal=14)


@pytest.mark.parametrize(
    ("true_object_name", "on_para_eq_constraint"),
    [("z-type1", True), ("z-type1", False)],
)
def test_calc_estimate_MLE_1qubit(true_object_name: str, on_para_eq_constraint: bool):
    # Arrange
    num_qubits = 1
    c_sys = generate_composite_system(mode="qubit", num=num_qubits)

    # Tester Objects
    state_names = ["x0", "y0", "z0", "z1"]
    povm_names = ["x", "y", "z"]

    tester_states = [
        generate_qoperation_object(
            mode="state", object_name="state", name=name, c_sys=c_sys
        )
        for name in state_names
    ]
    tester_povms = [
        generate_qoperation_object(
            mode="povm", object_name="povm", name=name, c_sys=c_sys
        )
        for name in povm_names
    ]

    # True Object
    true_object = generate_qoperation(
        mode="mprocess", name=true_object_name, c_sys=c_sys
    )
    if on_para_eq_constraint is False:
        true_object = MProcess(
            hss=true_object.hss, on_para_eq_constraint=False, c_sys=c_sys
        )

    # Qmpt
    qmpt = StandardQmpt(
        states=tester_states,
        povms=tester_povms,
        num_outcomes=true_object.num_outcomes,
        on_para_eq_constraint=on_para_eq_constraint,
        # eps_proj_physical=1e-5,
        schedules="all",
    )

    # empi_dists
    prob_dists = qmpt.calc_prob_dists(true_object)
    empi_dists = [(10, prob_dist) for prob_dist in prob_dists]

    # Estimator
    estimator = LossMinimizationEstimator()
    loss = StandardQTomographyBasedWeightedRelativeEntropy()
    loss_option = StandardQTomographyBasedWeightedRelativeEntropyOption("identity")
    algo = ProjectedGradientDescentBacktracking()
    algo_option = ProjectedGradientDescentBacktrackingOption(
        mode_stopping_criterion_gradient_descent="sum_absolute_difference_variable",
        num_history_stopping_criterion_gradient_descent=1,
        eps=1e-9,
    )

    # Act
    result = estimator.calc_estimate(
        qtomography=qmpt,
        empi_dists=empi_dists,
        loss=loss,
        loss_option=loss_option,
        algo=algo,
        algo_option=algo_option,
        is_computation_time_required=True,
    )
    actual = result.estimated_qoperation

    # Assert
    for a, e in zip(actual.hss, true_object.hss):
        npt.assert_almost_equal(a, e, decimal=6)


@pytest.mark.qmpt_twoqubit
@pytest.mark.parametrize(
    ("true_object_name", "on_para_eq_constraint"),
    [
        ("x-type1_x-type1", True),
        ("x-type1_x-type1", False),
        ("bell-type1", True),
        ("bell-type1", False),
    ],
)
def test_calc_estimate_MLE_2qubit(true_object_name: str, on_para_eq_constraint: bool):
    # Arrange
    num_qubits = 2
    c_sys = generate_composite_system(mode="qubit", num=num_qubits)

    # Tester Objects
    state_names = ["x0", "y0", "z0", "z1"]
    povm_names = ["x", "y", "z"]

    tester_states = [
        generate_qoperation_object(
            mode="state", object_name="state", name=f"{a}_{b}", c_sys=c_sys
        )
        for a, b in itertools.product(state_names, repeat=num_qubits)
    ]
    tester_povms = [
        generate_qoperation_object(
            mode="povm", object_name="povm", name=f"{a}_{b}", c_sys=c_sys
        )
        for a, b in itertools.product(povm_names, repeat=num_qubits)
    ]

    # True Object
    true_object = generate_qoperation(
        mode="mprocess", name=true_object_name, c_sys=c_sys
    )
    if on_para_eq_constraint is False:
        true_object = MProcess(
            hss=true_object.hss, on_para_eq_constraint=False, c_sys=c_sys
        )

    # Qmpt
    qmpt = StandardQmpt(
        states=tester_states,
        povms=tester_povms,
        num_outcomes=true_object.num_outcomes,
        on_para_eq_constraint=on_para_eq_constraint,
        eps_proj_physical=1e-5,
        eps_truncate_imaginary_part=1e-12,
        schedules="all",
    )

    # empi_dists
    prob_dists = qmpt.calc_prob_dists(true_object)
    empi_dists = [(10, prob_dist) for prob_dist in prob_dists]

    # Estimator
    estimator = LossMinimizationEstimator()
    loss = StandardQTomographyBasedWeightedRelativeEntropy()
    loss_option = StandardQTomographyBasedWeightedRelativeEntropyOption("identity")
    algo = ProjectedGradientDescentBacktracking()
    algo_option = ProjectedGradientDescentBacktrackingOption(
        mode_stopping_criterion_gradient_descent="sum_absolute_difference_variable",
        num_history_stopping_criterion_gradient_descent=1,
        eps=1e-9,
    )

    # Act
    result = estimator.calc_estimate(
        qtomography=qmpt,
        empi_dists=empi_dists,
        loss=loss,
        loss_option=loss_option,
        algo=algo,
        algo_option=algo_option,
        is_computation_time_required=True,
    )
    actual = result.estimated_qoperation

    # Assert
    for a, e in zip(actual.hss, true_object.hss):
        npt.assert_almost_equal(a, e, decimal=1)


@pytest.mark.qmpt_onequtrit
@pytest.mark.parametrize(
    ("true_object_name", "on_para_eq_constraint"),
    [("z3-type1", True), ("z3-type1", False), ("z2-type1", True), ("z2-type1", False)],
)
def test_calc_estimate_MLE_1qutrit(true_object_name: str, on_para_eq_constraint: bool):
    # Arrange
    num_qubits = 1
    c_sys = generate_composite_system(mode="qutrit", num=num_qubits)

    # Tester Objects
    state_names = [
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
    povm_names = ["01x3", "01y3", "z3", "12x3", "12y3", "02x3", "02y3"]

    tester_states = [
        generate_qoperation_object(
            mode="state", object_name="state", name=name, c_sys=c_sys
        )
        for name in state_names
    ]
    tester_povms = [
        generate_qoperation_object(
            mode="povm", object_name="povm", name=name, c_sys=c_sys
        )
        for name in povm_names
    ]

    # True Object
    true_object = generate_qoperation(
        mode="mprocess", name=true_object_name, c_sys=c_sys
    )
    if on_para_eq_constraint is False:
        true_object = MProcess(
            hss=true_object.hss, on_para_eq_constraint=False, c_sys=c_sys
        )

    # Qmpt
    qmpt = StandardQmpt(
        states=tester_states,
        povms=tester_povms,
        num_outcomes=true_object.num_outcomes,
        on_para_eq_constraint=on_para_eq_constraint,
        eps_proj_physical=1e-5,
        schedules="all",
    )

    # empi_dists
    prob_dists = qmpt.calc_prob_dists(true_object)
    empi_dists = [(10, prob_dist) for prob_dist in prob_dists]

    # Estimator
    estimator = LossMinimizationEstimator()
    loss = StandardQTomographyBasedWeightedRelativeEntropy()
    loss_option = StandardQTomographyBasedWeightedRelativeEntropyOption("identity")
    algo = ProjectedGradientDescentBacktracking()
    algo_option = ProjectedGradientDescentBacktrackingOption(
        mode_stopping_criterion_gradient_descent="sum_absolute_difference_variable",
        num_history_stopping_criterion_gradient_descent=1,
        eps=1e-9,
    )

    # Act
    result = estimator.calc_estimate(
        qtomography=qmpt,
        empi_dists=empi_dists,
        loss=loss,
        loss_option=loss_option,
        algo=algo,
        algo_option=algo_option,
        is_computation_time_required=True,
    )
    actual = result.estimated_qoperation

    # Assert
    for a, e in zip(actual.hss, true_object.hss):
        npt.assert_almost_equal(a, e, decimal=1)


@pytest.mark.parametrize(
    ("true_object_name", "on_para_eq_constraint"),
    [("z-type1", True), ("z-type1", False)],
)
def test_calc_estimate_LSE_1qubit(true_object_name: str, on_para_eq_constraint: bool):
    # Arrange
    num_qubits = 1
    c_sys = generate_composite_system(mode="qubit", num=num_qubits)

    # Tester Objects
    state_names = ["x0", "y0", "z0", "z1"]
    povm_names = ["x", "y", "z"]

    tester_states = [
        generate_qoperation_object(
            mode="state", object_name="state", name=name, c_sys=c_sys
        )
        for name in state_names
    ]
    tester_povms = [
        generate_qoperation_object(
            mode="povm", object_name="povm", name=name, c_sys=c_sys
        )
        for name in povm_names
    ]

    # True Object
    true_object = generate_qoperation(
        mode="mprocess", name=true_object_name, c_sys=c_sys
    )
    if on_para_eq_constraint is False:
        true_object = MProcess(
            hss=true_object.hss, on_para_eq_constraint=False, c_sys=c_sys
        )

    # Qmpt
    qmpt = StandardQmpt(
        states=tester_states,
        povms=tester_povms,
        num_outcomes=true_object.num_outcomes,
        on_para_eq_constraint=on_para_eq_constraint,
        # eps_proj_physical=1e-5,
        schedules="all",
    )

    # empi_dists
    prob_dists = qmpt.calc_prob_dists(true_object)
    empi_dists = [(10, prob_dist) for prob_dist in prob_dists]

    # Estimator
    estimator = LossMinimizationEstimator()
    loss = StandardQTomographyBasedWeightedProbabilityBasedSquaredError()
    loss_option = StandardQTomographyBasedWeightedProbabilityBasedSquaredErrorOption(
        "identity"
    )
    algo = ProjectedGradientDescentBacktracking()
    algo_option = ProjectedGradientDescentBacktrackingOption(
        mode_stopping_criterion_gradient_descent="sum_absolute_difference_variable",
        num_history_stopping_criterion_gradient_descent=1,
        eps=1e-9,
    )

    # Act
    result = estimator.calc_estimate(
        qtomography=qmpt,
        empi_dists=empi_dists,
        loss=loss,
        loss_option=loss_option,
        algo=algo,
        algo_option=algo_option,
        is_computation_time_required=True,
    )
    actual = result.estimated_qoperation

    # Assert
    for a, e in zip(actual.hss, true_object.hss):
        npt.assert_almost_equal(a, e, decimal=7)


@pytest.mark.qmpt_twoqubit
@pytest.mark.parametrize(
    ("true_object_name", "on_para_eq_constraint"),
    [
        ("x-type1_x-type1", True),
        ("x-type1_x-type1", False),
        ("bell-type1", True),
        ("bell-type1", False),
    ],
)
def test_calc_estimate_LSE_2qubit(true_object_name: str, on_para_eq_constraint: bool):
    # Arrange
    num_qubits = 2
    c_sys = generate_composite_system(mode="qubit", num=num_qubits)

    # Tester Objects
    state_names = ["x0", "y0", "z0", "z1"]
    povm_names = ["x", "y", "z"]

    tester_states = [
        generate_qoperation_object(
            mode="state", object_name="state", name=f"{a}_{b}", c_sys=c_sys
        )
        for a, b in itertools.product(state_names, repeat=num_qubits)
    ]
    tester_povms = [
        generate_qoperation_object(
            mode="povm", object_name="povm", name=f"{a}_{b}", c_sys=c_sys
        )
        for a, b in itertools.product(povm_names, repeat=num_qubits)
    ]

    # True Object
    true_object = generate_qoperation(
        mode="mprocess", name=true_object_name, c_sys=c_sys
    )
    if on_para_eq_constraint is False:
        true_object = MProcess(
            hss=true_object.hss, on_para_eq_constraint=False, c_sys=c_sys
        )

    # Qmpt
    qmpt = StandardQmpt(
        states=tester_states,
        povms=tester_povms,
        num_outcomes=true_object.num_outcomes,
        on_para_eq_constraint=on_para_eq_constraint,
        eps_proj_physical=1e-5,
        schedules="all",
    )

    # empi_dists
    prob_dists = qmpt.calc_prob_dists(true_object)
    empi_dists = [(10, prob_dist) for prob_dist in prob_dists]

    # Estimator
    estimator = LossMinimizationEstimator()
    loss = StandardQTomographyBasedWeightedProbabilityBasedSquaredError()
    loss_option = StandardQTomographyBasedWeightedProbabilityBasedSquaredErrorOption(
        "identity"
    )
    algo = ProjectedGradientDescentBacktracking()
    algo_option = ProjectedGradientDescentBacktrackingOption(
        mode_stopping_criterion_gradient_descent="sum_absolute_difference_variable",
        num_history_stopping_criterion_gradient_descent=1,
        eps=1e-9,
    )

    # Act
    result = estimator.calc_estimate(
        qtomography=qmpt,
        empi_dists=empi_dists,
        loss=loss,
        loss_option=loss_option,
        algo=algo,
        algo_option=algo_option,
        is_computation_time_required=True,
    )
    actual = result.estimated_qoperation

    # Assert
    for a, e in zip(actual.hss, true_object.hss):
        npt.assert_almost_equal(a, e, decimal=2)


@pytest.mark.qmpt_onequtrit
@pytest.mark.parametrize(
    ("true_object_name", "on_para_eq_constraint"),
    [("z3-type1", True), ("z3-type1", False), ("z2-type1", True), ("z2-type1", False)],
)
def test_calc_estimate_LSE_1qutrit(true_object_name: str, on_para_eq_constraint: bool):
    # Arrange
    num_qubits = 1
    c_sys = generate_composite_system(mode="qutrit", num=num_qubits)

    # Tester Objects
    state_names = [
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
    povm_names = ["01x3", "01y3", "z3", "12x3", "12y3", "02x3", "02y3"]

    tester_states = [
        generate_qoperation_object(
            mode="state", object_name="state", name=name, c_sys=c_sys
        )
        for name in state_names
    ]
    tester_povms = [
        generate_qoperation_object(
            mode="povm", object_name="povm", name=name, c_sys=c_sys
        )
        for name in povm_names
    ]

    # True Object
    true_object = generate_qoperation(
        mode="mprocess", name=true_object_name, c_sys=c_sys
    )
    if on_para_eq_constraint is False:
        true_object = MProcess(
            hss=true_object.hss, on_para_eq_constraint=False, c_sys=c_sys
        )

    # Qmpt
    qmpt = StandardQmpt(
        states=tester_states,
        povms=tester_povms,
        num_outcomes=true_object.num_outcomes,
        on_para_eq_constraint=on_para_eq_constraint,
        eps_proj_physical=1e-5,
        schedules="all",
    )

    # empi_dists
    prob_dists = qmpt.calc_prob_dists(true_object)
    empi_dists = [(10, prob_dist) for prob_dist in prob_dists]

    # Estimator
    estimator = LossMinimizationEstimator()
    loss = StandardQTomographyBasedWeightedProbabilityBasedSquaredError()
    loss_option = StandardQTomographyBasedWeightedProbabilityBasedSquaredErrorOption(
        "identity"
    )
    algo = ProjectedGradientDescentBacktracking()
    algo_option = ProjectedGradientDescentBacktrackingOption(
        mode_stopping_criterion_gradient_descent="sum_absolute_difference_variable",
        num_history_stopping_criterion_gradient_descent=1,
        eps=1e-9,
    )

    # Act
    result = estimator.calc_estimate(
        qtomography=qmpt,
        empi_dists=empi_dists,
        loss=loss,
        loss_option=loss_option,
        algo=algo,
        algo_option=algo_option,
        is_computation_time_required=True,
    )
    actual = result.estimated_qoperation

    # Assert
    for a, e in zip(actual.hss, true_object.hss):
        npt.assert_almost_equal(a, e, decimal=1)
