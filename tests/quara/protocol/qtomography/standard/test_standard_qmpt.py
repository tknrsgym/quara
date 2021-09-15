import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.qoperation_typical import generate_qoperation_object
from quara.objects.composite_system_typical import generate_composite_system
from quara.protocol.qtomography.standard.standard_qmpt import (
    cqpt_to_cqmpt,
    StandardQmpt,
)


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
        seed=7,
    )

    # Assert
    assert actual.calc_matA().shape == (48, 28)
    assert actual.calc_vecB().shape == (48,)
