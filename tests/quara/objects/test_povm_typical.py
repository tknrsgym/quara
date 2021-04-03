import itertools

import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import (
    get_normalized_pauli_basis,
    get_normalized_gell_mann_basis,
)
from quara.objects.operators import tensor_product
from quara.objects.povm import Povm
from quara.objects import povm_typical


def test_generate_povm_object_from_povm_name_object_name():
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # generate_povm_pure_state_vectors_from_name
    actual = povm_typical.generate_povm_pure_state_vectors_from_name("z")
    expected = [
        np.array([1, 0]),
        np.array([0, 1]),
    ]
    for a, e in zip(actual, expected):
        npt.assert_almost_equal(a, e, decimal=15)

    # generate_povm_matrices_from_name
    actual = povm_typical.generate_povm_matrices_from_name("z")
    expected = [
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
    ]
    for a, e in zip(actual, expected):
        npt.assert_almost_equal(a, e, decimal=15)

    # generate_povm_vectors_from_name
    actual = povm_typical.generate_povm_vectors_from_name(
        "z", get_normalized_pauli_basis()
    )
    expected = [
        np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2),
        np.array([1, 0, 0, -1], dtype=np.float64) / np.sqrt(2),
    ]
    for a, e in zip(actual, expected):
        npt.assert_almost_equal(a, e, decimal=15)

    # generate_povm_from_name
    actual = povm_typical.generate_povm_from_name("z", c_sys)
    expected = [
        np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2),
        np.array([1, 0, 0, -1], dtype=np.float64) / np.sqrt(2),
    ]
    for a, e in zip(actual.vecs, expected):
        npt.assert_almost_equal(a, e, decimal=15)


@pytest.mark.parametrize(
    ("povm_name", "expected_vecs"),
    [
        (
            "x",
            [
                np.array([1, 1, 0, 0], dtype=np.float64) / np.sqrt(2),
                np.array([1, -1, 0, 0], dtype=np.float64) / np.sqrt(2),
            ],
        ),
        (
            "y",
            [
                np.array([1, 0, 1, 0], dtype=np.float64) / np.sqrt(2),
                np.array([1, 0, -1, 0], dtype=np.float64) / np.sqrt(2),
            ],
        ),
        (
            "z",
            [
                np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2),
                np.array([1, 0, 0, -1], dtype=np.float64) / np.sqrt(2),
            ],
        ),
    ],
)
def test_generate_povm_from_name_1qubit(povm_name, expected_vecs):
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    actual = povm_typical.generate_povm_from_name(povm_name, c_sys=c_sys)
    for actual_vec, expected_vec in zip(actual.vecs, expected_vecs):
        npt.assert_almost_equal(actual_vec, expected_vec, decimal=15)


@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("povm_name", "expected_vecs"),
    [
        (
            "bell",
            [
                np.array(
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1], dtype=np.float64
                )
                / 2,
                np.array(
                    [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=np.float64
                )
                / 2,
                np.array(
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1], dtype=np.float64
                )
                / 2,
                np.array(
                    [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                    dtype=np.float64,
                )
                / 2,
            ],
        ),
        (
            "z_z",
            [
                np.array(
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], dtype=np.float64
                )
                / 2,
                np.array(
                    [1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1], dtype=np.float64
                )
                / 2,
                np.array(
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1], dtype=np.float64
                )
                / 2,
                np.array(
                    [1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1], dtype=np.float64
                )
                / 2,
            ],
        ),
    ],
)
def test_generate_povm_from_name_2qubit(povm_name, expected_vecs):
    e_sys0 = ElementalSystem(0, get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])

    actual = povm_typical.generate_povm_from_name(povm_name, c_sys=c_sys)
    for actual_vec, expected_vec in zip(actual.vecs, expected_vecs):
        npt.assert_almost_equal(actual_vec, expected_vec, decimal=15)


def get_z_tensors(num_tensor: int):
    z = [
        np.array([1, 1], dtype=np.float64),
        np.array([1, -1], dtype=np.float64),
    ]
    signs_list = z
    for _ in range(num_tensor - 1):
        signs_list = [
            np.kron(vec1, vec2) for vec1, vec2 in itertools.product(signs_list, z)
        ]

    if num_tensor == 1:
        indices = [0, 3]
    elif num_tensor == 2:
        indices = [0, 3, 12, 15]
    elif num_tensor == 3:
        indices = [0, 3, 12, 15, 48, 51, 60, 63]

    vecs = []
    for signs in signs_list:
        vec = np.zeros(4 ** num_tensor)

        for index, sign in zip(indices, signs):
            vec[index] = sign
        vecs.append(vec / np.sqrt(2 ** num_tensor))
    return vecs


@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("povm_name", "expected_vecs"),
    [("z_z_z", get_z_tensors(3))],
)
def test_generate_povm_from_name_3qubit(povm_name, expected_vecs):
    e_sys0 = ElementalSystem(0, get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, get_normalized_pauli_basis())
    e_sys2 = ElementalSystem(2, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1, e_sys2])

    actual = povm_typical.generate_povm_from_name(povm_name, c_sys=c_sys)
    for actual_vec, expected_vec in zip(actual.vecs, expected_vecs):
        npt.assert_almost_equal(actual_vec, expected_vec, decimal=15)


@pytest.mark.parametrize(
    ("povm_name", "expected_vecs"),
    [
        (
            "z3",
            [
                np.array(
                    [np.sqrt(1 / 3), 0, 0, np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
                np.array(
                    [np.sqrt(1 / 3), 0, 0, -np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
                np.array(
                    [np.sqrt(1 / 3), 0, 0, 0, 0, 0, 0, 0, -2 * np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
            ],
        ),
        (
            "z2",
            [
                np.array(
                    [np.sqrt(1 / 3), 0, 0, np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
                np.array(
                    [np.sqrt(1 / 3), 0, 0, -np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 6)],
                    dtype=np.float64,
                )
                + np.array(
                    [np.sqrt(1 / 3), 0, 0, 0, 0, 0, 0, 0, -2 * np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
            ],
        ),
        (
            "01x3",
            [
                np.array(
                    [np.sqrt(1 / 3), np.sqrt(1 / 2), 0, 0, 0, 0, 0, 0, np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
                np.array(
                    [np.sqrt(1 / 3), -np.sqrt(1 / 2), 0, 0, 0, 0, 0, 0, np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
                np.array(
                    [np.sqrt(1 / 3), 0, 0, 0, 0, 0, 0, 0, -2 * np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
            ],
        ),
        (
            "02x3",
            [
                np.array(
                    [
                        np.sqrt(1 / 3),
                        0,
                        0,
                        np.sqrt(1 / 2) / 2,
                        np.sqrt(1 / 2),
                        0,
                        0,
                        0,
                        -np.sqrt(1 / 6) / 2,
                    ],
                    dtype=np.float64,
                ),
                np.array(
                    [
                        np.sqrt(1 / 3),
                        0,
                        0,
                        np.sqrt(1 / 2) / 2,
                        -np.sqrt(1 / 2),
                        0,
                        0,
                        0,
                        -np.sqrt(1 / 6) / 2,
                    ],
                    dtype=np.float64,
                ),
                np.array(
                    [np.sqrt(1 / 3), 0, 0, -np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
            ],
        ),
        (
            "12y3",
            [
                np.array(
                    [
                        np.sqrt(1 / 3),
                        0,
                        0,
                        -np.sqrt(1 / 2) / 2,
                        0,
                        0,
                        0,
                        np.sqrt(1 / 2),
                        -np.sqrt(1 / 6) / 2,
                    ],
                    dtype=np.float64,
                ),
                np.array(
                    [
                        np.sqrt(1 / 3),
                        0,
                        0,
                        -np.sqrt(1 / 2) / 2,
                        0,
                        0,
                        0,
                        -np.sqrt(1 / 2),
                        -np.sqrt(1 / 6) / 2,
                    ],
                    dtype=np.float64,
                ),
                np.array(
                    [np.sqrt(1 / 3), 0, 0, np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
            ],
        ),
    ],
)
def test_generate_povm_from_name_1qutrit(povm_name, expected_vecs):
    e_sys = ElementalSystem(0, get_normalized_gell_mann_basis())
    c_sys = CompositeSystem([e_sys])

    actual = povm_typical.generate_povm_from_name(povm_name, c_sys=c_sys)
    for actual_vec, expected_vec in zip(actual.vecs, expected_vecs):
        npt.assert_almost_equal(actual_vec, expected_vec, decimal=15)


def test_generate_povm_from_name_2qutrit_z3_z3():
    # Arrange
    e_sys0 = ElementalSystem(0, get_normalized_gell_mann_basis())
    c_sys0 = CompositeSystem([e_sys0])
    e_sys1 = ElementalSystem(1, get_normalized_gell_mann_basis())
    c_sys1 = CompositeSystem([e_sys1])
    c_sys = CompositeSystem([e_sys0, e_sys1])

    # Act
    actual = povm_typical.generate_povm_from_name("z3_z3", c_sys=c_sys)

    # Assert
    vecs_z3 = [
        np.array(
            [np.sqrt(1 / 3), 0, 0, np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 6)],
            dtype=np.float64,
        ),
        np.array(
            [np.sqrt(1 / 3), 0, 0, -np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 6)],
            dtype=np.float64,
        ),
        np.array(
            [np.sqrt(1 / 3), 0, 0, 0, 0, 0, 0, 0, -2 * np.sqrt(1 / 6)],
            dtype=np.float64,
        ),
    ]
    povm0 = Povm(c_sys0, vecs_z3)
    povm1 = Povm(c_sys1, vecs_z3)
    expected = tensor_product(povm0, povm1)

    for actual_vec, expected_vec in zip(actual.vecs, expected.vecs):
        npt.assert_almost_equal(actual_vec, expected_vec, decimal=15)


def test_generate_povm_from_name_2qutrit_z2_z2():
    # Arrange
    e_sys0 = ElementalSystem(0, get_normalized_gell_mann_basis())
    c_sys0 = CompositeSystem([e_sys0])
    e_sys1 = ElementalSystem(1, get_normalized_gell_mann_basis())
    c_sys1 = CompositeSystem([e_sys1])
    c_sys = CompositeSystem([e_sys0, e_sys1])

    # Act
    actual = povm_typical.generate_povm_from_name("z2_z2", c_sys=c_sys)

    # Assert
    vecs_z2 = [
        np.array(
            [np.sqrt(1 / 3), 0, 0, np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 6)],
            dtype=np.float64,
        ),
        np.array(
            [np.sqrt(1 / 3), 0, 0, -np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 6)],
            dtype=np.float64,
        )
        + np.array(
            [np.sqrt(1 / 3), 0, 0, 0, 0, 0, 0, 0, -2 * np.sqrt(1 / 6)],
            dtype=np.float64,
        ),
    ]
    povm0 = Povm(c_sys0, vecs_z2)
    povm1 = Povm(c_sys1, vecs_z2)
    expected = tensor_product(povm0, povm1)

    for actual_vec, expected_vec in zip(actual.vecs, expected.vecs):
        npt.assert_almost_equal(actual_vec, expected_vec, decimal=15)


def test_generate_povm_from_name_2qutrit_01x3_12y3():
    # Arrange
    e_sys0 = ElementalSystem(0, get_normalized_gell_mann_basis())
    c_sys0 = CompositeSystem([e_sys0])
    e_sys1 = ElementalSystem(1, get_normalized_gell_mann_basis())
    c_sys1 = CompositeSystem([e_sys1])
    c_sys = CompositeSystem([e_sys0, e_sys1])

    # Act
    actual = povm_typical.generate_povm_from_name("01x3_12y3", c_sys=c_sys)

    # Assert
    vecs_01x3 = [
        np.array(
            [np.sqrt(1 / 3), np.sqrt(1 / 2), 0, 0, 0, 0, 0, 0, np.sqrt(1 / 6)],
            dtype=np.float64,
        ),
        np.array(
            [np.sqrt(1 / 3), -np.sqrt(1 / 2), 0, 0, 0, 0, 0, 0, np.sqrt(1 / 6)],
            dtype=np.float64,
        ),
        np.array(
            [np.sqrt(1 / 3), 0, 0, 0, 0, 0, 0, 0, -2 * np.sqrt(1 / 6)],
            dtype=np.float64,
        ),
    ]
    vecs_12y3 = [
        np.array(
            [
                np.sqrt(1 / 3),
                0,
                0,
                -np.sqrt(1 / 2) / 2,
                0,
                0,
                0,
                np.sqrt(1 / 2),
                -np.sqrt(1 / 6) / 2,
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                np.sqrt(1 / 3),
                0,
                0,
                -np.sqrt(1 / 2) / 2,
                0,
                0,
                0,
                -np.sqrt(1 / 2),
                -np.sqrt(1 / 6) / 2,
            ],
            dtype=np.float64,
        ),
        np.array(
            [np.sqrt(1 / 3), 0, 0, np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 6)],
            dtype=np.float64,
        ),
    ]
    povm0 = Povm(c_sys0, vecs_01x3)
    povm1 = Povm(c_sys1, vecs_12y3)
    expected = tensor_product(povm0, povm1)

    for actual_vec, expected_vec in zip(actual.vecs, expected.vecs):
        npt.assert_almost_equal(actual_vec, expected_vec, decimal=15)
