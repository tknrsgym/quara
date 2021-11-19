import numpy as np
import numpy.testing as npt
import pytest


from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.mprocess_typical import (
    generate_mprocess_object_from_mprocess_name_object_name,
)
from quara.settings import Settings


@pytest.mark.parametrize(
    ("mprocess_name", "expected_set_pure_state_vectors"),
    [
        (
            "x-type1",
            [
                [
                    np.array([1, 1], dtype=np.complex128) / np.sqrt(2),
                ],
                [
                    np.array([1, -1], dtype=np.complex128) / np.sqrt(2),
                ],
            ],
        ),
        (
            "y-type1",
            [
                [
                    np.array([1, 1j], dtype=np.complex128) / np.sqrt(2),
                ],
                [
                    np.array([1, -1j], dtype=np.complex128) / np.sqrt(2),
                ],
            ],
        ),
        (
            "z-type1",
            [
                [
                    np.array([1, 0], dtype=np.complex128),
                ],
                [
                    np.array([0, 1], dtype=np.complex128),
                ],
            ],
        ),
        (
            "bell-type1",
            [
                [
                    np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2),
                ],
                [
                    np.array([1, 0, 0, -1], dtype=np.complex128) / np.sqrt(2),
                ],
                [
                    np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2),
                ],
                [
                    np.array([0, 1, -1, 0], dtype=np.complex128) / np.sqrt(2),
                ],
            ],
        ),
        (
            "z3-type1",
            [
                [
                    np.array([1, 0, 0], dtype=np.complex128),
                ],
                [
                    np.array([0, 1, 0], dtype=np.complex128),
                ],
                [
                    np.array([0, 0, 1], dtype=np.complex128),
                ],
            ],
        ),
        (
            "z2-type1",
            [
                [
                    np.array([1, 0, 0], dtype=np.complex128),
                ],
                [
                    np.array([0, 1, 0], dtype=np.complex128),
                    np.array([0, 0, 1], dtype=np.complex128),
                ],
            ],
        ),
    ],
)
def test_generate_mprocess_object_from_mprocess_name_object_name__set_pure_state_vectors(
    mprocess_name, expected_set_pure_state_vectors
):
    actual = generate_mprocess_object_from_mprocess_name_object_name(
        mprocess_name, "set_pure_state_vectors"
    )
    for actual_vectors, expected_vectors in zip(
        actual, expected_set_pure_state_vectors
    ):
        for actual_vec, expected_vec in zip(actual_vectors, expected_vectors):
            npt.assert_almost_equal(actual_vec, expected_vec, decimal=15)


@pytest.mark.parametrize(
    ("mprocess_name", "expected_set_pure_state_vectors"),
    [
        (
            "x-type1_z-type1",
            [
                [
                    np.array([1, 0, 1, 0], dtype=np.complex128) / np.sqrt(2),
                ],
                [
                    np.array([0, 1, 0, 1], dtype=np.complex128) / np.sqrt(2),
                ],
                [
                    np.array([1, 0, -1, 0], dtype=np.complex128) / np.sqrt(2),
                ],
                [
                    np.array([0, 1, 0, -1], dtype=np.complex128) / np.sqrt(2),
                ],
            ],
        ),
    ],
)
def test_generate_mprocess_object_from_mprocess_name_object_name__set_pure_state_vectors__tensor_product(
    mprocess_name, expected_set_pure_state_vectors
):
    actual = generate_mprocess_object_from_mprocess_name_object_name(
        mprocess_name, "set_pure_state_vectors"
    )
    for actual_vectors, expected_vectors in zip(
        actual, expected_set_pure_state_vectors
    ):
        for actual_vec, expected_vec in zip(actual_vectors, expected_vectors):
            npt.assert_almost_equal(actual_vec, expected_vec, decimal=15)


@pytest.mark.parametrize(
    ("mprocess_name", "expected_set_kraus_matrices"),
    [
        (
            "x-type2",
            [
                [
                    np.array([[1, 1], [1, 1]], dtype=np.complex128) / 2,
                ],
                [
                    np.array([[1, -1], [1, -1]], dtype=np.complex128) / 2,
                ],
            ],
        ),
        (
            "y-type2",
            [
                [
                    np.array([[1, -1j], [1j, 1]], dtype=np.complex128) / 2,
                ],
                [
                    np.array([[1, 1j], [1j, -1]], dtype=np.complex128) / 2,
                ],
            ],
        ),
        (
            "z-type2",
            [
                [
                    np.array([[1, 0], [0, 0]], dtype=np.complex128),
                ],
                [
                    np.array([[0, 1], [0, 0]], dtype=np.complex128),
                ],
            ],
        ),
        (
            "xxparity-type1",
            [
                [
                    0.5
                    * np.array(
                        [[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]],
                        dtype=np.complex128,
                    )
                ],
                [
                    0.5
                    * np.array(
                        [[1, 0, 0, -1], [0, 1, -1, 0], [0, -1, 1, 0], [-1, 0, 0, 1]],
                        dtype=np.complex128,
                    )
                ],
            ],
        ),
        (
            "zzparity-type1",
            [
                [
                    np.array(
                        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
                        dtype=np.complex128,
                    )
                ],
                [
                    np.array(
                        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                        dtype=np.complex128,
                    )
                ],
            ],
        ),
        (
            "z3-type2",
            [
                [
                    np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.complex128),
                ],
                [
                    np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.complex128),
                ],
                [
                    np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.complex128),
                ],
            ],
        ),
        (
            "z2-type2",
            [
                [
                    np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.complex128),
                ],
                [
                    np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.complex128),
                    np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.complex128),
                ],
            ],
        ),
    ],
)
def test_generate_mprocess_object_from_mprocess_name_object_name__set_kraus_matrices(
    mprocess_name, expected_set_kraus_matrices
):
    actual = generate_mprocess_object_from_mprocess_name_object_name(
        mprocess_name, "set_kraus_matrices"
    )
    for actual_vectors, expected_vectors in zip(actual, expected_set_kraus_matrices):
        for actual_vec, expected_vec in zip(actual_vectors, expected_vectors):
            npt.assert_almost_equal(actual_vec, expected_vec, decimal=15)


@pytest.mark.parametrize(
    ("mprocess_name", "expected_set_kraus_matrices"),
    [
        (
            "x-type1_z-type1",
            [
                [
                    np.array(
                        [[1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]],
                        dtype=np.complex128,
                    )
                    / 2,
                ],
                [
                    np.array(
                        [[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]],
                        dtype=np.complex128,
                    )
                    / 2,
                ],
                [
                    np.array(
                        [[1, 0, -1, 0], [0, 0, 0, 0], [-1, 0, 1, 0], [0, 0, 0, 0]],
                        dtype=np.complex128,
                    )
                    / 2,
                ],
                [
                    np.array(
                        [[0, 0, 0, 0], [0, 1, 0, -1], [0, 0, 0, 0], [0, -1, 0, 1]],
                        dtype=np.complex128,
                    )
                    / 2,
                ],
            ],
        ),
    ],
)
def test_generate_mprocess_object_from_mprocess_name_object_name__set_kraus_matrices__tensor_product(
    mprocess_name, expected_set_kraus_matrices
):
    actual = generate_mprocess_object_from_mprocess_name_object_name(
        mprocess_name, "set_kraus_matrices"
    )
    print(actual)
    for actual_vectors, expected_vectors in zip(actual, expected_set_kraus_matrices):
        for actual_vec, expected_vec in zip(actual_vectors, expected_vectors):
            npt.assert_almost_equal(actual_vec, expected_vec, decimal=15)


@pytest.mark.parametrize(
    ("mprocess_name", "expected_hss"),
    [
        (
            "z-type1",
            [
                np.array(
                    [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]],
                    dtype=np.complex128,
                )
                / 2,
                np.array(
                    [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]],
                    dtype=np.complex128,
                )
                / 2,
            ],
        ),
    ],
)
def test_generate_mprocess_object_from_mprocess_name_object_name__hss(
    mprocess_name, expected_hss
):
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    actual = generate_mprocess_object_from_mprocess_name_object_name(
        mprocess_name,
        "hss",
        c_sys=c_sys,
    )
    for actual_hs, expected_hs in zip(actual, expected_hss):
        npt.assert_almost_equal(actual_hs, expected_hs, decimal=15)


@pytest.mark.parametrize(
    ("mprocess_name", "expected_hss"),
    [
        (
            "x-type1_z-type1",
            [
                np.array(
                    [
                        [0.25, 0, 0, 0.25, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0.25, 0, 0, 0.25, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0.25, 0, 0, 0.25, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0.25, 0, 0, 0.25, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.complex128,
                ),
                np.array(
                    [
                        [0.25, 0, 0, -0.25, 0.25, 0, 0, -0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-0.25, 0, 0, 0.25, -0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0.25, 0, 0, -0.25, 0.25, 0, 0, -0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-0.25, 0, 0, 0.25, -0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.complex128,
                ),
                np.array(
                    [
                        [0.25, 0, 0, 0.25, -0.25, 0, 0, -0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0.25, 0, 0, 0.25, -0.25, 0, 0, -0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-0.25, 0, 0, -0.25, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-0.25, 0, 0, -0.25, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.complex128,
                ),
                np.array(
                    [
                        [0.25, 0, 0, -0.25, -0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-0.25, 0, 0, 0.25, 0.25, 0, 0, -0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-0.25, 0, 0, 0.25, 0.25, 0, 0, -0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0.25, 0, 0, -0.25, -0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.complex128,
                ),
            ],
        ),
    ],
)
def test_generate_mprocess_object_from_mprocess_name_object_name__hss__tensor_product(
    mprocess_name, expected_hss
):
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])

    actual = generate_mprocess_object_from_mprocess_name_object_name(
        mprocess_name,
        "hss",
        c_sys=c_sys,
    )
    print(actual)
    for actual_hs, expected_hs in zip(actual, expected_hss):
        npt.assert_almost_equal(actual_hs, expected_hs, decimal=15)


@pytest.mark.parametrize(
    ("mprocess_name", "expected_hss"),
    [
        (
            "z-type1",
            [
                np.array(
                    [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]],
                    dtype=np.complex128,
                )
                / 2,
                np.array(
                    [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]],
                    dtype=np.complex128,
                )
                / 2,
            ],
        ),
    ],
)
def test_generate_mprocess_object_from_mprocess_name_object_name__mprocess(
    mprocess_name, expected_hss
):
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    actual = generate_mprocess_object_from_mprocess_name_object_name(
        mprocess_name,
        "mprocess",
        c_sys=c_sys,
    )
    for actual_hs, expected_hs in zip(actual.hss, expected_hss):
        npt.assert_almost_equal(actual_hs, expected_hs, decimal=15)
    assert actual.dim == 2
    assert actual.shape == (2,)
    assert actual.mode_sampling == False
    assert actual.is_physicality_required == True
    assert actual.is_estimation_object == True
    assert actual.on_para_eq_constraint == True
    assert actual.on_algo_eq_constraint == True
    assert actual.on_algo_ineq_constraint == True
    assert actual.eps_proj_physical == Settings.get_atol() / 10.0


@pytest.mark.parametrize(
    ("mprocess_name", "expected_hss"),
    [
        (
            "xxparity-type1",
            [
                np.array(
                    [
                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1],
                    ],
                    dtype=np.complex128,
                )
                / 2,
                np.array(
                    [
                        [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                    ],
                    dtype=np.complex128,
                )
                / 2,
            ],
        ),
        (
            "zzparity-type1",
            [
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ],
                    dtype=np.complex128,
                )
                / 2,
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ],
                    dtype=np.complex128,
                )
                / 2,
            ],
        ),
    ],
)
def test_generate_mprocess_object_from_mprocess_name_object_name__mprocess_2qubits(
    mprocess_name, expected_hss
):
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])

    actual = generate_mprocess_object_from_mprocess_name_object_name(
        mprocess_name,
        "mprocess",
        c_sys=c_sys,
    )
    for actual_hs, expected_hs in zip(actual.hss, expected_hss):
        print(f"actual  ={actual_hs}")
        print(f"expected={expected_hs}")
        npt.assert_almost_equal(actual_hs, expected_hs, decimal=15)
    assert actual.dim == 4
    assert actual.shape == (2,)
    assert actual.mode_sampling == False
    assert actual.is_physicality_required == True
    assert actual.is_estimation_object == True
    assert actual.on_para_eq_constraint == True
    assert actual.on_algo_eq_constraint == True
    assert actual.on_algo_ineq_constraint == True
    assert actual.eps_proj_physical == Settings.get_atol() / 10.0


@pytest.mark.parametrize(
    ("mprocess_name", "expected_hss"),
    [
        (
            "x-type1_z-type1",
            [
                np.array(
                    [
                        [0.25, 0, 0, 0.25, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0.25, 0, 0, 0.25, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0.25, 0, 0, 0.25, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0.25, 0, 0, 0.25, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.complex128,
                ),
                np.array(
                    [
                        [0.25, 0, 0, -0.25, 0.25, 0, 0, -0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-0.25, 0, 0, 0.25, -0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0.25, 0, 0, -0.25, 0.25, 0, 0, -0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-0.25, 0, 0, 0.25, -0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.complex128,
                ),
                np.array(
                    [
                        [0.25, 0, 0, 0.25, -0.25, 0, 0, -0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0.25, 0, 0, 0.25, -0.25, 0, 0, -0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-0.25, 0, 0, -0.25, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-0.25, 0, 0, -0.25, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.complex128,
                ),
                np.array(
                    [
                        [0.25, 0, 0, -0.25, -0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-0.25, 0, 0, 0.25, 0.25, 0, 0, -0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-0.25, 0, 0, 0.25, 0.25, 0, 0, -0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0.25, 0, 0, -0.25, -0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.complex128,
                ),
            ],
        ),
    ],
)
def test_generate_mprocess_object_from_mprocess_name_object_name__mprocess__tensor_product(
    mprocess_name, expected_hss
):
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])

    actual = generate_mprocess_object_from_mprocess_name_object_name(
        mprocess_name,
        "mprocess",
        c_sys=c_sys,
    )
    for actual_hs, expected_hs in zip(actual.hss, expected_hss):
        npt.assert_almost_equal(actual_hs, expected_hs, decimal=15)
