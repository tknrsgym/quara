import itertools
import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

import quara.objects.composite_system as csys
import quara.objects.elemental_system as esys
from quara.objects.matrix_basis import (
    get_comp_basis,
    get_gell_mann_basis,
    get_normalized_pauli_basis,
    get_pauli_basis,
)
from quara.objects.operators import tensor_product
from quara.objects.povm import (
    Povm,
    get_x_measurement,
    get_xx_measurement,
    get_xy_measurement,
    get_xz_measurement,
    get_y_measurement,
    get_yx_measurement,
    get_yy_measurement,
    get_yz_measurement,
    get_z_measurement,
    get_zx_measurement,
    get_zy_measurement,
    get_zz_measurement,
)
from quara.protocol import simple_io as s_io


class TestPovm:
    def test_validate_set_of_hermitian_matrices_ok(self):
        # Arrange
        p1 = np.array([1, 0, 0, 0], dtype=np.complex128)
        p2 = np.array([0, 0, 0, 1], dtype=np.complex128)
        vecs = [p1, p2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)

        # Assert
        expected = [p1, p2]
        assert (povm[0] == expected[0]).all()
        assert (povm[1] == expected[1]).all()
        assert povm.composite_system is c_sys

    def test_validate_set_of_hermitian_matrices_ng(self):
        # Arrange
        p1 = np.array([1, 0, 0, 0], dtype=np.complex128)
        p2 = np.array([0, 1, 0, 0], dtype=np.complex128)
        vecs = [p1, p2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        with pytest.raises(ValueError):
            # ValueError: povm must be a set of Hermitian matrices
            _ = Povm(c_sys=c_sys, vecs=vecs)

    def test_validate_set_of_hermitian_matrices_not_physical_ok(self):
        # Arrange
        p1 = np.array([1, 0, 0, 0], dtype=np.complex128)
        p2 = np.array([0, 1, 0, 0], dtype=np.complex128)
        vecs = [p1, p2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        # Test that no exceptions are raised.
        _ = Povm(c_sys=c_sys, vecs=vecs, is_physical=False)

    def test_validate_sum_is_identity_ok(self):
        # Arrange
        p1 = np.array([1, 0, 0, 0], dtype=np.complex128)
        p2 = np.array([0, 0, 0, 1], dtype=np.complex128)
        vecs = [p1, p2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)
        actual = povm.is_identity()

        # Assert
        assert actual is True

    def test_validate_sum_is_identity_ng(self):
        # Arrange
        p1 = np.array([1, 0, 0, 0], dtype=np.complex128)
        p2 = np.array([0, 1, 0, 0], dtype=np.complex128)
        vecs = [p1, p2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        with pytest.raises(ValueError):
            # ValueError: The sum of the elements of POVM must be an identity matrix.
            _ = Povm(c_sys=c_sys, vecs=vecs)

    def test_validate_sum_is_identity_not_physical_ok(self):
        # Arrange
        p1 = np.array(
            [0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j], dtype=np.complex128
        )
        p2 = np.array(
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128
        )
        vecs = [p1, p2]

        e_sys = esys.ElementalSystem(1, get_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        # Test that no exceptions are raised.
        _ = Povm(c_sys=c_sys, vecs=vecs, is_physical=False)

    def test_validate_is_positive_semidefinite_ok(self):
        # Arrange
        ps_1 = np.array([1, 0, 0, 0], dtype=np.complex128)
        ps_2 = np.array([0, 0, 0, 1], dtype=np.complex128)
        vecs = [ps_1, ps_2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)
        actual = povm.is_positive_semidefinite()

        # Assert
        assert actual is True

    def test_validate_is_positive_semidefinite_ng(self):
        # Arrange
        ps = np.array([1, 0, 0, 0], dtype=np.complex128)
        not_ps = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        vecs = [ps, not_ps]

        e_sys = esys.ElementalSystem(1, get_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        with pytest.raises(ValueError):
            _ = Povm(c_sys=c_sys, vecs=vecs)

    def test_validate_is_positive_semidefinite_not_physical_ok(self):
        # Arrange
        ps = np.array([1, 0, 0, 0], dtype=np.complex128)
        not_ps = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        vecs = [ps, not_ps]

        e_sys = esys.ElementalSystem(1, get_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        # Test that no exceptions are raised.
        povm = Povm(c_sys=c_sys, vecs=vecs, is_physical=False)
        actual = povm.is_positive_semidefinite()

        # Assert
        assert actual is False

    def test_calc_eigenvalues_all(self):
        # Arrange
        vec_1 = np.array([1, 0, 0, 0], dtype=np.complex128)
        vec_2 = np.array([0, 0, 0, 1], dtype=np.complex128)
        vecs = [vec_1, vec_2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)
        actual = povm.calc_eigenvalues()

        # Assert
        expected = [
            np.array([1, 0], dtype=np.complex128),
            np.array([0, 1], dtype=np.complex128),
        ]

        assert len(actual) == len(expected)
        npt.assert_almost_equal(actual[0], expected[0], decimal=15)
        npt.assert_almost_equal(actual[1], expected[1], decimal=15)

    def test_calc_eigenvalues_one(self):
        # Arrange
        vec_1 = np.array([1, 0, 0, 0], dtype=np.complex128)
        vec_2 = np.array([0, 0, 0, 1], dtype=np.complex128)
        vecs = [vec_1, vec_2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)
        actual = povm.calc_eigenvalues(0)

        # Assert
        expected = np.array([1, 0], dtype=np.complex128)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)
        actual = povm.calc_eigenvalues(1)

        # Assert
        expected = np.array([0, 1], dtype=np.complex128)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_validate_dim_ng(self):
        # Arrange
        test_root_dir = Path(os.path.dirname(__file__)).parent.parent
        data_dir = test_root_dir / "data"

        dim = 2 ** 2  # 2 qubits
        num_state = 16
        num_povm = 9
        num_outcome = 4

        povms = s_io.load_povm_list(
            data_dir / "tester_2qubit_povm.csv",
            dim=dim,
            num_povm=num_povm,
            num_outcome=num_outcome,
        )
        vecs = list(povms[0])  # 2qubit

        e_sys = esys.ElementalSystem(1, get_pauli_basis())  # 1qubit
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        with pytest.raises(ValueError):
            _ = Povm(c_sys=c_sys, vecs=vecs)

    def test_convert_basis(self):
        # Arrange
        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])
        ps_1 = np.array([1, 0, 0, 0], dtype=np.complex128)
        ps_2 = np.array([0, 0, 0, 1], dtype=np.complex128)
        vecs = [ps_1, ps_2]
        povm = Povm(c_sys=c_sys, vecs=vecs)
        to_basis = get_normalized_pauli_basis()

        # Act
        actual = povm.convert_basis(to_basis)

        # Assert
        expected = [
            1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.complex128),
            1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.complex128),
        ]
        assert len(actual) == len(expected)
        for i, a in enumerate(actual):
            assert np.all(a == expected[i])

    def test_measurements(self):
        # Case 1:
        # Arrange
        basis1 = get_comp_basis()
        e_sys1 = esys.ElementalSystem(1, basis1)
        c_sys1 = csys.CompositeSystem([e_sys1])
        vecs1 = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]
        povm1 = Povm(c_sys1, vecs1, is_physical=False)

        # Act
        actual = povm1.measurements

        # Assert
        expected = ["0", "1"]
        assert len(actual) == len(expected)

        for a, e in zip(actual, expected):
            assert a == e

        # Case2:
        # Arrange
        basis2 = get_comp_basis()
        e_sys2 = esys.ElementalSystem(2, basis2)
        c_sys2 = csys.CompositeSystem([e_sys2])
        vecs2 = [
            np.array([23, 29, 31, 37], dtype=np.float64),
            np.array([41, 43, 47, 53], dtype=np.float64),
        ]
        povm2 = Povm(c_sys2, vecs2, is_physical=False)
        povm12 = tensor_product(povm1, povm2)

        # Act
        actual = povm12.measurements

        # Assert
        expected = ["00", "01", "10", "11"]
        assert len(actual) == len(expected)

        for a, e in zip(actual, expected):
            assert a == e

    def test_measurement(self):
        # Case 1:
        # Arrange
        basis1 = get_comp_basis()
        e_sys1 = esys.ElementalSystem(1, basis1)
        c_sys1 = csys.CompositeSystem([e_sys1])
        vecs1 = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]
        povm1 = Povm(c_sys1, vecs1, is_physical=False)

        # Act
        actual = povm1.measurement("0")
        # Assert
        expected = povm1[0]
        assert np.all(actual == expected)

        # Act
        actual = povm1.measurement("1")
        # Assert
        expected = povm1[1]
        assert np.all(actual == expected)

        # Case2:
        # Arrange
        basis2 = get_comp_basis()
        e_sys2 = esys.ElementalSystem(2, basis2)
        c_sys2 = csys.CompositeSystem([e_sys2])
        vecs2 = [
            np.array([23, 29, 31, 37], dtype=np.float64),
            np.array([41, 43, 47, 53], dtype=np.float64),
        ]
        povm2 = Povm(c_sys2, vecs2, is_physical=False)
        povm12 = tensor_product(povm1, povm2)

        # Act
        actual = povm12.measurement("00")
        # Assert
        expected = povm12[0]
        assert np.all(actual == expected)

        # Act
        actual = povm12.measurement("01")
        # Assert
        expected = povm12[1]
        assert np.all(actual == expected)

        # Act
        actual = povm12.measurement("10")
        # Assert
        expected = povm12[2]
        assert np.all(actual == expected)

        # Act
        actual = povm12.measurement("11")
        # Assert
        expected = povm12[3]
        assert np.all(actual == expected)

    def test_measurement_unexpected(self):
        # Arrange
        basis1 = get_comp_basis()
        e_sys1 = esys.ElementalSystem(1, basis1)
        c_sys1 = csys.CompositeSystem([e_sys1])
        vecs1 = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]
        povm1 = Povm(c_sys1, vecs1, is_physical=False)

        # Act & Assert
        with pytest.raises(ValueError):
            # ValueError: That measurement does not exist.
            # See the list of measurements by 'measurement' property.
            _ = povm1.measurement("10")

    def test_matrix(self):
        # Case 1:
        # Arrange
        basis1 = get_comp_basis()
        e_sys1 = esys.ElementalSystem(1, basis1)
        c_sys1 = csys.CompositeSystem([e_sys1])
        vecs1 = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]
        povm1 = Povm(c_sys1, vecs1, is_physical=False)

        # Act
        actual = povm1.matrix("0")
        # Assert
        expected = povm1.matrixes()
        npt.assert_almost_equal(actual, expected[0], decimal=15)

        # Act
        actual = povm1.matrix("1")
        # Assert
        npt.assert_almost_equal(actual, expected[1], decimal=15)

        # Case2:
        # Arrange
        basis2 = get_comp_basis()
        e_sys2 = esys.ElementalSystem(2, basis2)
        c_sys2 = csys.CompositeSystem([e_sys2])
        vecs2 = [
            np.array([23, 29, 31, 37], dtype=np.float64),
            np.array([41, 43, 47, 53], dtype=np.float64),
        ]
        povm2 = Povm(c_sys2, vecs2, is_physical=False)
        povm12 = tensor_product(povm1, povm2)

        # Act
        actual = povm12.matrix("00")
        # Assert
        expected = povm12.matrixes()
        npt.assert_almost_equal(actual, expected[0], decimal=15)

        # Act
        actual = povm12.matrix("01")
        # Assert
        npt.assert_almost_equal(actual, expected[1], decimal=15)

        # Act
        actual = povm12.matrix("10")
        # Assert
        npt.assert_almost_equal(actual, expected[2], decimal=15)

        # Act
        actual = povm12.matrix("11")
        # Assert
        npt.assert_almost_equal(actual, expected[3], decimal=15)

    def test_matrix_unexpected(self):
        # Arrange
        basis1 = get_comp_basis()
        e_sys1 = esys.ElementalSystem(1, basis1)
        c_sys1 = csys.CompositeSystem([e_sys1])
        vecs1 = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]
        povm1 = Povm(c_sys1, vecs1, is_physical=False)

        # Act & Assert
        unexpected_type = [0]
        with pytest.raises(TypeError):
            # TypeError: The type of `key` must be int or str.
            _ = povm1.matrix(unexpected_type)

    # def test_vecs_size(self):
    #     # Arange
    #     e_sys = esys.ElementalSystem(1, get_comp_basis())
    #     c_sys = csys.CompositeSystem([e_sys])
    #     vec_1 = np.array([1, 0], dtype=np.complex128)
    #     vec_2 = np.array([0, 0, 0, 1], dtype=np.complex128)
    #     vecs = [vec_1, vec_2]
    #     povm = Povm(c_sys=c_sys, vecs=vecs, is_physical=False)

    #     # Act
    #     vec_sizes = povm.vec_sizes()

    #     # Assert
    #     expected = [2, 4]
    #     assert len(vec_sizes) == len(expected)
    #     for a, b in (vec_sizes, expected):
    #         assert a == b

    # def test_e_sys_dims(self):
    #     # Arange
    #     e_sys = esys.ElementalSystem(1, get_comp_basis())
    #     c_sys = csys.CompositeSystem([e_sys])
    #     vec_1 = np.array([1, 0, 0, 0], dtype=np.complex128)
    #     vec_2 = np.array([0, 0, 0, 1], dtype=np.complex128)
    #     vecs = [vec_1, vec_2]
    #     povm = Povm(c_sys=c_sys, vecs=vecs, is_physical=False)

    #     # Act
    #     vec_sizes = povm.e_sys_dims()

    #     # Assert
    #     expected = [4]
    #     assert len(vec_sizes) == len(expected)
    #     for a, b in zip(vec_sizes, expected):
    #         assert a == b


def test_get_x_measurement():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    c_sys1 = csys.CompositeSystem([e_sys1])

    # Act
    actual = get_x_measurement(c_sys1)

    # Assert
    expected = [
        1 / 2 * np.array([1, 1, 1, 1], dtype=np.complex128),
        1 / 2 * np.array([1, -1, -1, 1], dtype=np.complex128),
    ]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys2 = csys.CompositeSystem([e_sys1, e_sys2])
    with pytest.raises(ValueError):
        get_x_measurement(c_sys2)

    # Test that not 2-dim CompositeSystem
    e_sys3 = esys.ElementalSystem(3, get_gell_mann_basis())
    c_sys3 = csys.CompositeSystem([e_sys3])
    with pytest.raises(ValueError):
        get_x_measurement(c_sys3)


def test_get_y_measurement():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    c_sys1 = csys.CompositeSystem([e_sys1])

    # Act
    actual = get_y_measurement(c_sys1)

    # Assert
    expected = [
        1 / 2 * np.array([1, -1j, 1j, 1], dtype=np.complex128),
        1 / 2 * np.array([1, 1j, -1j, 1], dtype=np.complex128),
    ]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys2 = csys.CompositeSystem([e_sys1, e_sys2])
    with pytest.raises(ValueError):
        get_y_measurement(c_sys2)

    # Test that not 2-dim CompositeSystem
    e_sys3 = esys.ElementalSystem(3, get_gell_mann_basis())
    c_sys3 = csys.CompositeSystem([e_sys3])
    with pytest.raises(ValueError):
        get_y_measurement(c_sys3)


def test_get_z_measurement():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    c_sys1 = csys.CompositeSystem([e_sys1])

    # Act
    actual = get_z_measurement(c_sys1)

    # Assert
    expected = [
        np.array([1, 0, 0, 0], dtype=np.complex128),
        np.array([0, 0, 0, 1], dtype=np.complex128),
    ]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys2 = csys.CompositeSystem([e_sys1, e_sys2])
    with pytest.raises(ValueError):
        get_z_measurement(c_sys2)

    # Test that not 2-dim CompositeSystem
    e_sys3 = esys.ElementalSystem(3, get_gell_mann_basis())
    c_sys3 = csys.CompositeSystem([e_sys3])
    with pytest.raises(ValueError):
        get_z_measurement(c_sys3)


def test_get_xx_measurement():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_xx_measurement(c_sys)

    # Assert
    vecs1 = [
        1 / 2 * np.array([1, 1, 1, 1], dtype=np.complex128),
        1 / 2 * np.array([1, -1, -1, 1], dtype=np.complex128),
    ]
    vecs2 = [
        1 / 2 * np.array([1, 1, 1, 1], dtype=np.complex128),
        1 / 2 * np.array([1, -1, -1, 1], dtype=np.complex128),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)


def test_get_xy_measurement():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_xy_measurement(c_sys)

    # Assert
    vecs1 = [
        1 / 2 * np.array([1, 1, 1, 1], dtype=np.complex128),
        1 / 2 * np.array([1, -1, -1, 1], dtype=np.complex128),
    ]
    vecs2 = [
        1 / 2 * np.array([1, -1j, 1j, 1], dtype=np.complex128),
        1 / 2 * np.array([1, 1j, -1j, 1], dtype=np.complex128),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)


def test_get_xz_measurement():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_xz_measurement(c_sys)

    # Assert
    vecs1 = [
        1 / 2 * np.array([1, 1, 1, 1], dtype=np.complex128),
        1 / 2 * np.array([1, -1, -1, 1], dtype=np.complex128),
    ]
    vecs2 = [
        np.array([1, 0, 0, 0], dtype=np.complex128),
        np.array([0, 0, 0, 1], dtype=np.complex128),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)


def test_get_yx_measurement():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_yx_measurement(c_sys)

    # Assert
    vecs1 = [
        1 / 2 * np.array([1, -1j, 1j, 1], dtype=np.complex128),
        1 / 2 * np.array([1, 1j, -1j, 1], dtype=np.complex128),
    ]
    vecs2 = [
        1 / 2 * np.array([1, 1, 1, 1], dtype=np.complex128),
        1 / 2 * np.array([1, -1, -1, 1], dtype=np.complex128),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)


def test_get_yy_measurement():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_yy_measurement(c_sys)

    # Assert
    vecs1 = [
        1 / 2 * np.array([1, -1j, 1j, 1], dtype=np.complex128),
        1 / 2 * np.array([1, 1j, -1j, 1], dtype=np.complex128),
    ]
    vecs2 = [
        1 / 2 * np.array([1, -1j, 1j, 1], dtype=np.complex128),
        1 / 2 * np.array([1, 1j, -1j, 1], dtype=np.complex128),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)


def test_get_yz_measurement():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_yz_measurement(c_sys)

    # Assert
    vecs1 = [
        1 / 2 * np.array([1, -1j, 1j, 1], dtype=np.complex128),
        1 / 2 * np.array([1, 1j, -1j, 1], dtype=np.complex128),
    ]
    vecs2 = [
        np.array([1, 0, 0, 0], dtype=np.complex128),
        np.array([0, 0, 0, 1], dtype=np.complex128),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)


def test_get_zx_measurement():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_zx_measurement(c_sys)

    # Assert
    vecs1 = [
        np.array([1, 0, 0, 0], dtype=np.complex128),
        np.array([0, 0, 0, 1], dtype=np.complex128),
    ]
    vecs2 = [
        1 / 2 * np.array([1, 1, 1, 1], dtype=np.complex128),
        1 / 2 * np.array([1, -1, -1, 1], dtype=np.complex128),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)


def test_get_zy_measurement():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_zy_measurement(c_sys)

    # Assert
    vecs1 = [
        np.array([1, 0, 0, 0], dtype=np.complex128),
        np.array([0, 0, 0, 1], dtype=np.complex128),
    ]
    vecs2 = [
        1 / 2 * np.array([1, -1j, 1j, 1], dtype=np.complex128),
        1 / 2 * np.array([1, 1j, -1j, 1], dtype=np.complex128),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)


def test_get_zz_measurement():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_zz_measurement(c_sys)

    # Assert
    vecs1 = [
        np.array([1, 0, 0, 0], dtype=np.complex128),
        np.array([0, 0, 0, 1], dtype=np.complex128),
    ]
    vecs2 = [
        np.array([1, 0, 0, 0], dtype=np.complex128),
        np.array([0, 0, 0, 1], dtype=np.complex128),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)
