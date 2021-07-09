import itertools
import os
from pathlib import Path
import copy

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
    convert_vec,
)
from quara.objects.operators import tensor_product
from quara.objects.povm import (
    Povm,
    convert_var_index_to_povm_index,
    convert_povm_index_to_var_index,
    convert_var_to_povm,
    convert_vecs_to_var,
    calc_gradient_from_povm,
    get_x_povm,
    get_xx_povm,
    get_xy_povm,
    get_xz_povm,
    get_y_povm,
    get_yx_povm,
    get_yy_povm,
    get_yz_povm,
    get_z_povm,
    get_zx_povm,
    get_zy_povm,
    get_zz_povm,
)
from quara.objects.state import get_x0_1q
from quara.settings import Settings


class TestPovm:
    def test_validate_dtype_ng(self):
        p1 = np.array([1, 0, 0, 0], dtype=np.complex128)
        p2 = np.array([0, 0, 0, 1], dtype=np.complex128)
        vecs = [p1, p2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # entries of vecs are not real numbers
        with pytest.raises(ValueError):
            Povm(c_sys=c_sys, vecs=vecs)

    def test_validate_set_of_hermitian_matrices_ok(self):
        # Arrange
        p1 = np.array([1, 0, 0, 0], dtype=np.float64)
        p2 = np.array([0, 0, 0, 1], dtype=np.float64)
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
        p1 = np.array([1, 0, 0, 0], dtype=np.float64)
        p2 = np.array([0, 1, 0, 0], dtype=np.float64)
        vecs = [p1, p2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        with pytest.raises(ValueError):
            # ValueError: povm must be a set of Hermitian matrices
            _ = Povm(c_sys=c_sys, vecs=vecs)

    def test_validate_set_of_hermitian_matrices_not_physical_ok(self):
        # Arrange
        p1 = np.array([1, 0, 0, 0], dtype=np.float64)
        p2 = np.array([0, 1, 0, 0], dtype=np.float64)
        vecs = [p1, p2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        # Test that no exceptions are raised.
        _ = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

    def test_validate_sum_is_identity_sum_ok(self):
        # Arrange
        p1 = np.array([1, 0, 0, 0], dtype=np.float64)
        p2 = np.array([0, 0, 0, 1], dtype=np.float64)
        vecs = [p1, p2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)
        actual = povm.is_identity_sum()

        # Assert
        assert actual is True

    def test_validate_sum_is_identity_sum_ng(self):
        # Arrange
        p1 = np.array([1, 0, 0, 0], dtype=np.float64)
        p2 = np.array([0, 1, 0, 0], dtype=np.float64)
        vecs = [p1, p2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        with pytest.raises(ValueError):
            # ValueError: The sum of the elements of POVM must be an identity matrix.
            _ = Povm(c_sys=c_sys, vecs=vecs)

    def test_validate_sum_is_identity_sum_not_physical_ok(self):
        # Arrange
        p1 = np.array([1, 0, 0, 1], dtype=np.float64)
        p2 = np.array([1, 0, 0, 1], dtype=np.float64)
        vecs = [p1, p2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        # Test that no exceptions are raised.
        _ = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

    def test_validate_is_positive_semidefinite_ok(self):
        # Arrange
        ps_1 = np.array([1, 0, 0, 0], dtype=np.float64)
        ps_2 = np.array([0, 0, 0, 1], dtype=np.float64)
        vecs = [ps_1, ps_2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)
        actual = povm.is_positive_semidefinite()

        # Assert
        assert actual is True

        # Act
        actual = povm.is_ineq_constraint_satisfied()
        # Assert
        assert actual is True

    def test_validate_is_positive_semidefinite_ng(self):
        # Arrange
        ps = np.array([1, 0, 0, 2], dtype=np.float64)
        not_ps = np.array([[0, 0, 0, -1]], dtype=np.float64)
        vecs = [ps, not_ps]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        with pytest.raises(ValueError):
            _ = Povm(c_sys=c_sys, vecs=vecs)

    def test_validate_is_positive_semidefinite_not_physical_ok(self):
        # Arrange
        ps = np.array([1, 0, 0, 2], dtype=np.float64)
        not_ps = np.array([0, 0, 0, -1], dtype=np.float64)
        vecs = [ps, not_ps]

        e_sys = esys.ElementalSystem(1, get_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act & Assert
        # Test that no exceptions are raised.
        povm = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)
        actual = povm.is_positive_semidefinite()

        # Assert
        assert actual is False

        # Act
        actual = povm.is_ineq_constraint_satisfied()
        # Assert
        assert actual is False

    def test_calc_eigenvalues_all(self):
        # Arrange
        vec_1 = np.array([1, 0, 0, 0], dtype=np.float64)
        vec_2 = np.array([0, 0, 0, 1], dtype=np.float64)
        vecs = [vec_1, vec_2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)
        actual = povm.calc_eigenvalues()

        # Assert
        expected = [
            np.array([1, 0], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
        ]

        assert len(actual) == len(expected)
        npt.assert_almost_equal(actual[0], expected[0], decimal=15)
        npt.assert_almost_equal(actual[1], expected[1], decimal=15)

    def test_calc_eigenvalues_one(self):
        # Arrange
        vec_1 = np.array([1, 0, 0, 0], dtype=np.float64)
        vec_2 = np.array([0, 0, 0, 1], dtype=np.float64)
        vecs = [vec_1, vec_2]

        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)
        actual = povm.calc_eigenvalues(0)

        # Assert
        expected = np.array([1, 0], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Act
        povm = Povm(c_sys=c_sys, vecs=vecs)
        actual = povm.calc_eigenvalues(1)

        # Assert
        expected = np.array([1, 0], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

    # def test_validate_dim_ng(self):
    #     # Arrange
    #     test_root_dir = Path(os.path.dirname(__file__)).parent.parent
    #     data_dir = test_root_dir / "data"

    #     dim = 2 ** 2  # 2 qubits
    #     num_state = 16
    #     num_povm = 9
    #     num_outcome = 4

    #     povms = s_io.load_povm_list(
    #         data_dir / "tester_2qubit_povm.csv",
    #         dim=dim,
    #         num_povm=num_povm,
    #         num_outcome=num_outcome,
    #     )
    #     vecs = list(povms[0])  # 2qubit

    #     e_sys = esys.ElementalSystem(1, get_pauli_basis())  # 1qubit
    #     c_sys = csys.CompositeSystem([e_sys])

    #     # Act & Assert
    #     with pytest.raises(ValueError):
    #         _ = Povm(c_sys=c_sys, vecs=vecs)

    def test_convert_basis(self):
        # Arrange
        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])
        ps_1 = np.array([1, 0, 0, 0], dtype=np.float64)
        ps_2 = np.array([0, 0, 0, 1], dtype=np.float64)
        vecs = [ps_1, ps_2]
        povm = Povm(c_sys=c_sys, vecs=vecs)
        to_basis = get_normalized_pauli_basis()

        # Act
        actual = povm.convert_basis(to_basis)

        # Assert
        expected = [
            1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.float64),
            1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.float64),
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
        povm1 = Povm(c_sys1, vecs1, is_physicality_required=False)

        # Act
        actual = povm1.nums_local_outcomes

        # Assert
        expected = [2]
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a == e
        assert povm1.num_outcomes == 2

        # Case 2:
        # Act
        povm1._nums_local_outcomes = [1, 2]
        actual = povm1.nums_local_outcomes
        # Assert
        expected = [1, 2]
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a == e
        assert povm1.num_outcomes == 2

    def test_vec(self):
        # Case 1:
        # Arrange
        basis1 = get_comp_basis()
        e_sys1 = esys.ElementalSystem(1, basis1)
        c_sys1 = csys.CompositeSystem([e_sys1])
        vecs1 = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]
        povm1 = Povm(c_sys1, vecs1, is_physicality_required=False)

        # Act
        actual0 = povm1.vec(0)
        actual1 = povm1.vec(1)
        actual2 = povm1.vec((0))
        actual3 = povm1.vec((1))

        # Assert
        assert np.all(actual0 == vecs1[0])
        assert np.all(actual1 == vecs1[1])
        assert np.all(actual2 == vecs1[0])
        assert np.all(actual3 == vecs1[1])

        # Case2: type of argument is tuple
        # Arrange
        basis2 = get_comp_basis()
        e_sys2 = esys.ElementalSystem(2, basis2)
        c_sys2 = csys.CompositeSystem([e_sys2])
        vecs2 = [
            np.array([23, 29, 31, 37], dtype=np.float64),
            np.array([41, 43, 47, 53], dtype=np.float64),
        ]
        povm2 = Povm(c_sys2, vecs2, is_physicality_required=False)
        povm12 = tensor_product(povm1, povm2)

        # Act
        actual = [
            povm12.vec((0, 0)),
            povm12.vec((0, 1)),
            povm12.vec((1, 0)),
            povm12.vec((1, 1)),
        ]

        # Assert
        expected = [
            np.kron(vec1, vec2)
            for vec1, vec2 in itertools.product(povm1.vecs, povm2.vecs)
        ]
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert np.all(a == e)

        # Case3: type of argument is int
        # Act
        actual = [
            povm12.vec(0),
            povm12.vec(1),
            povm12.vec(2),
            povm12.vec(3),
        ]

        # Assert
        expected = [
            np.kron(vec1, vec2)
            for vec1, vec2 in itertools.product(povm1.vecs, povm2.vecs)
        ]
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert np.all(a == e)

    def test_vec_unexpected(self):
        # Arrange
        basis1 = get_comp_basis()
        e_sys1 = esys.ElementalSystem(1, basis1)
        c_sys1 = csys.CompositeSystem([e_sys1])
        vecs1 = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]
        povm1 = Povm(c_sys1, vecs1, is_physicality_required=False)

        # Case 1:
        # Act & Assert
        with pytest.raises(ValueError):
            # ValueError: length of tuple does not equal length of the list of measurements.
            _ = povm1.vec((0, 0))

        # Case 2:
        # Act & Assert
        with pytest.raises(IndexError):
            # IndexError: specified index does not exist in the list of measurements.
            _ = povm1.vec(2)

    def test_is_physical(self):
        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        p1 = np.array([1, 0, 0, 0], dtype=np.float64)
        p2 = np.array([0, 0, 0, 1], dtype=np.float64)
        povm = Povm(c_sys=c_sys, vecs=[p1, p2])
        assert povm.is_physical() == True

        p1 = np.array([1, 0, 0, 2], dtype=np.float64)
        p2 = np.array([0, 0, 0, -1], dtype=np.float64)
        povm = Povm(c_sys=c_sys, vecs=[p1, p2], is_physicality_required=False)
        assert povm.is_physical() == False

        p1 = np.array([1, 0, 0, 1], dtype=np.float64)
        p2 = np.array([1, 0, 0, 1], dtype=np.float64)
        povm = Povm(c_sys=c_sys, vecs=[p1, p2], is_physicality_required=False)
        assert povm.is_physical() == False

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
        povm1 = Povm(c_sys1, vecs1, is_physicality_required=False)

        # Act
        actual = povm1.matrix(0)
        # Assert
        expected = povm1.matrices()
        npt.assert_almost_equal(actual, expected[0], decimal=15)

        # Act
        actual = povm1.matrix(1)
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
        povm2 = Povm(c_sys2, vecs2, is_physicality_required=False)
        povm12 = tensor_product(povm1, povm2)

        # Act
        actual = povm12.matrix((0, 0))
        # Assert
        expected = povm12.matrices()
        npt.assert_almost_equal(actual, expected[0], decimal=15)

        # Act
        actual = povm12.matrix((0, 1))
        # Assert
        npt.assert_almost_equal(actual, expected[1], decimal=15)

        # Act
        actual = povm12.matrix((1, 0))
        # Assert
        npt.assert_almost_equal(actual, expected[2], decimal=15)

        # Act
        actual = povm12.matrix((1, 1))
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
        povm1 = Povm(c_sys1, vecs1, is_physicality_required=False)

        # Act & Assert
        unexpected_type = [0]
        with pytest.raises(TypeError):
            # TypeError: The type of `key` must be int or str.
            _ = povm1.matrix(unexpected_type)

    def test_to_var(self):
        # Arrange
        e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])
        vecs = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]

        # default
        povm = Povm(c_sys, vecs, is_physicality_required=False)

        # Act
        actual = povm.to_var()

        # Assert
        expected = np.array([2, 3, 5, 7], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Arrange
        vecs = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]
        povm = Povm(
            c_sys, vecs, is_physicality_required=False, on_para_eq_constraint=True
        )

        # Actual
        actual = povm.to_var()

        # Assert
        expected = np.array([2, 3, 5, 7], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Arrange
        vecs = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]
        povm = Povm(
            c_sys, vecs, is_physicality_required=False, on_para_eq_constraint=False
        )

        # Actual
        actual = povm.to_var()

        # Assert
        expected = np.array([2, 3, 5, 7, 11, 13, 17, 19], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_generate_from_var(self):
        # Arrange
        e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])

        vecs = [
            1 / np.sqrt(2) * np.array([1, 1, 0, 0], dtype=np.float64),
            1 / np.sqrt(2) * np.array([1, -1, 0, 0], dtype=np.float64),
        ]

        to_vecs = [
            convert_vec(vec, get_normalized_pauli_basis(), c_sys.basis()).real.astype(
                np.float64
            )
            for vec in vecs
        ]

        init_is_physicality_required = False
        init_is_estimation_object = True
        init_on_para_eq_constraint = False
        init_on_algo_eq_constraint = True
        init_on_algo_ineq_constraint = False
        init_eps_proj_physical = 10 ** (-3)
        source_povm = Povm(
            c_sys,
            vecs=to_vecs,
            is_physicality_required=init_is_physicality_required,
            is_estimation_object=init_is_estimation_object,
            on_para_eq_constraint=init_on_para_eq_constraint,
            on_algo_eq_constraint=init_on_algo_eq_constraint,
            on_algo_ineq_constraint=init_on_algo_ineq_constraint,
            eps_proj_physical=init_eps_proj_physical,
        )

        # Case 1: default
        var = np.array([2, 3, 5, 7, 11, 13, 17, 19], dtype=np.float64)
        # Act
        actual = source_povm.generate_from_var(var)
        # Assert
        expected = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]
        assert len(actual.vecs) == len(expected)
        for a, e in zip(actual.vecs, expected):
            npt.assert_almost_equal(a, e, decimal=15)
        assert actual._composite_system is c_sys
        assert actual.is_physicality_required is init_is_physicality_required
        assert actual.is_estimation_object is init_is_estimation_object
        assert actual.on_para_eq_constraint is init_on_para_eq_constraint
        assert actual.on_algo_eq_constraint is init_on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is init_on_algo_ineq_constraint
        assert actual.eps_proj_physical is init_eps_proj_physical

        # Case 2:
        with pytest.raises(ValueError):
            # ValueError: the POVM is not physically correct.
            _ = source_povm.generate_from_var(var, is_physicality_required=True)

        # Case 3:
        # Arrange
        # var = np.array([2, 3, 5, 7], dtype=np.float64)
        # source_is_estimation_object = False
        # source_on_para_eq_constraint = True
        # source_on_algo_eq_constraint = False
        # source_on_algo_ineq_constraint = True
        # source_eps_proj_physical = 10 ** (-2)

        # Act
        # actual = source_povm.generate_from_var(
        #     var,
        #     is_estimation_object=source_is_estimation_object,
        #     on_para_eq_constraint=source_on_para_eq_constraint,
        #     on_algo_eq_constraint=source_on_algo_eq_constraint,
        #     on_algo_ineq_constraint=source_on_algo_ineq_constraint,
        #     eps_proj_physical=source_eps_proj_physical,
        # )

        # Assert
        # expected = [
        #     np.array([2, 3, 5, 7], dtype=np.float64),
        #     np.array([-1, -3, -5, -6], dtype=np.float64),
        # ]
        # assert len(actual.vecs) == len(expected)
        # for a, e in zip(actual.vecs, expected):
        #     npt.assert_almost_equal(a, e, decimal=15)
        # assert actual._composite_system is c_sys
        # assert actual.is_physicality_required is init_is_physicality_required
        # assert actual.is_estimation_object is source_is_estimation_object
        # assert actual.on_para_eq_constraint is source_on_para_eq_constraint
        # assert actual.on_algo_eq_constraint is source_on_algo_eq_constraint
        # assert actual.on_algo_ineq_constraint is source_on_algo_ineq_constraint
        # assert actual.eps_proj_physical == source_eps_proj_physical

        # Case 4:
        # Array
        e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])

        a0, a1, a2, a3 = 1, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)
        m1 = (1 / np.sqrt(2)) * np.array([a0, a1, a2, a3])
        m2 = (1 / np.sqrt(2)) * np.array([2 - a0, -a1, -a2, -a3])
        source_is_estimation_object = False
        source_on_para_eq_constraint = True
        source_on_algo_eq_constraint = False
        source_on_algo_ineq_constraint = True
        source_eps_proj_physical = 10 ** (-2)

        source_povm = Povm(
            vecs=[m1, m2],
            c_sys=c_sys,
            is_estimation_object=source_is_estimation_object,
            on_para_eq_constraint=source_on_para_eq_constraint,
            on_algo_eq_constraint=source_on_algo_eq_constraint,
            on_algo_ineq_constraint=source_on_algo_ineq_constraint,
            eps_proj_physical=source_eps_proj_physical,
        )
        var = source_povm.to_var()
        source_is_estimation_object = False
        source_on_para_eq_constraint = True
        source_on_algo_eq_constraint = False
        source_on_algo_ineq_constraint = True
        source_eps_proj_physical = 10 ** (-2)

        # Act
        actual = source_povm.generate_from_var(
            var,
            is_estimation_object=source_is_estimation_object,
            on_para_eq_constraint=source_on_para_eq_constraint,
            on_algo_eq_constraint=source_on_algo_eq_constraint,
            on_algo_ineq_constraint=source_on_algo_ineq_constraint,
            eps_proj_physical=source_eps_proj_physical,
        )

        expected = source_povm.vecs
        assert len(actual.vecs) == len(expected)
        for a, e in zip(actual.vecs, expected):
            npt.assert_almost_equal(a, e, decimal=15)
        assert actual._composite_system is c_sys
        assert actual.is_physicality_required is True  # default
        assert actual.is_estimation_object is source_is_estimation_object
        assert actual.on_para_eq_constraint is source_on_para_eq_constraint
        assert actual.on_algo_eq_constraint is source_on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is source_on_algo_ineq_constraint
        assert actual.eps_proj_physical == source_eps_proj_physical

    def test_set_zero(self):
        # Arrange
        e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])
        povm = get_x_povm(c_sys)
        old_povm = copy.copy(povm)
        # Act
        povm.set_zero()

        assert len(povm.vecs) == len(old_povm.vecs)
        for actual, old in zip(povm.vecs, old_povm.vecs):
            assert actual.size == old.size
            expected = np.zeros(old.size, dtype=np.float64)
            npt.assert_almost_equal(actual, expected, decimal=15)

        assert povm.dim == old_povm.dim
        assert povm.is_physicality_required == False
        assert povm.is_estimation_object == old_povm.is_estimation_object
        assert povm.on_para_eq_constraint == old_povm.on_para_eq_constraint
        assert povm.on_algo_eq_constraint == old_povm.on_algo_eq_constraint
        assert povm.on_algo_ineq_constraint == old_povm.on_algo_ineq_constraint
        assert povm.eps_proj_physical == old_povm.eps_proj_physical

    def test_generate_origin_obj(self):
        # generate_origin_obj()
        # Arrange
        e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])

        vec_1 = np.array([1, 2, 3, 4], dtype=np.float64)
        vec_2 = np.array([5, 6, 7, 8], dtype=np.float64)
        vecs = [vec_1, vec_2]
        povm = Povm(
            c_sys=c_sys,
            vecs=vecs,
            is_physicality_required=False,
            is_estimation_object=True,
            on_para_eq_constraint=False,
            on_algo_eq_constraint=True,
            on_algo_ineq_constraint=False,
            eps_proj_physical=0.2,
        )

        # Act
        actual = povm.generate_origin_obj()

        expected_vecs = [
            np.array([np.sqrt(2) / 2, 0, 0, 0], dtype=np.float64),
            np.array([np.sqrt(2) / 2, 0, 0, 0], dtype=np.float64),
        ]
        assert len(actual.vecs) == len(expected_vecs)
        for a, e in zip(actual.vecs, expected_vecs):
            npt.assert_almost_equal(a, e, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == povm.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == povm.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == povm.on_algo_ineq_constraint
        assert actual.eps_proj_physical == povm.eps_proj_physical

    def test_add(self):
        # Arrange
        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        vec_11 = np.array([1, 2, 3, 4], dtype=np.float64)
        vec_12 = np.array([5, 6, 7, 8], dtype=np.float64)
        vecs = [vec_11, vec_12]
        povm_1 = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        vec_21 = np.array([10, 20, 30, 40], dtype=np.float64)
        vec_22 = np.array([50, 60, 70, 80], dtype=np.float64)
        vecs = [vec_21, vec_22]
        povm_2 = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Act
        actual = povm_1 + povm_2

        # Assert
        expected_vecs = [
            np.array([11, 22, 33, 44], dtype=np.float64),
            np.array([55, 66, 77, 88], dtype=np.float64),
        ]
        assert type(actual) == Povm
        assert len(actual.vecs) == len(expected_vecs)
        for a, e in zip(actual.vecs, expected_vecs):
            npt.assert_almost_equal(a, e, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == povm_1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == povm_1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == povm_1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == povm_1.eps_proj_physical

    def test_add_exception(self):
        # Arrange
        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        vec_11 = np.array([1, 2, 3, 4], dtype=np.float64)
        vec_12 = np.array([5, 6, 7, 8], dtype=np.float64)
        vecs = [vec_11, vec_12]
        povm_1 = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Case 1:
        # Arrange
        state = get_x0_1q(c_sys)

        # Act & Assert
        with pytest.raises(TypeError):
            _ = povm_1 + state

        # Case 2:
        # Arrange
        vec_21 = np.array([10, 20, 30, 40], dtype=np.float64)
        vec_22 = np.array([50, 60, 70, 80], dtype=np.float64)
        vecs = [vec_21, vec_22]
        povm_2 = Povm(
            c_sys=c_sys,
            vecs=vecs,
            is_physicality_required=False,
            on_para_eq_constraint=False,
        )

        # Act & Assert
        with pytest.raises(ValueError):
            _ = povm_1 + povm_2

        # Case 3:
        # Arrange
        vec_21 = np.array([10, 20, 30, 40], dtype=np.float64)
        vec_22 = np.array([50, 60, 70, 80], dtype=np.float64)
        vecs = [vec_21, vec_22, vec_21]
        povm_2 = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Act & Assert
        with pytest.raises(ValueError):
            _ = povm_1 + povm_2

        # Case 4:
        # Arrange
        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])
        vec_31 = np.array([10, 20, 30, 40], dtype=np.float64)
        vec_32 = np.array([50, 60, 70, 80], dtype=np.float64)
        vecs = [vec_31, vec_32]
        povm_3 = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Act & Assert
        with pytest.raises(ValueError):
            actual = povm_1 + povm_3

    def test_sub(self):
        # Arrange
        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        vec_11 = np.array([1, 2, 3, 4], dtype=np.float64)
        vec_12 = np.array([5, 6, 7, 8], dtype=np.float64)
        vecs = [vec_11, vec_12]
        povm_1 = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        vec_21 = np.array([10, 20, 30, 40], dtype=np.float64)
        vec_22 = np.array([50, 60, 70, 80], dtype=np.float64)
        vecs = [vec_21, vec_22]
        povm_2 = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Act
        actual = povm_1 - povm_2

        # Assert
        expected_vecs = [
            np.array([-9, -18, -27, -36], dtype=np.float64),
            np.array([-45, -54, -63, -72], dtype=np.float64),
        ]
        assert type(actual) == Povm
        assert len(actual.vecs) == len(expected_vecs)
        for a, e in zip(actual.vecs, expected_vecs):
            npt.assert_almost_equal(a, e, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == povm_1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == povm_1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == povm_1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == povm_1.eps_proj_physical

    def test_sub_exception(self):
        # Arrange
        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        vec_11 = np.array([1, 2, 3, 4], dtype=np.float64)
        vec_12 = np.array([5, 6, 7, 8], dtype=np.float64)
        vecs = [vec_11, vec_12]
        povm_1 = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Case 1:
        # Arrange
        state = get_x0_1q(c_sys)

        # Act & Assert
        with pytest.raises(TypeError):
            _ = povm_1 - state

        # Case 2:
        # Arrange
        vec_21 = np.array([10, 20, 30, 40], dtype=np.float64)
        vec_22 = np.array([50, 60, 70, 80], dtype=np.float64)
        vecs = [vec_21, vec_22]
        povm_2 = Povm(
            c_sys=c_sys,
            vecs=vecs,
            is_physicality_required=False,
            on_para_eq_constraint=False,
        )

        # Act & Assert
        with pytest.raises(ValueError):
            _ = povm_1 - povm_2

        # Case 3:
        # Arrange
        vec_21 = np.array([10, 20, 30, 40], dtype=np.float64)
        vec_22 = np.array([50, 60, 70, 80], dtype=np.float64)
        vecs = [vec_21, vec_22, vec_21]
        povm_2 = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Act & Assert
        with pytest.raises(ValueError):
            _ = povm_1 - povm_2

        # Case 4:
        # Arrange
        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])
        vec_31 = np.array([10, 20, 30, 40], dtype=np.float64)
        vec_32 = np.array([50, 60, 70, 80], dtype=np.float64)
        vecs = [vec_31, vec_32]
        povm_3 = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Act & Assert
        with pytest.raises(ValueError):
            actual = povm_1 - povm_3

    def test_mul(self):
        # Arrange
        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        vec_11 = np.array([1, 2, 3, 4], dtype=np.float64)
        vec_12 = np.array([5, 6, 7, 8], dtype=np.float64)
        vecs = [vec_11, vec_12]
        povm_1 = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Case 1:
        # Act
        actual = povm_1 * 10

        # Assert
        expected_vecs = [
            np.array([10, 20, 30, 40], dtype=np.float64),
            np.array([50, 60, 70, 80], dtype=np.float64),
        ]
        assert type(actual) == Povm
        assert len(actual.vecs) == len(expected_vecs)
        for a, e in zip(actual.vecs, expected_vecs):
            npt.assert_almost_equal(a, e, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == povm_1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == povm_1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == povm_1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == povm_1.eps_proj_physical

        # Case 2:
        # Act
        actual = povm_1 * 0.1

        # Assert
        expected_vecs = [
            np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
            np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64),
        ]
        assert type(actual) == Povm
        assert len(actual.vecs) == len(expected_vecs)
        for a, e in zip(actual.vecs, expected_vecs):
            npt.assert_almost_equal(a, e, decimal=15)
        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == povm_1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == povm_1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == povm_1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == povm_1.eps_proj_physical

        # Case 3: Exception
        # Act & Assert
        with pytest.raises(TypeError):
            _ = povm_1 * povm_1

    def test_rmul(self):
        # Arrange
        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        vec_11 = np.array([1, 2, 3, 4], dtype=np.float64)
        vec_12 = np.array([5, 6, 7, 8], dtype=np.float64)
        vecs = [vec_11, vec_12]
        povm_1 = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Case 1:
        # Act
        actual = 10 * povm_1

        # Assert
        expected_vecs = [
            np.array([10, 20, 30, 40], dtype=np.float64),
            np.array([50, 60, 70, 80], dtype=np.float64),
        ]
        assert type(actual) == Povm
        assert len(actual.vecs) == len(expected_vecs)
        for a, e in zip(actual.vecs, expected_vecs):
            npt.assert_almost_equal(a, e, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == povm_1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == povm_1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == povm_1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == povm_1.eps_proj_physical

        # Case 2:
        # Act
        actual = 0.1 * povm_1

        # Assert
        expected_vecs = [
            np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
            np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64),
        ]
        assert type(actual) == Povm
        assert len(actual.vecs) == len(expected_vecs)
        for a, e in zip(actual.vecs, expected_vecs):
            npt.assert_almost_equal(a, e, decimal=15)
        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == povm_1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == povm_1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == povm_1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == povm_1.eps_proj_physical

        # Case 3: Exception
        # Act & Assert
        with pytest.raises(TypeError):
            _ = povm_1 * povm_1

    def test_truediv(self):
        # Arrange
        e_sys = esys.ElementalSystem(1, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])

        vec_11 = np.array([1, 2, 3, 4], dtype=np.float64)
        vec_12 = np.array([5, 6, 7, 8], dtype=np.float64)
        vecs = [vec_11, vec_12]
        povm_1 = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Case 1:
        # Act
        actual = povm_1 / 10

        # Assert
        expected_vecs = [
            np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
            np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64),
        ]
        assert type(actual) == Povm
        assert len(actual.vecs) == len(expected_vecs)
        for a, e in zip(actual.vecs, expected_vecs):
            npt.assert_almost_equal(a, e, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == povm_1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == povm_1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == povm_1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == povm_1.eps_proj_physical

        # Case 2:
        # Act
        actual = povm_1 / 0.1

        # Assert
        expected_vecs = [
            np.array([10, 20, 30, 40], dtype=np.float64),
            np.array([50, 60, 70, 80], dtype=np.float64),
        ]
        assert type(actual) == Povm
        assert len(actual.vecs) == len(expected_vecs)
        for a, e in zip(actual.vecs, expected_vecs):
            npt.assert_almost_equal(a, e, decimal=15)
        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == povm_1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == povm_1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == povm_1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == povm_1.eps_proj_physical

        # Case 4:
        # Act
        actual = povm_1 / 0

        # Assert
        expected_vecs = [
            np.array(
                [float("inf"), float("inf"), float("inf"), float("inf")],
                dtype=np.float64,
            ),
            np.array(
                [float("inf"), float("inf"), float("inf"), float("inf")],
                dtype=np.float64,
            ),
        ]
        assert type(actual) == Povm
        assert len(actual.vecs) == len(expected_vecs)
        for a, e in zip(actual.vecs, expected_vecs):
            npt.assert_almost_equal(a, e, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == povm_1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == povm_1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == povm_1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == povm_1.eps_proj_physical

        # Case 3: Exception
        # Act & Assert
        with pytest.raises(TypeError):
            _ = povm_1 / povm_1

        # Case 4: Exception
        # Act & Assert
        with pytest.raises(TypeError):
            _ = 1 / povm_1

    def test_calc_proj_eq_constraint(self):
        # Case 1:
        e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])
        povm = get_x_povm(c_sys)

        # Act
        actual = povm.calc_proj_eq_constraint()

        vec_1 = (
            povm.vecs[0]
            - (1 / 2) * np.array([2 / np.sqrt(2), 0, 0, 0])
            + np.array([np.sqrt(2) / 2, 0, 0, 0])
        )
        vec_2 = (
            povm.vecs[1]
            - (1 / 2) * np.array([2 / np.sqrt(2), 0, 0, 0])
            + np.array([np.sqrt(2) / 2, 0, 0, 0])
        )
        vecs = [vec_1, vec_2]
        expected_vecs = [vec_1, vec_2]
        assert len(actual.vecs) == len(expected_vecs)
        for a, e in zip(actual.vecs, expected_vecs):
            npt.assert_almost_equal(a, e, decimal=15)

    def test_calc_proj_eq_constraint_with_var(self):
        e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])
        povm = get_x_povm(c_sys)

        vec_1 = (
            povm.vecs[0]
            - (1 / 2) * np.array([2 / np.sqrt(2), 0, 0, 0])
            + np.array([np.sqrt(2) / 2, 0, 0, 0])
        )
        vec_2 = (
            povm.vecs[1]
            - (1 / 2) * np.array([2 / np.sqrt(2), 0, 0, 0])
            + np.array([np.sqrt(2) / 2, 0, 0, 0])
        )
        vecs = [vec_1, vec_2]

        # Case 1: default
        actual = povm.calc_proj_eq_constraint_with_var(c_sys, vec_1)
        expected = vec_1
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 2: on_para_eq_constraint=True
        actual = povm.calc_proj_eq_constraint_with_var(
            c_sys, vec_1, on_para_eq_constraint=True
        )
        expected = vec_1
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 3: on_para_eq_constraint=False
        actual = povm.calc_proj_eq_constraint_with_var(
            c_sys, np.hstack(vecs), on_para_eq_constraint=False
        )
        expected = np.hstack(vecs)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_proj_ineq_constraint(self):
        # Case 1:
        e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])
        povm = get_x_povm(c_sys)

        # Act
        actual = povm.calc_proj_ineq_constraint()

        vec_1 = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0])
        vec_2 = np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0, 0])

        vecs = [vec_1, vec_2]
        expected_vecs = [vec_1, vec_2]
        assert len(actual.vecs) == len(expected_vecs)
        for a, e in zip(actual.vecs, expected_vecs):
            npt.assert_almost_equal(a, e, decimal=15)

    def test_calc_proj_ineq_constraint_with_var(self):
        e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])
        povm = get_x_povm(c_sys)

        vec_1 = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0])
        vec_2 = np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0, 0])
        vecs = [vec_1, vec_2]

        # Case 1: default
        actual = povm.calc_proj_ineq_constraint_with_var(c_sys, vec_1)
        expected = vec_1
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 2: on_para_eq_constraint=True
        actual = povm.calc_proj_ineq_constraint_with_var(
            c_sys, vec_1, on_para_eq_constraint=True
        )
        expected = vec_1
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 3: on_para_eq_constraint=False
        actual = povm.calc_proj_ineq_constraint_with_var(
            c_sys, np.hstack(vecs), on_para_eq_constraint=False
        )
        expected = np.hstack(vecs)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_gradient(self):
        # Arrange
        e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Case 1: default
        # Arrange
        vecs = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]
        povm = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Act
        actual = povm.calc_gradient(3)

        # Assert
        expected_vecs = [
            np.array([0, 0, 0, 1], dtype=np.float64),
            np.array([0, 0, 0, 0], dtype=np.float64),
        ]
        assert len(actual.vecs) == len(expected_vecs)
        for a, e in zip(actual.vecs, expected_vecs):
            npt.assert_almost_equal(a, e, decimal=15)

        assert actual.composite_system is povm.composite_system
        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == True
        assert actual.on_para_eq_constraint == True
        assert actual.on_algo_eq_constraint == True
        assert actual.on_algo_ineq_constraint == True
        assert actual.eps_proj_physical == Settings.get_atol() / 10.0

        # Case 2:
        # Arrange
        vecs = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]
        povm = Povm(
            c_sys=c_sys,
            vecs=vecs,
            is_physicality_required=False,
            on_para_eq_constraint=False,
        )

        # Act
        actual = povm.calc_gradient(7)
        # Assert
        expected_vecs = [
            np.array([0, 0, 0, 0], dtype=np.float64),
            np.array([0, 0, 0, 1], dtype=np.float64),
        ]
        assert len(actual.vecs) == len(expected_vecs)
        for a, e in zip(actual.vecs, expected_vecs):
            npt.assert_almost_equal(a, e, decimal=15)
        assert actual.composite_system is povm.composite_system

    def test_to_stacked_vector(self):
        # Arrange
        e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])
        vecs = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]
        povm = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Act
        actual = povm.to_stacked_vector()

        # Assert
        expected = np.array([2, 3, 5, 7, 11, 13, 17, 19])
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_proj_eq_constraint_unexpected(self):
        # Array
        e_sys = esys.ElementalSystem(0, get_comp_basis())
        c_sys = csys.CompositeSystem([e_sys])
        m_1 = (1 / 2) * np.array([1, 0, 0, 1])
        m_2 = (1 / 2) * np.array([1, 0, 0, 1])

        vecs = [m_1, m_2]
        povm = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Act & Assert
        with pytest.raises(ValueError):
            _ = povm.calc_proj_eq_constraint()

    def test_calc_proj(self):
        # Array
        e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])

        # Case 1
        # ">=": True "=I": True
        # Array
        m_1 = np.array([1 / np.sqrt(2), 0, 0, 0])
        m_2 = np.array([1 / np.sqrt(2), 0, 0, 0])

        vecs = [m_1, m_2]
        povm = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Make sure the prerequisites are met
        assert povm.is_positive_semidefinite() is True
        assert povm.is_identity_sum() is True

        # Act
        actual_ineq = povm.calc_proj_ineq_constraint()
        actual_eq = povm.calc_proj_eq_constraint()

        # Assert
        assert actual_ineq.is_positive_semidefinite() is True
        npt.assert_almost_equal(povm.vecs, actual_ineq.vecs, decimal=15)

        assert actual_eq.is_identity_sum() is True
        npt.assert_almost_equal(povm.vecs, actual_eq.vecs, decimal=15)

        # Case 2
        # ">=": False "=I": True
        # Array
        m_1 = np.array([np.sqrt(2), 0.0, 0.0, np.sqrt(2)])
        m_2 = np.array([0, 0.0, 0.0, -np.sqrt(2)])

        vecs = [m_1, m_2]
        povm = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Make sure the prerequisites are met
        assert povm.is_positive_semidefinite() is False
        assert povm.is_identity_sum() is True

        # Act
        actual_ineq = povm.calc_proj_ineq_constraint()
        actual_eq = povm.calc_proj_eq_constraint()

        # Assert
        assert actual_ineq.is_positive_semidefinite() is True
        assert not np.allclose(povm.vecs, actual_ineq.vecs)

        assert actual_eq.is_identity_sum() is True
        npt.assert_almost_equal(povm.vecs, actual_eq.vecs, decimal=15)

        # Case 3
        # ">=": True "=I": False
        # Array
        m_1 = np.array([np.sqrt(2), 0.0, 0.0, 1.0])
        m_2 = np.array([np.sqrt(2), 0.0, 0.0, 1.0])

        vecs = [m_1, m_2]
        povm = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Make sure the prerequisites are met
        assert povm.is_positive_semidefinite() is True
        assert povm.is_identity_sum() is False

        # Act
        actual_ineq = povm.calc_proj_ineq_constraint()
        actual_eq = povm.calc_proj_eq_constraint()

        # Assert
        assert actual_ineq.is_positive_semidefinite() is True
        npt.assert_almost_equal(povm.vecs, actual_ineq.vecs, decimal=15)

        assert actual_eq.is_identity_sum() is True
        assert not np.allclose(povm.vecs, actual_eq.vecs)

        # Case 4
        # ">=": False "=I": False
        # Array
        m_1 = np.array([np.sqrt(2), 0.0, 0.0, 1.0])
        m_2 = np.array([-np.sqrt(2), 0.0, 0.0, 1.0])

        vecs = [m_1, m_2]
        povm = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)

        # Make sure the prerequisites are met
        assert povm.is_positive_semidefinite() is False
        assert povm.is_identity_sum() is False

        # Act
        actual_ineq = povm.calc_proj_ineq_constraint()
        actual_eq = povm.calc_proj_eq_constraint()

        # Assert
        assert actual_ineq.is_positive_semidefinite() is True
        assert not np.allclose(povm.vecs, actual_ineq.vecs)

        assert actual_eq.is_identity_sum() is True
        assert not np.allclose(povm.vecs, actual_eq.vecs)

    def test_convert_var_to_stacked_vector(self):
        e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])
        povm = get_z_povm(c_sys)

        # Case 1: default
        vecs = [
            np.array([1, 0, 0, 1] / np.sqrt(2), dtype=np.float64),
            np.array([1, 0, 0, -1] / np.sqrt(2), dtype=np.float64),
        ]
        actual = povm.convert_var_to_stacked_vector(c_sys, vecs[0])
        npt.assert_almost_equal(actual, np.hstack(vecs), decimal=15)

        # Case 2: on_para_eq_constraint=True
        vecs = [
            np.array([1, 0, 0, 1] / np.sqrt(2), dtype=np.float64),
            np.array([1, 0, 0, -1] / np.sqrt(2), dtype=np.float64),
        ]
        actual = povm.convert_var_to_stacked_vector(
            c_sys, vecs[0], on_para_eq_constraint=True
        )
        npt.assert_almost_equal(actual, np.hstack(vecs), decimal=15)

        # Case 3: on_para_eq_constraint=False
        vecs = [
            np.array([1, 0, 0, 1] / np.sqrt(2), dtype=np.float64),
            np.array([1, 0, 0, -1] / np.sqrt(2), dtype=np.float64),
        ]
        actual = povm.convert_var_to_stacked_vector(
            c_sys, np.hstack(vecs), on_para_eq_constraint=False
        )
        npt.assert_almost_equal(actual, np.hstack(vecs), decimal=15)

    def test_convert_stacked_vector_to_var(self):
        e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = csys.CompositeSystem([e_sys])
        povm = get_z_povm(c_sys)

        # Case 1: default
        vecs = [
            np.array([1, 0, 0, 1] / np.sqrt(2), dtype=np.float64),
            np.array([1, 0, 0, -1] / np.sqrt(2), dtype=np.float64),
        ]
        actual = povm.convert_stacked_vector_to_var(c_sys, np.hstack(vecs))
        expected = np.array([1, 0, 0, 1] / np.sqrt(2), dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 2: on_para_eq_constraint=True
        vecs = [
            np.array([1, 0, 0, 1] / np.sqrt(2), dtype=np.float64),
            np.array([1, 0, 0, -1] / np.sqrt(2), dtype=np.float64),
        ]
        actual = povm.convert_stacked_vector_to_var(
            c_sys, np.hstack(vecs), on_para_eq_constraint=True
        )
        expected = np.array([1, 0, 0, 1] / np.sqrt(2), dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 3: on_para_eq_constraint=False
        vecs = [
            np.array([1, 0, 0, 1] / np.sqrt(2), dtype=np.float64),
            np.array([1, 0, 0, -1] / np.sqrt(2), dtype=np.float64),
        ]
        actual = povm.convert_stacked_vector_to_var(
            c_sys, np.hstack(vecs), on_para_eq_constraint=False
        )
        expected = np.array([1, 0, 0, 1, 1, 0, 0, -1] / np.sqrt(2), dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)


def test_convert_var_index_to_povm_index():
    # Arrange
    e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = csys.CompositeSystem([e_sys])
    vecs = [
        np.array([2, 3, 5, 7], dtype=np.float64),
        np.array([11, 13, 17, 19], dtype=np.float64),
    ]

    # default
    actual = convert_var_index_to_povm_index(c_sys, vecs, 3)
    assert actual == (0, 3)

    # on_para_eq_constraint=True
    actual = convert_var_index_to_povm_index(c_sys, vecs, 3, on_para_eq_constraint=True)
    assert actual == (0, 3)

    # on_para_eq_constraint=False
    actual = convert_var_index_to_povm_index(
        c_sys, vecs, 7, on_para_eq_constraint=False
    )
    assert actual == (1, 3)


def test_convert_povm_index_to_var_index():
    # Arrange
    e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = csys.CompositeSystem([e_sys])
    vecs = [
        np.array([2, 3, 5, 7], dtype=np.float64),
        np.array([11, 13, 17, 19], dtype=np.float64),
    ]

    # default
    actual = convert_povm_index_to_var_index(c_sys, vecs, (0, 3))
    assert actual == 3

    # on_para_eq_constraint=True
    actual = convert_povm_index_to_var_index(
        c_sys, vecs, (0, 3), on_para_eq_constraint=True
    )
    assert actual == 3

    # on_para_eq_constraint=False
    actual = convert_povm_index_to_var_index(
        c_sys, vecs, (1, 3), on_para_eq_constraint=False
    )
    assert actual == 7


def test_convert_var_to_povm():
    # Arrange
    e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = csys.CompositeSystem([e_sys])

    # Case 1: default
    # Arrange
    # vecs = np.array([2, 3, 5, 7], dtype=np.float64)
    # # Act
    # actual = convert_var_to_povm(c_sys, vecs, is_physicality_required=False)
    # # Assert
    # expected = [
    #     np.array([2, 3, 5, 7], dtype=np.float64),
    #     np.array([-1, -3, -5, -6], dtype=np.float64),
    # ]
    # assert len(actual.vecs) == len(expected)
    # for a, e in zip(actual.vecs, expected):
    #     npt.assert_almost_equal(a, e, decimal=15)

    # Case 2: on_para_eq_constraint=True
    # Arrange
    # vecs = np.array([2, 3, 5, 7], dtype=np.float64)
    # # Act
    # actual = convert_var_to_povm(
    #     c_sys, vecs, on_para_eq_constraint=True, is_physicality_required=False
    # )
    # # Assert
    # expected = [
    #     np.array([2, 3, 5, 7], dtype=np.float64),
    #     np.array([-1, -3, -5, -6], dtype=np.float64),
    # ]
    # assert len(actual.vecs) == len(expected)
    # for a, e in zip(actual.vecs, expected):
    #     npt.assert_almost_equal(a, e, decimal=15)

    # Case 3: on_para_eq_constraint=False
    # Arrange
    vecs = np.array([2, 3, 5, 7, 11, 13, 17, 19], dtype=np.float64)
    # Act
    actual = convert_var_to_povm(
        c_sys, vecs, on_para_eq_constraint=False, is_physicality_required=False
    )
    # Assert
    expected = [
        np.array([2, 3, 5, 7], dtype=np.float64),
        np.array([11, 13, 17, 19], dtype=np.float64),
    ]
    assert len(actual.vecs) == len(expected)
    for a, e in zip(actual.vecs, expected):
        npt.assert_almost_equal(a, e, decimal=15)

    # Case 4:
    e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = csys.CompositeSystem([e_sys])

    a0, a1, a2, a3 = 1, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)
    m1 = (1 / np.sqrt(2)) * np.array([a0, a1, a2, a3])
    m2 = (1 / np.sqrt(2)) * np.array([2 - a0, -a1, -a2, -a3])

    true_object = Povm(vecs=[m1, m2], c_sys=c_sys)
    true_var = true_object.to_var()

    actual = convert_var_to_povm(c_sys, true_var, is_physicality_required=False)
    expected = true_object.vecs
    assert len(actual.vecs) == len(expected)
    for a, e in zip(actual.vecs, expected):
        npt.assert_almost_equal(a, e, decimal=15)


def test_convert_vecs_to_var():
    # Arrange
    e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = csys.CompositeSystem([e_sys])

    # Case 1: default
    vecs = [
        np.array([2, 3, 5, 7], dtype=np.float64),
        np.array([11, 13, 17, 19], dtype=np.float64),
    ]

    # Act
    actual = convert_vecs_to_var(c_sys, vecs)

    # Assert
    expected = np.array([2, 3, 5, 7], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case 2: on_para_eq_constraint=True
    vecs = [
        np.array([2, 3, 5, 7], dtype=np.float64),
        np.array([11, 13, 17, 19], dtype=np.float64),
    ]

    # Act
    actual = convert_vecs_to_var(c_sys, vecs, on_para_eq_constraint=True)

    # Assert
    expected = np.array([2, 3, 5, 7], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case 3: on_para_eq_constraint=False
    vecs = [
        np.array([2, 3, 5, 7], dtype=np.float64),
        np.array([11, 13, 17, 19], dtype=np.float64),
    ]

    # Act
    actual = convert_vecs_to_var(c_sys, vecs, on_para_eq_constraint=False)

    # Assert
    expected = np.array([2, 3, 5, 7, 11, 13, 17, 19], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_calc_gradient_from_povm():
    # Arrange
    e_sys = esys.ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = csys.CompositeSystem([e_sys])

    # default
    vecs = [
        np.array([2, 3, 5, 7], dtype=np.float64),
        np.array([11, 13, 17, 19], dtype=np.float64),
    ]
    actual = calc_gradient_from_povm(c_sys, vecs, 3)
    expected = [
        np.array([0, 0, 0, 1], dtype=np.float64),
        np.array([0, 0, 0, 0], dtype=np.float64),
    ]
    assert len(actual.vecs) == len(expected)
    for a, e in zip(actual.vecs, expected):
        npt.assert_almost_equal(a, e, decimal=15)

    # on_para_eq_constraint=True
    vecs = [
        np.array([2, 3, 5, 7], dtype=np.float64),
        np.array([11, 13, 17, 19], dtype=np.float64),
    ]
    actual = calc_gradient_from_povm(c_sys, vecs, 3, on_para_eq_constraint=True)
    expected = [
        np.array([0, 0, 0, 1], dtype=np.float64),
        np.array([0, 0, 0, 0], dtype=np.float64),
    ]
    assert len(actual.vecs) == len(expected)
    for a, e in zip(actual.vecs, expected):
        npt.assert_almost_equal(a, e, decimal=15)

    # on_para_eq_constraint=False
    vecs = [
        np.array([2, 3, 5, 7], dtype=np.float64),
        np.array([11, 13, 17, 19], dtype=np.float64),
    ]
    actual = calc_gradient_from_povm(c_sys, vecs, 7, on_para_eq_constraint=False)
    expected = [
        np.array([0, 0, 0, 0], dtype=np.float64),
        np.array([0, 0, 0, 1], dtype=np.float64),
    ]
    assert len(actual.vecs) == len(expected)
    for a, e in zip(actual.vecs, expected):
        npt.assert_almost_equal(a, e, decimal=15)


def test_get_x_povm():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    c_sys1 = csys.CompositeSystem([e_sys1])

    # Act
    actual = get_x_povm(c_sys1)

    # Assert
    expected = [
        1 / 2 * np.array([1, 1, 1, 1], dtype=np.float64),
        1 / 2 * np.array([1, -1, -1, 1], dtype=np.float64),
    ]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys2 = csys.CompositeSystem([e_sys1, e_sys2])
    with pytest.raises(ValueError):
        get_x_povm(c_sys2)

    # Test that not 2-dim CompositeSystem
    e_sys3 = esys.ElementalSystem(3, get_gell_mann_basis())
    c_sys3 = csys.CompositeSystem([e_sys3])
    with pytest.raises(ValueError):
        get_x_povm(c_sys3)


def test_get_y_povm():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_normalized_pauli_basis())
    c_sys1 = csys.CompositeSystem([e_sys1])

    # Act
    actual = get_y_povm(c_sys1)

    # Assert
    expected = [
        1 / np.sqrt(2) * np.array([1, 0, 1, 0], dtype=np.float64),
        1 / np.sqrt(2) * np.array([1, 0, -1, 0], dtype=np.float64),
    ]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys2 = csys.CompositeSystem([e_sys1, e_sys2])
    with pytest.raises(ValueError):
        get_y_povm(c_sys2)

    # Test that not 2-dim CompositeSystem
    e_sys3 = esys.ElementalSystem(3, get_gell_mann_basis())
    c_sys3 = csys.CompositeSystem([e_sys3])
    with pytest.raises(ValueError):
        get_y_povm(c_sys3)


def test_get_z_povm():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    c_sys1 = csys.CompositeSystem([e_sys1])

    # Act
    actual = get_z_povm(c_sys1)

    # Assert
    expected = [
        np.array([1, 0, 0, 0], dtype=np.float64),
        np.array([0, 0, 0, 1], dtype=np.float64),
    ]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys2 = csys.CompositeSystem([e_sys1, e_sys2])
    with pytest.raises(ValueError):
        get_z_povm(c_sys2)

    # Test that not 2-dim CompositeSystem
    e_sys3 = esys.ElementalSystem(3, get_gell_mann_basis())
    c_sys3 = csys.CompositeSystem([e_sys3])
    with pytest.raises(ValueError):
        get_z_povm(c_sys3)


def test_get_xx_povm():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_xx_povm(c_sys)

    # Assert
    vecs1 = [
        1 / 2 * np.array([1, 1, 1, 1], dtype=np.float64),
        1 / 2 * np.array([1, -1, -1, 1], dtype=np.float64),
    ]
    vecs2 = [
        1 / 2 * np.array([1, 1, 1, 1], dtype=np.float64),
        1 / 2 * np.array([1, -1, -1, 1], dtype=np.float64),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)


def test_get_xy_povm():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    e_sys2 = esys.ElementalSystem(2, get_normalized_pauli_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_xy_povm(c_sys)

    # Assert
    vecs1 = [
        1 / 2 * np.array([1, 1, 1, 1], dtype=np.float64),
        1 / 2 * np.array([1, -1, -1, 1], dtype=np.float64),
    ]
    vecs2 = [
        1 / np.sqrt(2) * np.array([1, 0, 1, 0], dtype=np.float64),
        1 / np.sqrt(2) * np.array([1, 0, -1, 0], dtype=np.float64),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)


def test_get_xz_povm():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_xz_povm(c_sys)

    # Assert
    vecs1 = [
        1 / 2 * np.array([1, 1, 1, 1], dtype=np.float64),
        1 / 2 * np.array([1, -1, -1, 1], dtype=np.float64),
    ]
    vecs2 = [
        np.array([1, 0, 0, 0], dtype=np.float64),
        np.array([0, 0, 0, 1], dtype=np.float64),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)


def test_get_yx_povm():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_normalized_pauli_basis())
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_yx_povm(c_sys)

    # Assert
    vecs1 = [
        1 / np.sqrt(2) * np.array([1, 0, 1, 0], dtype=np.float64),
        1 / np.sqrt(2) * np.array([1, 0, -1, 0], dtype=np.float64),
    ]
    vecs2 = [
        1 / 2 * np.array([1, 1, 1, 1], dtype=np.float64),
        1 / 2 * np.array([1, -1, -1, 1], dtype=np.float64),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)


def test_get_yy_povm():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_normalized_pauli_basis())
    e_sys2 = esys.ElementalSystem(2, get_normalized_pauli_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_yy_povm(c_sys)

    # Assert
    vecs1 = [
        1 / np.sqrt(2) * np.array([1, 0, 1, 0], dtype=np.float64),
        1 / np.sqrt(2) * np.array([1, 0, -1, 0], dtype=np.float64),
    ]
    vecs2 = [
        1 / np.sqrt(2) * np.array([1, 0, 1, 0], dtype=np.float64),
        1 / np.sqrt(2) * np.array([1, 0, -1, 0], dtype=np.float64),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)


def test_get_yz_povm():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_normalized_pauli_basis())
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_yz_povm(c_sys)

    # Assert
    vecs1 = [
        1 / np.sqrt(2) * np.array([1, 0, 1, 0], dtype=np.float64),
        1 / np.sqrt(2) * np.array([1, 0, -1, 0], dtype=np.float64),
    ]
    vecs2 = [
        np.array([1, 0, 0, 0], dtype=np.float64),
        np.array([0, 0, 0, 1], dtype=np.float64),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)


def test_get_zx_povm():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_zx_povm(c_sys)

    # Assert
    vecs1 = [
        np.array([1, 0, 0, 0], dtype=np.float64),
        np.array([0, 0, 0, 1], dtype=np.float64),
    ]
    vecs2 = [
        1 / 2 * np.array([1, 1, 1, 1], dtype=np.float64),
        1 / 2 * np.array([1, -1, -1, 1], dtype=np.float64),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)


def test_get_zy_povm():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_normalized_pauli_basis())
    e_sys2 = esys.ElementalSystem(2, get_normalized_pauli_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_zy_povm(c_sys)

    # Assert
    vecs1 = [
        1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.float64),
        1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.float64),
    ]
    vecs2 = [
        1 / np.sqrt(2) * np.array([1, 0, 1, 0], dtype=np.float64),
        1 / np.sqrt(2) * np.array([1, 0, -1, 0], dtype=np.float64),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)


def test_get_zz_povm():
    # Arrange
    e_sys1 = esys.ElementalSystem(1, get_comp_basis())
    e_sys2 = esys.ElementalSystem(2, get_comp_basis())
    c_sys = csys.CompositeSystem([e_sys1, e_sys2])

    # Act
    actual = get_zz_povm(c_sys)

    # Assert
    vecs1 = [
        np.array([1, 0, 0, 0], dtype=np.float64),
        np.array([0, 0, 0, 1], dtype=np.float64),
    ]
    vecs2 = [
        np.array([1, 0, 0, 0], dtype=np.float64),
        np.array([0, 0, 0, 1], dtype=np.float64),
    ]
    expected = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(actual.vecs) == len(expected)
    for i, a in enumerate(actual):
        npt.assert_almost_equal(a, expected[i], decimal=15)


class TestPovmImmutable:
    def test_deney_update_vecs(self):
        # Arrange
        basis = get_comp_basis()
        e_sys = esys.ElementalSystem(1, basis)
        c_sys = csys.CompositeSystem([e_sys])
        vec_0 = np.array([2, 3, 5, 7], dtype=np.float64)
        vec_1 = np.array([11, 13, 17, 19], dtype=np.float64)
        source_vecs = [vec_0, vec_1]
        povm = Povm(c_sys, source_vecs, is_physicality_required=False)
        assert id(source_vecs) != id(povm.vecs)

        # Case 1
        # If "source_vec" is updated, the data in POVM is not updated
        # Act
        source_vecs[0] = np.zeros([2, 2], dtype=np.float64)

        # Assert
        expected = np.array([2, 3, 5, 7], dtype=np.float64)
        assert np.array_equal(povm.vecs[0], expected)
        assert np.array_equal(povm[0], expected)

        # Case 2
        # If "vec_0" is updated, the data in POVM is not updated
        # Act
        vec_0[0] = 100

        # Assert
        assert np.array_equal(povm.vecs[0], expected)
        assert np.array_equal(povm[0], expected)

    def test_deney_update_povm_item(self):
        # Arrange
        basis = get_comp_basis()
        e_sys = esys.ElementalSystem(1, basis)
        c_sys = csys.CompositeSystem([e_sys])
        vec_0 = np.array([2, 3, 5, 7], dtype=np.float64)
        vec_1 = np.array([11, 13, 17, 19], dtype=np.float64)
        source_vecs = [vec_0, vec_1]
        povm = Povm(c_sys, source_vecs, is_physicality_required=False)

        expected = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]

        # Act & Assert
        with pytest.raises(TypeError):
            # TypeError: 'Povm' object does not support item assignment
            povm[0] = np.array([100, 100, 100, 100], dtype=np.float64)
        assert len(povm.vecs) == len(expected)
        for actual, e in zip(povm.vecs, expected):
            assert np.array_equal(actual, e)

        # Act & Assert
        with pytest.raises(TypeError):
            # TypeError: 'tuple' object does not support item assignment
            povm.vecs[0] = np.array([100, 100, 100, 100], dtype=np.float64)
        assert len(povm.vecs) == len(expected)
        for actual, e in zip(povm.vecs, expected):
            assert np.array_equal(actual, e)

        # Act & Assert
        with pytest.raises(ValueError):
            # ValueError: assignment destination is read-only
            povm.vecs[0][0] = 100
        assert len(povm.vecs) == len(expected)
        for actual, e in zip(povm.vecs, expected):
            assert np.array_equal(actual, e)

        # Test to ensure that no copies are made on each access
        first_access = id(povm[0])
        second_access = id(povm[0])
        assert first_access == second_access
