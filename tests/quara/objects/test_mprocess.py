import numpy as np
import numpy.testing as npt
import pytest


from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.mprocess import (
    MProcess,
    convert_var_index_to_mprocess_index,
    convert_mprocess_index_to_var_index,
    convert_hss_to_var,
    convert_var_to_hss,
)
from quara.objects.mprocess_typical import generate_mprocess_from_name
from quara.objects.povm_typical import generate_povm_from_name
from quara.settings import Settings
from quara.utils import matrix_util as mutil


class TestMProcess:
    def test_init_error(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # Test that HS must be square matrix
        hs = np.array(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.float64,
        )
        with pytest.raises(ValueError):
            MProcess(c_sys, [hs])

        # Test that dim of HS must be square number
        hs = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
        with pytest.raises(ValueError):
            MProcess(c_sys, [hs])

        # Test that HS must be real matrix
        hs = np.array(
            [[1j, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.complex128,
        )
        with pytest.raises(ValueError):
            MProcess(c_sys, [hs])

        # Test that dim of HS equals dim of CompositeSystem
        hs = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        with pytest.raises(ValueError):
            MProcess(c_sys, [hs])

        # Test shape
        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]
        with pytest.raises(ValueError):
            MProcess(c_sys, hss, shape=(1,))

        # Test
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])

        # Test that c_sys.is_orthonormal_hermitian_0thprop_identity == False
        hs = np.array(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.float64,
        )
        with pytest.raises(ValueError):
            MProcess(c_sys, [hs])

    def test_init_is_physicality_required(self):
        e_sys = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # gate is not TP
        hs_0 = np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.float64
        )
        hs_1 = np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]], dtype=np.float64
        )
        hs_not_tp = [hs_0, hs_1]
        with pytest.raises(ValueError):
            MProcess(c_sys, hs_not_tp)
        with pytest.raises(ValueError):
            MProcess(c_sys, hs_not_tp, is_physicality_required=True)

        # gate is not CP
        hs_0 = (1 / 2) * np.array(
            [[2, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.float64
        )
        hs_1 = (1 / 2) * np.array(
            [[-1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]],
            dtype=np.float64,
        )
        hs_not_cp = [hs_0, hs_1]
        with pytest.raises(ValueError):
            MProcess(c_sys, hs_not_cp)
        with pytest.raises(ValueError):
            MProcess(c_sys, hs_not_cp, is_physicality_required=True)

        # case: when is_physicality_required is False, it is not happened ValueError
        MProcess(c_sys, hs_not_tp, is_physicality_required=False)
        MProcess(c_sys, hs_not_cp, is_physicality_required=False)

    def test_access_dim(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")

        # Act
        actual = mprocess.dim

        # Assert
        expected = 2
        assert actual == expected

        # Test that "dim" cannot be updated
        with pytest.raises(AttributeError):
            mprocess.dim = 100

    def test_access_num_outcomes(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")

        # Act
        actual = mprocess.num_outcomes

        # Assert
        expected = 2
        assert actual == expected

        # Test that "num_outcomes" cannot be updated
        with pytest.raises(AttributeError):
            mprocess.num_outcomes = 100

    def test_access_hss(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")

        # Act
        actual = mprocess.hss

        # Assert
        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        expected = [hs_0, hs_1]
        for a, e in zip(actual, expected):
            npt.assert_almost_equal(a, e, decimal=15)

        # Test that "num_outcomes" cannot be updated
        with pytest.raises(AttributeError):
            mprocess.num_outcomes = 100

    def test_hs(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # case 1: one-dimensional
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")

        actual = mprocess.hs(0)
        expected = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

        actual = mprocess.hs(1)
        expected = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case 2: multi-dimensional
        hs = np.zeros((4, 4), dtype=np.float64)
        hss = []
        for index in range(6):
            tmp_hs = hs.copy()
            tmp_hs[0][0] = index
            hss.append(tmp_hs)
        mprocess = MProcess(c_sys, hss, shape=(2, 3), is_physicality_required=False)

        assert mprocess.hs((0, 0))[0][0] == 0
        assert mprocess.hs((0, 1))[0][0] == 1
        assert mprocess.hs((0, 2))[0][0] == 2
        assert mprocess.hs((1, 0))[0][0] == 3
        assert mprocess.hs((1, 1))[0][0] == 4
        assert mprocess.hs((1, 2))[0][0] == 5

    def test_access_shape(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # case 1: one-dimensional
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")
        actual = mprocess.shape
        expected = (2,)
        assert actual == expected

        # case 2: multi-dimensional
        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, shape=(1, 2))
        actual = mprocess.shape
        expected = (1, 2)
        assert actual == expected

        # Test that "shape" cannot be updated
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")
        with pytest.raises(AttributeError):
            mprocess.shape = (2,)

    def test_access_mode_sampling(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]

        # case 1: default(False)
        mprocess = MProcess(c_sys, hss)
        assert mprocess.mode_sampling == False

        # case 2: mode_sampling=False
        mprocess = MProcess(c_sys, hss, mode_sampling=False)
        assert mprocess.mode_sampling == False

        # case 3: mode_sampling=True
        mprocess = MProcess(c_sys, hss, mode_sampling=True)
        assert mprocess.mode_sampling == True

        # Test that "mode_sampling" cannot be updated
        with pytest.raises(AttributeError):
            mprocess.mode_sampling = False

    def test_access_random_seed_or_generator(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]

        # case 1: default(None)
        mprocess = MProcess(c_sys, hss)
        assert mprocess.random_seed_or_generator == None

        # case 2: random_seed_or_generator=1
        mprocess = MProcess(c_sys, hss, mode_sampling=True, random_seed_or_generator=1)
        assert mprocess.random_seed_or_generator == 1

        # Test that "random_seed_or_generator" cannot be updated
        with pytest.raises(AttributeError):
            mprocess.random_seed_or_generator = 1

    def test_set_mode_sampling(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")

        # case 1: mode_sampling=True, random_seed_or_generator=None
        mprocess.set_mode_sampling(True)
        assert mprocess.mode_sampling == True
        assert mprocess.random_seed_or_generator == None

        # case 2: mode_sampling=True, random_seed_or_generator=1
        mprocess.set_mode_sampling(True, random_seed_or_generator=1)
        assert mprocess.mode_sampling == True
        assert mprocess.random_seed_or_generator == 1

        # case 3: mode_sampling=True -> mode_sampling=False
        mprocess.set_mode_sampling(True, random_seed_or_generator=1)
        mprocess.set_mode_sampling(False)
        assert mprocess.mode_sampling == False
        assert mprocess.random_seed_or_generator == None

        # case 4: mode_sampling=False, mode_sampling is not None
        with pytest.raises(ValueError):
            mprocess.set_mode_sampling(False, random_seed_or_generator=1)

    def test_access_eps_zero(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]

        # case 1: default(10 ** -8)
        mprocess = MProcess(c_sys, hss)
        assert mprocess.eps_zero == 10 ** -8

        # case 2: eps_zero=1
        mprocess = MProcess(c_sys, hss, eps_zero=10 ** -5)
        assert mprocess.eps_zero == 10 ** -5

        # Test that "eps_zero" cannot be updated
        with pytest.raises(AttributeError):
            mprocess.eps_zero = 1

    def test_is_eq_constraint_satisfied(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # case 1: is_eq_constraint_satisfied=True
        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)
        assert mprocess.is_eq_constraint_satisfied() == True

        # case 2: is_eq_constraint_satisfied=False
        hs_0 = np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.float64
        )
        hs_1 = np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]], dtype=np.float64
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)
        assert mprocess.is_eq_constraint_satisfied() == False

        # case 3: atol=1e-1
        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1.1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)
        assert mprocess.is_eq_constraint_satisfied(atol=1e-1) == True

    def test_is_ineq_constraint_satisfied(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # case 1: is_eq_constraint_satisfied=True
        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)
        assert mprocess.is_ineq_constraint_satisfied() == True

        # case 2: is_eq_constraint_satisfied=False
        hs_0 = (1 / 2) * np.array(
            [[2, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.float64
        )
        hs_1 = (1 / 2) * np.array(
            [[-1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]],
            dtype=np.float64,
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)
        assert mprocess.is_ineq_constraint_satisfied() == False

        # case 3: atol=1e-1
        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1.1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)
        assert mprocess.is_ineq_constraint_satisfied(atol=1e-1) == True

    def test_set_zero(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")

        # Act
        mprocess.set_zero()
        actual = mprocess.hss

        # Assert
        expected = [
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ]
        for a, e in zip(actual, expected):
            npt.assert_almost_equal(a, e, decimal=15)
        assert mprocess.dim == 2
        assert mprocess.shape == (2,)
        assert mprocess.mode_sampling == False
        assert mprocess.is_physicality_required == False
        assert mprocess.is_estimation_object == True
        assert mprocess.on_para_eq_constraint == True
        assert mprocess.on_algo_eq_constraint == True
        assert mprocess.on_algo_ineq_constraint == True
        assert mprocess.eps_proj_physical == Settings.get_atol() / 10.0

    def test_generate_zero_obj(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        z = generate_mprocess_from_name(c_sys, "z-type1")

        # Act
        mprocess = z.generate_zero_obj()
        actual = mprocess.hss

        # Assert
        expected = [
            np.array(
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=np.float64,
            ),
            np.array(
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=np.float64,
            ),
        ]
        for a, e in zip(actual, expected):
            npt.assert_almost_equal(a, e, decimal=15)
        assert mprocess.dim == 2
        assert mprocess.shape == (2,)
        assert mprocess.mode_sampling == False
        assert mprocess.is_physicality_required == False
        assert mprocess.is_estimation_object == False
        assert mprocess.on_para_eq_constraint == True
        assert mprocess.on_algo_eq_constraint == True
        assert mprocess.on_algo_ineq_constraint == True
        assert mprocess.eps_proj_physical == Settings.get_atol() / 10.0

    def test_generate_origin_obj(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        z = generate_mprocess_from_name(c_sys, "z-type1")

        # Act
        mprocess = z.generate_origin_obj()
        actual = mprocess.hss

        # Assert
        expected = [
            np.array(
                [[1 / 2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=np.float64,
            ),
            np.array(
                [[1 / 2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=np.float64,
            ),
        ]
        for a, e in zip(actual, expected):
            npt.assert_almost_equal(a, e, decimal=15)
        assert mprocess.dim == 2
        assert mprocess.shape == (2,)
        assert mprocess.mode_sampling == False
        assert mprocess.is_physicality_required == False
        assert mprocess.is_estimation_object == False
        assert mprocess.on_para_eq_constraint == True
        assert mprocess.on_algo_eq_constraint == True
        assert mprocess.on_algo_ineq_constraint == True
        assert mprocess.eps_proj_physical == Settings.get_atol() / 10.0

    def test_to_var(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]

        # case 1: on_para_eq_constraint=default(True)
        mprocess = MProcess(c_sys, hss)
        actual = mprocess.to_var()
        expected = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        ).flatten()
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case 2: on_para_eq_constraint=True
        mprocess = MProcess(c_sys, hss, on_para_eq_constraint=True)
        actual = mprocess.to_var()
        expected = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        ).flatten()
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case 3: on_para_eq_constraint=False
        mprocess = MProcess(c_sys, hss, on_para_eq_constraint=False)
        actual = mprocess.to_var()
        expected = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 0, -1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        ).flatten()
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_to_stacked_vector(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]
        expected = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 0, -1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        ).flatten()

        # case 1: on_para_eq_constraint=default(True)
        mprocess = MProcess(c_sys, hss)
        actual = mprocess.to_stacked_vector()
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case 2: on_para_eq_constraint=True
        mprocess = MProcess(c_sys, hss, on_para_eq_constraint=True)
        actual = mprocess.to_stacked_vector()
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case 3: on_para_eq_constraint=False
        mprocess = MProcess(c_sys, hss, on_para_eq_constraint=False)
        actual = mprocess.to_stacked_vector()
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_gradient(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]
        expected = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 0, -1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        ).flatten()

        # case 1: on_para_eq_constraint=default(True)
        mprocess = MProcess(c_sys, hss)

        # var_index = 0
        actual = mprocess.calc_gradient(0)
        expected = [
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ]
        expected[0][0][0] = 1
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

        # var_index = 1
        actual = mprocess.calc_gradient(1)
        expected = [
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ]
        expected[0][0][1] = 1
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

        # var_index = 4
        actual = mprocess.calc_gradient(4)
        expected = [
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ]
        expected[0][1][0] = 1
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

        # var_index = 16
        actual = mprocess.calc_gradient(16)
        expected = [
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ]
        expected[1][1][0] = 1
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

        # var_index = 27
        actual = mprocess.calc_gradient(27)
        expected = [
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ]
        expected[1][3][3] = 1
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

        ## case 2: on_para_eq_constraint=True
        mprocess = MProcess(c_sys, hss, on_para_eq_constraint=True)

        # var_index = 0
        actual = mprocess.calc_gradient(0)
        expected = [
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ]
        expected[0][0][0] = 1
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

        # var_index = 1
        actual = mprocess.calc_gradient(1)
        expected = [
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ]
        expected[0][0][1] = 1
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

        # var_index = 4
        actual = mprocess.calc_gradient(4)
        expected = [
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ]
        expected[0][1][0] = 1
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

        # var_index = 16
        actual = mprocess.calc_gradient(16)
        expected = [
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ]
        expected[1][1][0] = 1
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

        # var_index = 27
        actual = mprocess.calc_gradient(27)
        expected = [
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ]
        expected[1][3][3] = 1
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

        ## case 3: on_para_eq_constraint=False
        mprocess = MProcess(c_sys, hss, on_para_eq_constraint=False)

        # var_index = 0
        actual = mprocess.calc_gradient(0)
        expected = [
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ]
        expected[0][0][0] = 1
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

        # var_index = 1
        actual = mprocess.calc_gradient(1)
        expected = [
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ]
        expected[0][0][1] = 1
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

        # var_index = 4
        actual = mprocess.calc_gradient(4)
        expected = [
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ]
        expected[0][1][0] = 1
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

        # var_index = 16
        actual = mprocess.calc_gradient(16)
        expected = [
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ]
        expected[1][0][0] = 1
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

        # var_index = 31
        actual = mprocess.calc_gradient(31)
        expected = [
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ]
        expected[1][3][3] = 1
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

    def test_calc_proj_eq_constraint(self):
        ## case 1: z
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")

        # Act
        actual = mprocess.calc_proj_eq_constraint()

        # Assert
        expected = [
            (1 / 2)
            * np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]),
            (1 / 2)
            * np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]),
        ]
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)
        assert mprocess.dim == 2
        assert mprocess.shape == (2,)
        assert mprocess.mode_sampling == False
        assert mprocess.is_physicality_required == True
        assert mprocess.is_estimation_object == True
        assert mprocess.on_para_eq_constraint == True
        assert mprocess.on_algo_eq_constraint == True
        assert mprocess.on_algo_ineq_constraint == True
        assert mprocess.eps_proj_physical == Settings.get_atol() / 10.0

        ## case 2:
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hss = [
            (1 / 2)
            * np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]),
            (1 / 2)
            * np.array([[1, 1, 1, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]),
        ]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)

        # Act
        actual = mprocess.calc_proj_eq_constraint()

        # Assert
        expected = [
            (1 / 2)
            * np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]),
            (1 / 2)
            * np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]),
        ]
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

    def test_calc_proj_eq_constraint_with_var(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")

        # case 1: on_para_eq_constraint=default(True)
        actual = mprocess.calc_proj_eq_constraint_with_var(c_sys, mprocess.to_var())
        expected = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        ).flatten()
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case 2: on_para_eq_constraint=True
        actual = mprocess.calc_proj_eq_constraint_with_var(
            c_sys, mprocess.to_var(), on_para_eq_constraint=True
        )
        expected = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        ).flatten()
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case 3: on_para_eq_constraint=False
        actual = mprocess.calc_proj_eq_constraint_with_var(
            c_sys, mprocess.to_stacked_vector(), on_para_eq_constraint=False
        )
        expected = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 0, -1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        ).flatten()
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_proj_ineq_constraint(self):
        ## case 1: z
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")

        # Act
        actual = mprocess.calc_proj_ineq_constraint()

        # Assert
        expected = [
            (1 / 2)
            * np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]),
            (1 / 2)
            * np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]),
        ]
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)
        assert mprocess.dim == 2
        assert mprocess.shape == (2,)
        assert mprocess.mode_sampling == False
        assert mprocess.is_physicality_required == True
        assert mprocess.is_estimation_object == True
        assert mprocess.on_para_eq_constraint == True
        assert mprocess.on_algo_eq_constraint == True
        assert mprocess.on_algo_ineq_constraint == True
        assert mprocess.eps_proj_physical == Settings.get_atol() / 10.0

        ## case 2:
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hss = [
            (1 / 2)
            * np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]),
            (1 / 2)
            * np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]),
        ]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)

        # Act
        actual = mprocess.calc_proj_ineq_constraint()

        # Assert
        expected = [
            (1 / 4)
            * np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]),
            (1 / 4)
            * np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]),
        ]
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

    def test_calc_proj_ineq_constraint_with_var(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")

        # case 1: on_para_eq_constraint=default(True)
        actual = mprocess.calc_proj_ineq_constraint_with_var(c_sys, mprocess.to_var())
        expected = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        ).flatten()
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case 2: on_para_eq_constraint=True
        actual = mprocess.calc_proj_ineq_constraint_with_var(
            c_sys, mprocess.to_var(), on_para_eq_constraint=True
        )
        expected = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        ).flatten()
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case 3: on_para_eq_constraint=False
        actual = mprocess.calc_proj_ineq_constraint_with_var(
            c_sys, mprocess.to_stacked_vector(), on_para_eq_constraint=False
        )
        expected = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 0, -1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        ).flatten()
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_generate_from_var(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")

        var = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 0, -1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        ).flatten()

        # Act
        actual = mprocess.generate_from_var(
            var=var,
            is_physicality_required=False,
            is_estimation_object=False,
            on_para_eq_constraint=False,
            on_algo_eq_constraint=False,
            on_algo_ineq_constraint=False,
            mode_proj_order="ineq_eq",
            eps_proj_physical=1e-1,
        )

        # Assert
        assert actual.dim == 2
        assert actual.shape == (2,)
        assert actual.mode_sampling == False
        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == False
        assert actual.on_algo_eq_constraint == False
        assert actual.on_algo_ineq_constraint == False
        assert actual.eps_proj_physical == 1e-1
        hs_0 = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2
        hs_1 = np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]) / 2
        expected = [hs_0, hs_1]
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

    def test_add(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess1 = generate_mprocess_from_name(c_sys, "z-type1")
        mprocess2 = generate_mprocess_from_name(c_sys, "z-type1")

        # Act
        actual = mprocess1 + mprocess2

        # Assert
        hs_0 = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
        hs_1 = np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]])
        expected = [hs_0, hs_1]
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

    def test_sub(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess1 = generate_mprocess_from_name(c_sys, "z-type1")
        mprocess2 = generate_mprocess_from_name(c_sys, "z-type1")

        # Act
        actual = mprocess1 - mprocess2

        # Assert
        hs_0 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        hs_1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        expected = [hs_0, hs_1]
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

    def test_mul_rmul(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess1 = generate_mprocess_from_name(c_sys, "z-type1")

        # Case 1: mul
        # Act
        actual = mprocess1 * 2

        # Assert
        hs_0 = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
        hs_1 = np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]])
        expected = [hs_0, hs_1]
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

        # Case 2: rmul
        # Act
        actual = 2 * mprocess1

        # Assert
        hs_0 = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
        hs_1 = np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]])
        expected = [hs_0, hs_1]
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

    def test_truediv(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess1 = generate_mprocess_from_name(c_sys, "z-type1")

        # Act
        actual = mprocess1 / 2

        # Assert
        hs_0 = (1 / 4) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.float64
        )
        hs_1 = (1 / 4) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]],
            dtype=np.float64,
        )
        expected = [hs_0, hs_1]
        for a, e in zip(actual.hss, expected):
            npt.assert_almost_equal(a, e, decimal=15)

    def test_get_basis(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")

        # Act
        actual = mprocess.get_basis()

        # Assert
        expected = matrix_basis.get_normalized_pauli_basis()
        for a, e in zip(actual, expected):
            # npt.assert_almost_equal(a, e, decimal=15)
            assert mutil.allclose(a, e, atol=1e-15)

    def test_is_sum_tp(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # case 1: is_sum_tp=True
        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)
        assert mprocess.is_sum_tp() == True

        # case 2: is_sum_tp=False
        hs_0 = np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.float64
        )
        hs_1 = np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]], dtype=np.float64
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)
        assert mprocess.is_sum_tp() == False

    def test_is_cp(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # case 1: is_cp=True
        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)
        assert mprocess.is_cp() == True

        # case 2: is_cp=False
        hs_0 = (1 / 2) * np.array(
            [[2, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.float64
        )
        hs_1 = (1 / 2) * np.array(
            [[-1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]],
            dtype=np.float64,
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)
        assert mprocess.is_cp() == False

    def test_convert_basis(self):
        ## case 1 : z
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")

        # Act
        actual = mprocess.convert_basis(matrix_basis.get_comp_basis())

        # Assert
        expected_0 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        expected_1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        npt.assert_almost_equal(actual[0], expected_0, decimal=15)
        npt.assert_almost_equal(actual[1], expected_1, decimal=15)

        ## case 2 : hs of x gate and y gate
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hs_0 = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
        )
        hs_1 = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
            dtype=np.float64,
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)

        # Act
        actual = mprocess.convert_basis(matrix_basis.get_comp_basis())

        # Assert
        expected_0 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
        expected_1 = np.array(
            [[0, 0, 0, 1], [0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]
        )
        npt.assert_almost_equal(actual[0], expected_0, decimal=15)
        npt.assert_almost_equal(actual[1], expected_1, decimal=15)

    def test_convert_to_comp_basis(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")

        # Act
        actual = mprocess.convert_to_comp_basis()

        # Assert
        expected_0 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        expected_1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        npt.assert_almost_equal(actual[0], expected_0, decimal=15)
        npt.assert_almost_equal(actual[1], expected_1, decimal=15)

    def test_to_choi_matrix(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)

        # Act
        choi_0 = mprocess.to_choi_matrix(0)
        choi_1 = mprocess.to_choi_matrix(1)

        # Assert
        expected_0 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        expected_1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        npt.assert_almost_equal(choi_0, expected_0, decimal=15)
        npt.assert_almost_equal(choi_1, expected_1, decimal=15)

    def test_to_choi_matrix(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)

        # Act
        choi_0 = mprocess.to_choi_matrix(0)
        choi_1 = mprocess.to_choi_matrix(1)

        # Assert
        expected_0 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        expected_1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        assert mutil.allclose(choi_0, expected_0, atol=1e-15)
        assert mutil.allclose(choi_1, expected_1, atol=1e-15)

    def test_to_choi_matrix_with_dict(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)

        # Act
        choi_0 = mprocess.to_choi_matrix_with_dict(0)
        choi_1 = mprocess.to_choi_matrix_with_dict(1)

        # Assert
        expected_0 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        expected_1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        print(f"{choi_1}")
        print(f"{expected_1}")
        assert mutil.allclose(choi_0, expected_0, atol=1e-15)
        assert mutil.allclose(choi_1, expected_1, atol=1e-15)

    def test_to_choi_matrix_with_sparsity(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        hs_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        hs_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)

        # Act
        choi_0 = mprocess.to_choi_matrix_with_sparsity(0)
        choi_1 = mprocess.to_choi_matrix_with_sparsity(1)

        # Assert
        expected_0 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        expected_1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        npt.assert_almost_equal(choi_0, expected_0, decimal=15)
        npt.assert_almost_equal(choi_1, expected_1, decimal=15)

    @classmethod
    def calc_sum_of_kraus(cls, kraus):
        # calc \sum_{\alpha} K__{\alpha} K_{\alpha}^{\dagger}
        sum = np.zeros(kraus[0].shape, dtype=np.complex128)
        for matrix in kraus:
            sum += matrix @ matrix.conj().T
        return sum

    def test_to_kraus_matrices(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        eye2 = np.eye(2, dtype=np.complex128)

        hs_0 = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
        )
        hs_1 = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.float64
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)

        # Act
        actual_0 = mprocess.to_kraus_matrices(0)
        actual_1 = mprocess.to_kraus_matrices(1)

        # Assert
        expected_0 = [np.array([[0, 1], [1, 0]], dtype=np.complex128)]
        assert len(actual_0) == 1
        npt.assert_almost_equal(actual_0[0], expected_0[0], decimal=15)
        npt.assert_almost_equal(
            TestMProcess.calc_sum_of_kraus(actual_0), eye2, decimal=14
        )

        expected_1 = [np.array([[0, 1], [-1, 0]], dtype=np.complex128)]
        assert len(actual_1) == 1
        npt.assert_almost_equal(actual_1[0], expected_1[0], decimal=15)
        npt.assert_almost_equal(
            TestMProcess.calc_sum_of_kraus(actual_1), eye2, decimal=14
        )

    def test_to_process_matrix(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        hs_0 = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
        )
        hs_1 = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.float64
        )
        hss = [hs_0, hs_1]
        mprocess = MProcess(c_sys, hss, is_physicality_required=False)

        # Act
        actual_0 = mprocess.to_process_matrix(0)
        actual_1 = mprocess.to_process_matrix(1)

        # Assert
        expected_0 = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        npt.assert_almost_equal(actual_0, expected_0, decimal=15)

        expected_1 = np.array(
            [[0, 0, 0, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]]
        )
        npt.assert_almost_equal(actual_1, expected_1, decimal=15)

    def test_copy(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")

        # Act
        actual = mprocess.copy()

        # Assert
        expected_0 = (1 / 2) * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        )
        expected_1 = (1 / 2) * np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
        )
        npt.assert_almost_equal(actual.hs(0), expected_0, decimal=15)
        npt.assert_almost_equal(actual.hs(1), expected_1, decimal=15)
        assert actual.shape == (2,)
        assert actual.mode_sampling == False
        assert actual.random_seed_or_generator == None

    def test_convert_var_to_stacked_vector(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")
        expected = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 0, -1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        ).flatten()

        # case 1: on_para_eq_constraint=default(True)
        var = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        )

        actual = mprocess.convert_var_to_stacked_vector(c_sys, var.flatten())
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case 2: on_para_eq_constraint=True
        var = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        )

        actual = mprocess.convert_var_to_stacked_vector(
            c_sys, var.flatten(), on_para_eq_constraint=True
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case 3: on_para_eq_constraint=False
        var = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 0, -1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        )

        actual = mprocess.convert_var_to_stacked_vector(
            c_sys, var.flatten(), on_para_eq_constraint=False
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_convert_stacked_vector_to_var(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")
        stacked_vector = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 0, -1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        ).flatten()

        # case 1: on_para_eq_constraint=default(True)
        actual = mprocess.convert_stacked_vector_to_var(c_sys, stacked_vector)
        expected = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        ).flatten()
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case 2: on_para_eq_constraint=True
        actual = mprocess.convert_stacked_vector_to_var(
            c_sys, stacked_vector, on_para_eq_constraint=True
        )
        expected = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        ).flatten()
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case 3: on_para_eq_constraint=False
        actual = mprocess.convert_stacked_vector_to_var(
            c_sys, stacked_vector, on_para_eq_constraint=False
        )
        expected = (1 / 2) * np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 0, -1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 1],
            ]
        ).flatten()
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_to_povm(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z-type1")

        # Act
        actual = mprocess.to_povm()

        # Assert
        expected = generate_povm_from_name("z", c_sys)
        npt.assert_almost_equal(actual.vecs[0], expected.vecs[0], decimal=15)
        npt.assert_almost_equal(actual.vecs[1], expected.vecs[1], decimal=15)


def test_convert_var_index_to_mprocess_index():
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    hs_0 = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    hs_1 = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    hss = [hs_0, hs_1]

    # case 1: on_para_eq_constraint=True
    assert convert_var_index_to_mprocess_index(
        c_sys, hss, 0, on_para_eq_constraint=True
    ) == (0, 0, 0)
    assert convert_var_index_to_mprocess_index(
        c_sys, hss, 1, on_para_eq_constraint=True
    ) == (0, 0, 1)
    assert convert_var_index_to_mprocess_index(
        c_sys, hss, 4, on_para_eq_constraint=True
    ) == (0, 1, 0)
    assert convert_var_index_to_mprocess_index(
        c_sys, hss, 16, on_para_eq_constraint=True
    ) == (1, 1, 0)
    assert convert_var_index_to_mprocess_index(
        c_sys, hss, 27, on_para_eq_constraint=True
    ) == (1, 3, 3)

    # case 2: on_para_eq_constraint=False
    assert convert_var_index_to_mprocess_index(
        c_sys, hss, 0, on_para_eq_constraint=False
    ) == (0, 0, 0)
    assert convert_var_index_to_mprocess_index(
        c_sys, hss, 1, on_para_eq_constraint=False
    ) == (0, 0, 1)
    assert convert_var_index_to_mprocess_index(
        c_sys, hss, 4, on_para_eq_constraint=False
    ) == (0, 1, 0)
    assert convert_var_index_to_mprocess_index(
        c_sys, hss, 16, on_para_eq_constraint=False
    ) == (1, 0, 0)
    assert convert_var_index_to_mprocess_index(
        c_sys, hss, 31, on_para_eq_constraint=False
    ) == (1, 3, 3)


def test_convert_mprocess_index_to_var_index():
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    hs_0 = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    hs_1 = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    hss = [hs_0, hs_1]

    # case 1: on_para_eq_constraint=True
    assert (
        convert_mprocess_index_to_var_index(
            c_sys, (0, 0, 0), hss, on_para_eq_constraint=True
        )
        == 0
    )
    assert (
        convert_mprocess_index_to_var_index(
            c_sys, (0, 0, 1), hss, on_para_eq_constraint=True
        )
        == 1
    )
    assert (
        convert_mprocess_index_to_var_index(
            c_sys, (0, 1, 0), hss, on_para_eq_constraint=True
        )
        == 4
    )
    assert (
        convert_mprocess_index_to_var_index(
            c_sys, (1, 1, 0), hss, on_para_eq_constraint=True
        )
        == 16
    )
    assert (
        convert_mprocess_index_to_var_index(
            c_sys, (1, 3, 3), hss, on_para_eq_constraint=True
        )
        == 27
    )

    # case 2: on_para_eq_constraint=False
    assert (
        convert_mprocess_index_to_var_index(
            c_sys, (0, 0, 0), hss, on_para_eq_constraint=False
        )
        == 0
    )
    assert (
        convert_mprocess_index_to_var_index(
            c_sys, (0, 0, 1), hss, on_para_eq_constraint=False
        )
        == 1
    )
    assert (
        convert_mprocess_index_to_var_index(
            c_sys, (0, 1, 0), hss, on_para_eq_constraint=False
        )
        == 4
    )
    assert (
        convert_mprocess_index_to_var_index(
            c_sys, (1, 0, 0), hss, on_para_eq_constraint=False
        )
        == 16
    )
    assert (
        convert_mprocess_index_to_var_index(
            c_sys, (1, 3, 3), hss, on_para_eq_constraint=False
        )
        == 31
    )


def test_convert_hss_to_var():
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    mprocess = generate_mprocess_from_name(c_sys, "z-type1")

    # case 1: on_para_eq_constraint=default(True)
    actual = convert_hss_to_var(c_sys, mprocess.hss)
    expected = (1 / 2) * np.array(
        [
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-1, 0, 0, 1],
        ]
    )
    npt.assert_almost_equal(actual, expected.flatten(), decimal=15)

    # case 2: on_para_eq_constraint=True
    actual = convert_hss_to_var(c_sys, mprocess.hss, on_para_eq_constraint=True)
    expected = (1 / 2) * np.array(
        [
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-1, 0, 0, 1],
        ]
    )
    npt.assert_almost_equal(actual, expected.flatten(), decimal=15)

    # case 3: on_para_eq_constraint=False
    actual = convert_hss_to_var(c_sys, mprocess.hss, on_para_eq_constraint=False)
    expected = (1 / 2) * np.array(
        [
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 0, -1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-1, 0, 0, 1],
        ]
    )
    npt.assert_almost_equal(actual, expected.flatten(), decimal=15)


def test_convert_var_to_hss():
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    expected = generate_mprocess_from_name(c_sys, "z-type1")

    # case 1: on_para_eq_constraint=default(True)
    var = (1 / 2) * np.array(
        [
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-1, 0, 0, 1],
        ]
    )

    actual = convert_var_to_hss(c_sys, var.flatten())
    npt.assert_almost_equal(actual[0], expected.hs(0), decimal=15)
    npt.assert_almost_equal(actual[1], expected.hs(1), decimal=15)

    # case 2: on_para_eq_constraint=True
    var = (1 / 2) * np.array(
        [
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-1, 0, 0, 1],
        ]
    )

    actual = convert_var_to_hss(c_sys, var.flatten(), on_para_eq_constraint=True)
    npt.assert_almost_equal(actual[0], expected.hs(0), decimal=15)
    npt.assert_almost_equal(actual[1], expected.hs(1), decimal=15)

    # case 3: on_para_eq_constraint=False
    var = (1 / 2) * np.array(
        [
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 0, -1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-1, 0, 0, 1],
        ]
    )

    actual = convert_var_to_hss(c_sys, var.flatten(), on_para_eq_constraint=False)
    npt.assert_almost_equal(actual[0], expected.hs(0), decimal=15)
    npt.assert_almost_equal(actual[1], expected.hs(1), decimal=15)
