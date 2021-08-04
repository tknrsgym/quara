import numpy as np
import numpy.testing as npt
import pytest


from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.mprocess import MProcess
from quara.objects.mprocess_typical import generate_mprocess_from_name
from quara.settings import Settings


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

    def test_init_is_physicality_required(self):
        e_sys = ElementalSystem(1, matrix_basis.get_comp_basis())
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
        mprocess = generate_mprocess_from_name(c_sys, "z")

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
        mprocess = generate_mprocess_from_name(c_sys, "z")

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
        mprocess = generate_mprocess_from_name(c_sys, "z")

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
        mprocess = generate_mprocess_from_name(c_sys, "z")

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
        mprocess = generate_mprocess_from_name(c_sys, "z")
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
        mprocess = generate_mprocess_from_name(c_sys, "z")
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

    def test_access_random_seed_or_state(self):
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
        assert mprocess.random_seed_or_state == None

        # case 2: random_seed_or_state=1
        mprocess = MProcess(c_sys, hss, mode_sampling=True, random_seed_or_state=1)
        assert mprocess.random_seed_or_state == 1

        # Test that "random_seed_or_state" cannot be updated
        with pytest.raises(AttributeError):
            mprocess.random_seed_or_state = 1

    def test_set_mode_sampling(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z")

        # case 1: mode_sampling=True, random_seed_or_state=None
        mprocess.set_mode_sampling(True)
        assert mprocess.mode_sampling == True
        assert mprocess.random_seed_or_state == None

        # case 2: mode_sampling=True, random_seed_or_state=1
        mprocess.set_mode_sampling(True, random_seed_or_state=1)
        assert mprocess.mode_sampling == True
        assert mprocess.random_seed_or_state == 1

        # case 3: mode_sampling=True -> mode_sampling=False
        mprocess.set_mode_sampling(True, random_seed_or_state=1)
        mprocess.set_mode_sampling(False)
        assert mprocess.mode_sampling == False
        assert mprocess.random_seed_or_state == None

        # case 4: mode_sampling=False, mode_sampling is not None
        with pytest.raises(ValueError):
            mprocess.set_mode_sampling(False, random_seed_or_state=1)

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

    def test_set_zero(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess = generate_mprocess_from_name(c_sys, "z")

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

    def test_add(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        mprocess1 = generate_mprocess_from_name(c_sys, "z")
        mprocess2 = generate_mprocess_from_name(c_sys, "z")

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
        mprocess1 = generate_mprocess_from_name(c_sys, "z")
        mprocess2 = generate_mprocess_from_name(c_sys, "z")

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
        mprocess1 = generate_mprocess_from_name(c_sys, "z")

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
        mprocess1 = generate_mprocess_from_name(c_sys, "z")

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
        mprocess = generate_mprocess_from_name(c_sys, "z")

        # Act
        actual = mprocess.get_basis()

        # Assert
        expected = matrix_basis.get_normalized_pauli_basis()
        for a, e in zip(actual, expected):
            npt.assert_almost_equal(a, e, decimal=15)

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
        npt.assert_almost_equal(choi_0, expected_0, decimal=15)
        npt.assert_almost_equal(choi_1, expected_1, decimal=15)

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
        npt.assert_almost_equal(choi_0, expected_0, decimal=15)
        npt.assert_almost_equal(choi_1, expected_1, decimal=15)

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
