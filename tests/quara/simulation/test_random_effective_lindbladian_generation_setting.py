import numpy as np
import numpy.testing as npt
import pytest


from quara.simulation.random_effective_lindbladian_generation_setting import (
    RandomEffectiveLindbladianGenerationSetting,
)
from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.effective_lindbladian import EffectiveLindbladian
from quara.objects.gate import get_x
from quara.objects.mprocess_typical import generate_mprocess_from_name
from quara.objects.state import get_z0_1q
from quara.objects.povm import get_z_povm


class TestRandomEffectiveLindbladianGenerationSetting:
    def test_access_is_seed_or_generator_required(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        qoperation_base = get_z0_1q(c_sys)
        hs = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        lindbladian_base = EffectiveLindbladian(c_sys, hs)

        # Act
        actual = RandomEffectiveLindbladianGenerationSetting(
            c_sys, qoperation_base, lindbladian_base, 1.0, 2.0
        )

        # Assert
        assert actual.is_seed_or_generator_required == True

    def test_access_strength_h_part(self):
        ## init
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        qoperation_base = get_z0_1q(c_sys)
        hs = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        lindbladian_base = EffectiveLindbladian(c_sys, hs)

        # Act
        actual = RandomEffectiveLindbladianGenerationSetting(
            c_sys, qoperation_base, lindbladian_base, 1.0, 2.0
        )

        # Assert
        assert actual.strength_h_part == 1.0

        ## Test that "strength_h_part" cannot be updated
        with pytest.raises(AttributeError):
            actual.strength_h_part = 1.0

    def test_access_strength_k_part(self):
        ## init
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        qoperation_base = get_z0_1q(c_sys)
        hs = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        lindbladian_base = EffectiveLindbladian(c_sys, hs)

        # Act
        actual = RandomEffectiveLindbladianGenerationSetting(
            c_sys, qoperation_base, lindbladian_base, 1.0, 2.0
        )

        # Assert
        assert actual.strength_k_part == 2.0

        ## Test that "strength_k_part" cannot be updated
        with pytest.raises(AttributeError):
            actual.strength_k_part = 2.0

    def test_generate_random_effective_lindbladian_h_part(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        qoperation_base = get_z0_1q(c_sys)
        hs = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        lindbladian_base = EffectiveLindbladian(c_sys, hs)
        generation_setting = RandomEffectiveLindbladianGenerationSetting(
            c_sys, qoperation_base, lindbladian_base, 1.0, 2.0
        )

        # Act
        (
            random_h_part,
            random_variables,
        ) = generation_setting.generate_random_effective_lindbladian_h_part()

        # Assert
        assert random_h_part.shape == (4, 4)
        assert random_h_part[0, 0] == 0
        assert random_h_part[0, 3] == 0
        assert random_h_part[1, 1].real == 0
        assert random_h_part[1, 2] == 0
        assert random_h_part[2, 1] == 0
        assert random_h_part[2, 2].real == 0
        assert random_h_part[3, 0] == 0
        assert random_h_part[3, 3] == 0
        assert len(random_variables) == 3

    def test_generate_random_effective_lindbladian_d_part(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        qoperation_base = get_z0_1q(c_sys)
        hs = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        lindbladian_base = EffectiveLindbladian(c_sys, hs)
        generation_setting = RandomEffectiveLindbladianGenerationSetting(
            c_sys, qoperation_base, lindbladian_base, 1.0, 2.0
        )

        # Act
        (
            random_d_part,
            random_variables,
            random_unitary,
        ) = generation_setting.generate_random_effective_lindbladian_d_part()

        # Assert
        assert random_d_part.shape == (4, 4)
        assert len(random_variables) == 3
        assert random_unitary.shape == (3, 3)
        uni = random_unitary @ random_unitary.T.conj()
        npt.assert_almost_equal(uni, np.eye(3), decimal=15)

    def test_generate_random_effective_lindbladian(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        qoperation_base = get_z0_1q(c_sys)
        hs = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        lindbladian_base = EffectiveLindbladian(c_sys, hs)
        generation_setting = RandomEffectiveLindbladianGenerationSetting(
            c_sys, qoperation_base, lindbladian_base, 1.0, 2.0
        )

        # Act
        (
            el,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        ) = generation_setting.generate_random_effective_lindbladian()

        # Assert
        assert el.hs.shape == (4, 4)
        assert el.is_physical() == True
        assert len(random_variables_h_part) == 3
        assert len(random_variables_k_part) == 3
        assert random_unitary.shape == (3, 3)
        uni = random_unitary @ random_unitary.T.conj()
        npt.assert_almost_equal(uni, np.eye(3), decimal=15)
        assert random_el.shape == (4, 4)

    def test_generate_state(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        qoperation_base = get_z0_1q(c_sys)
        hs = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        lindbladian_base = EffectiveLindbladian(c_sys, hs)
        generation_setting = RandomEffectiveLindbladianGenerationSetting(
            c_sys, qoperation_base, lindbladian_base, 1.0, 2.0
        )

        ### generate_state
        # Act
        (
            state,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        ) = generation_setting.generate_state()

        # Assert
        assert len(state.vec) == 4
        assert state.is_physical() == True
        assert len(random_variables_h_part) == 3
        assert len(random_variables_k_part) == 3
        assert random_unitary.shape == (3, 3)
        uni = random_unitary @ random_unitary.T.conj()
        npt.assert_almost_equal(uni, np.eye(3), decimal=15)
        assert random_el.shape == (4, 4)

        ### generate
        # Act
        (
            state,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        ) = generation_setting.generate()

        # Assert
        assert len(state.vec) == 4
        assert state.is_physical() == True
        assert len(random_variables_h_part) == 3
        assert len(random_variables_k_part) == 3
        assert random_unitary.shape == (3, 3)
        uni = random_unitary @ random_unitary.T.conj()
        npt.assert_almost_equal(uni, np.eye(3), decimal=15)
        assert random_el.shape == (4, 4)

    def test_stream(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        qoperation_base = get_z0_1q(c_sys)
        hs = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        lindbladian_base = EffectiveLindbladian(c_sys, hs)
        generation_setting = RandomEffectiveLindbladianGenerationSetting(
            c_sys, qoperation_base, lindbladian_base, 1.0, 2.0
        )

        ### generate
        # Act(seed_or_generator: default)
        seed = 7
        # Compare that the results are the same the first time and the second time.
        # 1_1
        np.random.seed(seed)
        state1_1, *_ = generation_setting.generate()
        # 1_2
        np.random.seed(seed)
        state1_2, *_ = generation_setting.generate()
        # Assert
        npt.assert_almost_equal(state1_1.to_var(), state1_2.to_var(), decimal=15)

        # Act(seed_or_generator: int)
        # Compare that the results are the same the first time and the second time.
        # 2_1
        state2_1, *_ = generation_setting.generate(7)
        state2_2, *_ = generation_setting.generate(7)
        # Assert
        npt.assert_almost_equal(state2_1.to_var(), state2_2.to_var(), decimal=15)

        # Act(seed_or_generator: Generator)
        # Compare that the results are the same the first time and the second time.
        # 3_1
        random_gen = np.random.Generator(np.random.MT19937(7))
        state3_1, *_ = generation_setting.generate(random_gen)
        # 3_2
        random_gen = np.random.Generator(np.random.MT19937(7))
        state3_2, *_ = generation_setting.generate(random_gen)
        # Assert
        npt.assert_almost_equal(state3_1.to_var(), state3_2.to_var(), decimal=15)

        # Assert
        npt.assert_almost_equal(state2_1.to_var(), state3_1.to_var(), decimal=15)

    def test_generate_gate(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        qoperation_base = get_x(c_sys)
        hs = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        lindbladian_base = EffectiveLindbladian(c_sys, hs)
        generation_setting = RandomEffectiveLindbladianGenerationSetting(
            c_sys, qoperation_base, lindbladian_base, 1.0, 2.0
        )

        ### generate_gate
        # Act
        (
            gate,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        ) = generation_setting.generate_gate()

        # Assert
        assert gate.hs.shape == (4, 4)
        assert gate.is_physical() == True
        assert len(random_variables_h_part) == 3
        assert len(random_variables_k_part) == 3
        assert random_unitary.shape == (3, 3)
        uni = random_unitary @ random_unitary.T.conj()
        npt.assert_almost_equal(uni, np.eye(3), decimal=15)
        assert random_el.shape == (4, 4)

        ### generate
        # Act
        (
            gate,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        ) = generation_setting.generate()

        # Assert
        assert gate.hs.shape == (4, 4)
        assert gate.is_physical() == True
        assert len(random_variables_h_part) == 3
        assert len(random_variables_k_part) == 3
        assert random_unitary.shape == (3, 3)
        uni = random_unitary @ random_unitary.T.conj()
        npt.assert_almost_equal(uni, np.eye(3), decimal=15)
        assert random_el.shape == (4, 4)

    def test_generate_mprocess(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        qoperation_base = generate_mprocess_from_name(c_sys, "z-type1")

        hs = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        lindbladian_base = EffectiveLindbladian(c_sys, hs)
        generation_setting = RandomEffectiveLindbladianGenerationSetting(
            c_sys, qoperation_base, lindbladian_base, 1.0, 2.0
        )

        ### generate_mprocess
        # Act
        (
            mprocess,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        ) = generation_setting.generate_mprocess()

        # Assert
        assert len(mprocess.hss) == 2
        assert mprocess.hss[0].shape == (4, 4)
        assert mprocess.is_physical() == True
        assert len(random_variables_h_part) == 3
        assert len(random_variables_k_part) == 3
        assert random_unitary.shape == (3, 3)
        uni = random_unitary @ random_unitary.T.conj()
        npt.assert_almost_equal(uni, np.eye(3), decimal=15)
        assert random_el.shape == (4, 4)

        ### generate
        # Act
        (
            mprocess,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        ) = generation_setting.generate()

        # Assert
        assert len(mprocess.hss) == 2
        assert mprocess.hss[0].shape == (4, 4)
        assert mprocess.is_physical() == True
        assert len(random_variables_h_part) == 3
        assert len(random_variables_k_part) == 3
        assert random_unitary.shape == (3, 3)
        uni = random_unitary @ random_unitary.T.conj()
        npt.assert_almost_equal(uni, np.eye(3), decimal=15)
        assert random_el.shape == (4, 4)

    def test_generate_povm(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        qoperation_base = get_z_povm(c_sys)
        hs = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        lindbladian_base = EffectiveLindbladian(c_sys, hs)
        generation_setting = RandomEffectiveLindbladianGenerationSetting(
            c_sys, qoperation_base, lindbladian_base, 1.0, 2.0
        )

        ### generate_povm
        # Act
        (
            povm,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        ) = generation_setting.generate_povm()

        # Assert
        assert len(povm.vecs) == 2
        assert len(povm.vecs[0]) == 4
        assert len(povm.vecs[1]) == 4
        assert povm.is_physical() == True
        assert len(random_variables_h_part) == 3
        assert len(random_variables_k_part) == 3
        assert random_unitary.shape == (3, 3)
        uni = random_unitary @ random_unitary.T.conj()
        npt.assert_almost_equal(uni, np.eye(3), decimal=15)
        assert random_el.shape == (4, 4)

        ### generate
        # Act
        (
            povm,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        ) = generation_setting.generate()

        # Assert
        assert len(povm.vecs) == 2
        assert len(povm.vecs[0]) == 4
        assert len(povm.vecs[1]) == 4
        assert povm.is_physical() == True
        assert len(random_variables_h_part) == 3
        assert len(random_variables_k_part) == 3
        assert random_unitary.shape == (3, 3)
        uni = random_unitary @ random_unitary.T.conj()
        npt.assert_almost_equal(uni, np.eye(3), decimal=15)
        assert random_el.shape == (4, 4)
