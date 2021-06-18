from abc import abstractmethod

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
from quara.objects.state import get_z0_1q
from quara.objects.povm import get_z_povm


class TestRandomEffectiveLindbladianGenerationSetting:
    def test_access_is_seed_or_stream_required(self):
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
        assert actual.is_seed_or_stream_required == True

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
        # Act(seed_or_stream: default)
        seed = 7
        np.random.seed(seed)
        (
            state1,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        ) = generation_setting.generate()

        # Act(seed_or_stream: int)
        (
            state2,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        ) = generation_setting.generate(7)

        # Act(seed_or_stream: RandomState)
        (
            state3,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        ) = generation_setting.generate(np.random.RandomState(7))

        npt.assert_almost_equal(state1.to_var(), state2.to_var(), decimal=15)
        npt.assert_almost_equal(state2.to_var(), state3.to_var(), decimal=15)

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
