import numpy as np
import numpy.testing as npt
import pytest

import quara.objects.state_ensemble as se
from quara.objects.multinomial_distribution import MultinomialDistribution
from quara.objects.qoperation_typical import generate_qoperation_object
from quara.objects.composite_system_typical import generate_composite_system


class TestStateEnsemble:
    def test_init_unexpected(self):
        # Case 1:
        invalid_eps_zero = -0.1
        with pytest.raises(ValueError):
            _ = se.StateEnsemble(
                states="dummy", prob_dist="dummy", eps_zero=invalid_eps_zero
            )
        # Case 2: invalid length
        c_sys = generate_composite_system(mode="qubit", num=1, ids_esys=[0])
        state_z0 = generate_qoperation_object(
            mode="state", object_name="state", name="z0", c_sys=c_sys
        )
        state_z1 = generate_qoperation_object(
            mode="state", object_name="state", name="z1", c_sys=c_sys
        )
        state_y0 = generate_qoperation_object(
            mode="state", object_name="state", name="y0", c_sys=c_sys
        )
        states = [state_z0, state_z1, state_y0]
        prob_dist = MultinomialDistribution(ps=np.array([0.1, 0.9]), shape=(2,))

        with pytest.raises(ValueError):
            _ = se.StateEnsemble(states=states, prob_dist=prob_dist)

        invalid_prob_dist = [0.1, 0.9]
        with pytest.raises(TypeError):
            # TypeError: Type of prob_dist muste be MultinomialDistribution, not <class 'list'>
            _ = se.StateEnsemble(
                states="dummy",
                prob_dist=invalid_prob_dist,
            )

    def test_state(self):
        # Arrange
        c_sys_list = []
        for i in range(6):
            c_sys_list.append(
                generate_composite_system(mode="qubit", num=1, ids_esys=[i])
            )

        state_z0 = generate_qoperation_object(
            mode="state", object_name="state", name="z0", c_sys=c_sys_list[0]
        )
        state_z1 = generate_qoperation_object(
            mode="state", object_name="state", name="z1", c_sys=c_sys_list[1]
        )
        state_y0 = generate_qoperation_object(
            mode="state", object_name="state", name="y0", c_sys=c_sys_list[2]
        )
        state_y1 = generate_qoperation_object(
            mode="state", object_name="state", name="y1", c_sys=c_sys_list[3]
        )
        state_x0 = generate_qoperation_object(
            mode="state", object_name="state", name="x0", c_sys=c_sys_list[4]
        )
        state_x1 = generate_qoperation_object(
            mode="state", object_name="state", name="x0", c_sys=c_sys_list[5]
        )
        states = [state_z0, state_z1, state_y0, state_y1, state_x0, state_x1]

        ps = np.array([0.005, 0.025, 0.07, 0.045, 0.225, 0.63])
        prob_dist = MultinomialDistribution(ps=ps, shape=(2, 3))

        state_ensemble = se.StateEnsemble(states=states, prob_dist=prob_dist)

        # Case 1: Multi-dimensional access
        # Act & Assert
        actual = state_ensemble.state((0, 0))
        npt.assert_almost_equal(actual.vec, state_z0.vec, decimal=15)

        actual = state_ensemble.state((0, 1))
        npt.assert_almost_equal(actual.vec, state_z1.vec, decimal=15)

        actual = state_ensemble.state((0, 2))
        npt.assert_almost_equal(actual.vec, state_y0.vec, decimal=15)

        actual = state_ensemble.state((1, 0))
        npt.assert_almost_equal(actual.vec, state_y1.vec, decimal=15)

        actual = state_ensemble.state((1, 1))
        npt.assert_almost_equal(actual.vec, state_x0.vec, decimal=15)

        actual = state_ensemble.state((1, 2))
        npt.assert_almost_equal(actual.vec, state_x1.vec, decimal=15)

        # Case 2: serial index access
        # Act & Assert
        actual = state_ensemble.state(0)
        npt.assert_almost_equal(actual.vec, state_z0.vec, decimal=15)

        actual = state_ensemble.state(1)
        npt.assert_almost_equal(actual.vec, state_z1.vec, decimal=15)

        actual = state_ensemble.state(2)
        npt.assert_almost_equal(actual.vec, state_y0.vec, decimal=15)

        actual = state_ensemble.state(3)
        npt.assert_almost_equal(actual.vec, state_y1.vec, decimal=15)

        actual = state_ensemble.state(4)
        npt.assert_almost_equal(actual.vec, state_x0.vec, decimal=15)

        actual = state_ensemble.state(5)
        npt.assert_almost_equal(actual.vec, state_x1.vec, decimal=15)
