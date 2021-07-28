import pytest

import quara.objects.state_ensemble as se


class TestStateEnsemble:
    def test_init_unexpected(self):
        invalid_eps_zero = -0.1
        with pytest.raises(ValueError):
            _ = se.StateEnsemble(states="dummy", prob_dist="dummy", eps_zero=invalid_eps_zero)