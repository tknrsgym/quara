import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.multinomial_distribution import MultinomialDistribution


class TestMultinomialDistribution:
    def test_init_error(self):
        # some elements of ps are negative numbers.
        ps = np.array([-0.1, 0.2, 0.3, 0.5], dtype=np.float64)
        shape = (4,)
        with pytest.raises(ValueError):
            MultinomialDistribution(ps, shape)

        # the sum of ps is not 1.
        ps = np.array([0.1, 0.2, 0.3, 0.5], dtype=np.float64)
        shape = (4,)
        with pytest.raises(ValueError):
            MultinomialDistribution(ps, shape)

        # the size of ps and shape do not match.
        ps = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        shape = (3,)
        with pytest.raises(ValueError):
            MultinomialDistribution(ps, shape)

    def test_access_ps(self):
        ps = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        shape = (4,)
        dist = MultinomialDistribution(ps, shape)
        npt.assert_almost_equal(dist.ps, ps, decimal=15)

        # Test that "ps" cannot be updated
        with pytest.raises(AttributeError):
            dist.ps = ps

    def test_access_shape(self):
        ps = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        shape = (4,)
        dist = MultinomialDistribution(ps, shape)
        assert dist.shape == shape

        ps = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        dist = MultinomialDistribution(ps)
        assert dist.shape == (4,)

        # Test that "shape" cannot be updated
        with pytest.raises(AttributeError):
            dist.shape = shape

    def test_access_eps_zero(self):
        ps = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        shape = (4,)

        # case1: eps_zero=default
        dist = MultinomialDistribution(ps, shape)
        assert dist.eps_zero == 1e-8

        # case1: eps_zero=1e-10
        eps_zero = 1e-10
        dist = MultinomialDistribution(ps, shape, eps_zero)
        assert dist.eps_zero == eps_zero

        # Test that "eps_zero" cannot be updated
        with pytest.raises(AttributeError):
            dist.eps_zero = eps_zero

    def test_is_zero_dist(self):
        # case 1: is_zero_dist=False and do not adjust ps
        ps = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        shape = (4,)
        actual = MultinomialDistribution(ps, shape)
        expected = np.array([0.1, 0.2, 0.3, 0.4])
        npt.assert_almost_equal(actual.ps, expected, decimal=15)
        assert actual.is_zero_dist == False

        # case 2: is_zero_dist=False and do adjust ps
        ps = np.array([0.01, 0.2, 0.3, 0.49], dtype=np.float64)
        shape = (4,)
        eps_zero = 1e-1
        actual = MultinomialDistribution(ps, shape, eps_zero)
        expected = np.array([0.0, 0.2, 0.3, 0.49])
        expected = expected / np.sum(expected)
        npt.assert_almost_equal(actual.ps, expected, decimal=15)
        assert actual.is_zero_dist == False

        # case 3: is_zero_dist=True and do not adjust ps
        ps = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        shape = (4,)
        actual = MultinomialDistribution(ps, shape)
        expected = np.array([0.0, 0.0, 0.0, 0.0])
        npt.assert_almost_equal(actual.ps, expected, decimal=15)
        assert actual.is_zero_dist == True

        # case 4: is_zero_dist=True and do adjust ps
        ps = np.array([0.01, 0.0, 0.0, 0.0], dtype=np.float64)
        shape = (4,)
        eps_zero = 1e-1
        actual = MultinomialDistribution(ps, shape, eps_zero)
        expected = np.array([0.0, 0.0, 0.0, 0.0])
        npt.assert_almost_equal(actual.ps, expected, decimal=15)
        assert actual.is_zero_dist == True

    def test_getitem__int(self):
        factor = np.sum(range(30))
        ps = np.array(range(30)) / factor
        shape = (2, 3, 5)
        dist = MultinomialDistribution(ps, shape)

        # case 1:
        actual = dist[0]
        assert actual == 0 / factor

        # case 2:
        actual = dist[1]
        assert actual == 1 / factor

        # case 3:
        actual = dist[29]
        assert actual == 29 / factor

    def test_getitem__tuple(self):
        factor = np.sum(range(30))
        ps = np.array(range(30)) / factor
        shape = (2, 3, 5)
        dist = MultinomialDistribution(ps, shape)

        # case 1:
        actual = dist[(0, 0, 0)]
        assert actual == 0 / factor

        # case 2:
        actual = dist[(1, 1, 1)]
        assert actual == 21 / factor

        # case 3:
        actual = dist[(1, 2, 3)]
        assert actual == 28 / factor

    def test_getitem__error(self):
        factor = np.sum(range(30))
        ps = np.array(range(30)) / factor
        shape = (2, 3, 5)
        dist = MultinomialDistribution(ps, shape)

        # Test that unsupported type
        with pytest.raises(TypeError):
            dist[1.0]

    def test_marginalize(self):
        factor = np.sum(range(30))
        ps = np.array(range(30)) / factor
        shape = (2, 3, 5)
        dist = MultinomialDistribution(ps, shape)

        # case 1
        new_dist = dist.marginalize([2])
        expected = np.sum(ps.reshape(shape), axis=(0, 1))
        npt.assert_almost_equal(new_dist.ps, expected.flatten(), decimal=15)

        # case 2
        new_dist = dist.marginalize([0, 1])
        expected = np.sum(ps.reshape(shape), axis=2)
        npt.assert_almost_equal(new_dist.ps, expected.flatten(), decimal=15)

        # case 3
        new_dist = dist.marginalize([0, 1, 2])
        npt.assert_almost_equal(new_dist.ps, ps, decimal=15)

        # case 4
        with pytest.raises(ValueError):
            dist.marginalize([3])

    def test_conditionalize(self):
        factor = np.sum(range(30))
        ps = np.array(range(30)) / factor
        shape = (2, 3, 5)
        dist = MultinomialDistribution(ps, shape)

        # case 1:
        actual = dist.conditionalize([0], [0])
        tmp = ps.reshape(shape)[0]
        expected = tmp / np.sum(tmp)
        npt.assert_almost_equal(actual.ps, expected.flatten(), decimal=15)
        assert actual.shape == (3, 5)

        # case 2:
        actual = dist.conditionalize([0, 2], [0, 3])
        tmp = ps.reshape(shape)[0, :, 3]
        expected = tmp / np.sum(tmp)
        npt.assert_almost_equal(actual.ps, expected.flatten(), decimal=15)
        assert actual.shape == (3,)

    def test_conditionalize_error(self):
        factor = np.sum(range(30))
        ps = np.array(range(30)) / factor
        shape = (2, 3, 5)
        dist = MultinomialDistribution(ps, shape)

        with pytest.raises(ValueError):
            dist.conditionalize([0], [0, 1])
        with pytest.raises(ValueError):
            dist.conditionalize([-1], [0])
        with pytest.raises(ValueError):
            dist.conditionalize([0], [-1])

    def test_execute_random_sampling(self):
        ps = np.array([0.1, 0.2, 0.3, 0.4])
        shape = (4,)
        dist = MultinomialDistribution(ps, shape)

        num = 100
        size = 2

        # case 1: random_generator is None
        actual = dist.execute_random_sampling(num, size)
        assert len(actual) == size
        for a in actual:
            assert np.sum(a) == num

        # case 2: random_generator is int
        actual = dist.execute_random_sampling(num, size, random_generator=7)
        expected = [
            np.array([8, 22, 30, 40]),
            np.array([13, 21, 33, 33]),
        ]
        assert len(actual) == size
        for a, e in zip(actual, expected):
            assert np.sum(a) == num
            npt.assert_almost_equal(a, e, decimal=15)

        # case 3: random_generator is Generator
        expected = [
            np.array([8, 22, 30, 40]),
            np.array([13, 21, 33, 33]),
        ]
        random_gen = np.random.Generator(np.random.MT19937(7))
        actual = dist.execute_random_sampling(num, size, random_generator=random_gen)
        assert len(actual) == size
        for a, e in zip(actual, expected):
            assert np.sum(a) == num
            npt.assert_almost_equal(a, e, decimal=15)
