import pytest
import numpy as np
import numpy.testing as npt

from quara.utils import number_util


def test_check_positive_number():
    parameter_name = "dim"
    # valid
    target = 1
    number_util.check_positive_number(target, parameter_name)

    # invalid
    target = 0

    with pytest.raises(ValueError):
        number_util.check_positive_number(target, parameter_name)


def test_check_nonnegative_number():
    parameter_name = "dim"
    # valid
    target = 0
    number_util.check_nonnegative_number(target, parameter_name)

    # invalid
    target = -0.1

    with pytest.raises(ValueError):
        number_util.check_nonnegative_number(target, parameter_name)


def test_to_stream():
    # seed_or_generator :default
    np.random.seed(7)
    actual = number_util.to_stream()
    expected_1 = np.array(
        [
            0.076308289373957,
            0.779918792240115,
            0.438409231440893,
            0.723465177830941,
            0.977989511996603,
        ]
    )
    expected_2 = np.array(
        [
            0.538495870410434,
            0.501120463659938,
            0.072051133359762,
            0.268438980101871,
            0.49988250082556,
        ]
    )
    npt.assert_almost_equal(actual.random(5), expected_1, decimal=15)
    npt.assert_almost_equal(actual.random(5), expected_2, decimal=15)

    # seed_or_generator = None
    np.random.seed(7)
    actual = number_util.to_stream(None)
    npt.assert_almost_equal(actual.random(5), expected_1, decimal=15)
    npt.assert_almost_equal(actual.random(5), expected_2, decimal=15)

    # seed_or_generator = int:7
    actual = number_util.to_stream(7)
    expected_1 = np.array(
        [
            0.306304607065943,
            0.63299725243518,
            0.194027830619234,
            0.552264489175792,
            0.816913821715506,
        ]
    )
    expected_2 = np.array(
        [
            0.70583028124081,
            0.888642949180135,
            0.644867942631525,
            0.217427316111541,
            0.906666684668018,
        ]
    )
    npt.assert_almost_equal(actual.random(5), expected_1, decimal=15)
    npt.assert_almost_equal(actual.random(5), expected_2, decimal=15)

    # seed_or_generator = Generator(7)
    random_gen = np.random.Generator(np.random.MT19937(7))
    actual = number_util.to_stream(random_gen)
    npt.assert_almost_equal(actual.random(5), expected_1, decimal=15)
    npt.assert_almost_equal(actual.random(5), expected_2, decimal=15)
