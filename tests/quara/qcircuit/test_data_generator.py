import numpy as np
import numpy.testing as npt
import pytest

from quara.qcircuit import data_generator


def test_random_number_to_data():
    prob_dist = np.array([0, 0.2, 0.3, 0.5], dtype=np.float64)
    actual = data_generator._random_number_to_data(prob_dist, 0)
    assert actual == 1

    actual = data_generator._random_number_to_data(prob_dist, 0.1)
    assert actual == 1

    actual = data_generator._random_number_to_data(prob_dist, 0.2)
    assert actual == 2

    actual = data_generator._random_number_to_data(prob_dist, 1)
    assert actual == 3


def test_generate_data_from_prob_dist():
    # normal case
    prob_dist = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    actual = data_generator.generate_data_from_prob_dist(
        prob_dist, 10, seed_or_stream=7
    )
    expected = [0, 2, 1, 2, 2, 2, 2, 0, 1, 1]
    assert actual == expected

    ### error cases
    # some probabilities are not positive numbers.
    prob_dist = np.array([-0.2, 0.5, 0.5], dtype=np.float64)
    with pytest.raises(ValueError):
        data_generator.generate_data_from_prob_dist(prob_dist, 10)

    # the sum of probabilities does not equal 1.
    prob_dist = np.array([0.2, 0.5, 0.5], dtype=np.float64)
    with pytest.raises(ValueError):
        data_generator.generate_data_from_prob_dist(prob_dist, 10)


def test_generate_dataset_from_prob_dists():
    # Arrange
    prob_dists = [
        np.array([0.2, 0.3, 0.5], dtype=np.float64),
        np.array([0.3, 0.7], dtype=np.float64),
    ]
    data_nums = [10, 20]
    seeds = [7, 77]

    # Act
    actual = data_generator.generate_dataset_from_prob_dists(
        prob_dists, data_nums, seeds
    )

    # Assert
    expected = [
        [0, 2, 1, 2, 2, 2, 2, 0, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
    ]
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a == e

    ### error cases
    # the length of ``prob_dists`` does not equal the length of ``data_nums``.
    with pytest.raises(ValueError):
        data_generator.generate_dataset_from_prob_dists(prob_dists, [10, 20, 30], seeds)

    # ``seeds`` is not None and the length of ``prob_dists`` does not equal the length of ``seeds``.
    with pytest.raises(ValueError):
        data_generator.generate_dataset_from_prob_dists(
            prob_dists, data_nums, [7, 77, 777]
        )


def test_calc_empi_dist_sequence():
    # Arrange
    measurement_num = 2
    data = [1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1]
    num_sums = [5, 10, 20]

    # Act
    actual = data_generator.calc_empi_dist_sequence(measurement_num, data, num_sums)

    # Assert
    expected = [
        (5, np.array([0.4, 0.6], dtype=np.float64)),
        (10, np.array([0.3, 0.7], dtype=np.float64)),
        (20, np.array([0.3, 0.7], dtype=np.float64)),
    ]
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a[0] == e[0]
        assert np.all(a[1] == e[1])

    ### error cases
    # ``measurement_num`` is not non-negative integer.
    with pytest.raises(ValueError):
        data_generator.calc_empi_dist_sequence(-1, data, num_sums)

    # there is an element of ``num_sums`` that is not less than or equal to length of ``data``.
    with pytest.raises(ValueError):
        data_generator.calc_empi_dist_sequence(measurement_num, data, [21])
    with pytest.raises(ValueError):
        data_generator.calc_empi_dist_sequence(measurement_num, data, [10, 21])

    # there is an element of ``data`` that is not non-negative and less than ``measurement_num``.
    with pytest.raises(ValueError):
        data_generator.calc_empi_dist_sequence(measurement_num, [-1], num_sums)
    with pytest.raises(ValueError):
        data_generator.calc_empi_dist_sequence(measurement_num, [2], num_sums)

    # ``num_sums`` is not an increasing sequence.
    with pytest.raises(ValueError):
        data_generator.calc_empi_dist_sequence(measurement_num, data, [5, 10, 10])


def test_calc_empi_dists_sequence():
    # Arrange
    measurement_nums = [3, 2]
    dataset = [
        [0, 2, 1, 2, 2, 2, 2, 0, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
    ]
    list_list_num_sum = [
        [2, 10],
        [5, 10, 20],
    ]
    # Act
    list_actual = data_generator.calc_empi_dists_sequence(
        measurement_nums, dataset, list_list_num_sum
    )

    # Assert
    list_expected = [
        [
            (2, np.array([0.5, 0.0, 0.5], dtype=np.float64)),
            (10, np.array([0.2, 0.3, 0.5], dtype=np.float64)),
        ],
        [
            (5, np.array([0.4, 0.6], dtype=np.float64)),
            (10, np.array([0.3, 0.7], dtype=np.float64)),
            (20, np.array([0.3, 0.7], dtype=np.float64)),
        ],
    ]
    assert len(list_actual) == len(list_expected)
    for actual, expected in zip(list_actual, list_expected):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a[0] == e[0]
            assert np.all(a[1] == e[1])

    ### error cases
    # the length of ``measurement_nums`` does not equal the length of ``dataset``.
    wrong_dataset = [
        [0, 2, 1, 2, 2, 2, 2, 0, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
        [1, 1, 1, 0, 0],
    ]
    with pytest.raises(ValueError):
        data_generator.calc_empi_dists_sequence(
            measurement_nums, wrong_dataset, list_list_num_sum
        )

    # the length of ``measurement_nums`` does not equal the length of ``list_list_num_sum``.
    wrong_list_list_num_sum = [
        [2, 10],
        [5, 10, 20],
        [1, 2, 3],
    ]
    with pytest.raises(ValueError):
        data_generator.calc_empi_dists_sequence(
            measurement_nums, dataset, wrong_list_list_num_sum
        )
    pass
