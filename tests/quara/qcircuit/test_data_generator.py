import numpy as np
import numpy.testing as npt
import pytest

from quara.qcircuit import data_generator


def test_random_number_to_data():
    probdist = np.array([0, 0.2, 0.3, 0.5], dtype=np.float64)
    actual = data_generator._random_number_to_data(probdist, 0)
    assert actual == 1

    actual = data_generator._random_number_to_data(probdist, 0.1)
    assert actual == 1

    actual = data_generator._random_number_to_data(probdist, 0.2)
    assert actual == 2

    actual = data_generator._random_number_to_data(probdist, 1)
    assert actual == 3


def test_generate_data_from_probdist():
    # normal case
    probdist = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    actual = data_generator.generate_data_from_probdist(probdist, 10, seed=7)
    expected = [0, 2, 1, 2, 2, 2, 2, 0, 1, 1]
    assert actual == expected

    ### error cases
    # some probabilities are not positive numbers.
    probdist = np.array([-0.2, 0.5, 0.5], dtype=np.float64)
    with pytest.raises(ValueError):
        data_generator.generate_data_from_probdist(probdist, 10)

    # the sum of probabilities does not equal 1.
    probdist = np.array([0.2, 0.5, 0.5], dtype=np.float64)
    with pytest.raises(ValueError):
        data_generator.generate_data_from_probdist(probdist, 10)


def test_generate_dataset_from_list_probdist():
    # Arrange
    probdist_list = [
        np.array([0.2, 0.3, 0.5], dtype=np.float64),
        np.array([0.3, 0.7], dtype=np.float64),
    ]
    data_num_list = [10, 20]
    seed_list = [7, 77]

    # Act
    actual = data_generator.generate_dataset_from_list_probdist(
        probdist_list, data_num_list, seed_list
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
    # the length of ``probdist_list`` does not equal the length of ``data_num_list``.
    with pytest.raises(ValueError):
        data_generator.generate_dataset_from_list_probdist(
            probdist_list, [10, 20, 30], seed_list
        )

    # ``seed_list`` is not None and the length of ``probdist_list`` does not equal the length of ``seed_list``.
    with pytest.raises(ValueError):
        data_generator.generate_dataset_from_list_probdist(
            probdist_list, data_num_list, [7, 77, 777]
        )


def test_calc_empidist():
    # Arrange
    measurement_num = 2
    data = [1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1]
    list_num_sum = [5, 10, 20]

    # Act
    actual = data_generator.calc_empidist(measurement_num, data, list_num_sum)

    # Assert
    expected = [
        np.array([0.4, 0.6], dtype=np.float64),
        np.array([0.3, 0.7], dtype=np.float64),
        np.array([0.3, 0.7], dtype=np.float64),
    ]
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert np.all(a == e)

    ### error cases
    # ``measurement_num`` is not non-negative integer.
    with pytest.raises(ValueError):
        data_generator.calc_empidist(-1, data, list_num_sum)

    # there is an element of ``list_num_sum`` that is not less than or equal to length of ``data``.
    with pytest.raises(ValueError):
        data_generator.calc_empidist(measurement_num, data, [21])
    with pytest.raises(ValueError):
        data_generator.calc_empidist(measurement_num, data, [10, 21])

    # there is an element of ``data`` that is not non-negative and less than ``measurement_num``.
    with pytest.raises(ValueError):
        data_generator.calc_empidist(measurement_num, [-1], list_num_sum)
    with pytest.raises(ValueError):
        data_generator.calc_empidist(measurement_num, [2], list_num_sum)

    # ``list_num_sum`` is not an increasing sequence.
    with pytest.raises(ValueError):
        data_generator.calc_empidist(measurement_num, data, [5, 10, 10])


def test_calc_list_empidist():
    # Arrange
    list_measurement_num = [3, 2]
    list_data = [
        [0, 2, 1, 2, 2, 2, 2, 0, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
    ]
    list_list_num_sum = [
        [2, 10],
        [5, 10, 20],
    ]

    # Act
    list_actual = data_generator.calc_list_empidist(
        list_measurement_num, list_data, list_list_num_sum
    )

    # Assert
    list_expected = [
        [
            np.array([0.5, 0.0, 0.5], dtype=np.float64),
            np.array([0.2, 0.3, 0.5], dtype=np.float64),
        ],
        [
            np.array([0.4, 0.6], dtype=np.float64),
            np.array([0.3, 0.7], dtype=np.float64),
            np.array([0.3, 0.7], dtype=np.float64),
        ],
    ]
    assert len(list_actual) == len(list_expected)
    for actual, expected in zip(list_actual, list_expected):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert all(a == e)

    ### error cases
    # the length of ``list_measurement_num`` does not equal the length of ``list_data``.
    wrong_list_data = [
        [0, 2, 1, 2, 2, 2, 2, 0, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
        [1, 1, 1, 0, 0],
    ]
    with pytest.raises(ValueError):
        data_generator.calc_list_empidist(
            list_measurement_num, wrong_list_data, list_list_num_sum
        )

    # the length of ``list_measurement_num`` does not equal the length of ``list_list_num_sum``.
    wrong_list_list_num_sum = [
        [2, 10],
        [5, 10, 20],
        [1, 2, 3],
    ]
    with pytest.raises(ValueError):
        data_generator.calc_list_empidist(
            list_measurement_num, list_data, wrong_list_list_num_sum
        )
