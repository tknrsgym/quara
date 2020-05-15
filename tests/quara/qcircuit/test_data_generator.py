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
