import numpy as np
import pytest

from quara.math import matrix


def test_multiply_veca_vecb():
    vec_a = np.array([2, 3])
    vec_b = np.array([5, 7])
    actual = matrix.multiply_veca_vecb(vec_a, vec_b)
    assert actual == 31

    invalid_vec_a = np.array([[11, 13], [17, 19]])
    with pytest.raises(ValueError):
        matrix.multiply_veca_vecb(invalid_vec_a, vec_b)

    invalid_vec_b = np.array([[11, 13], [17, 19]])
    with pytest.raises(ValueError):
        matrix.multiply_veca_vecb(vec_a, invalid_vec_b)


def test_multiply_veca_vecb_matc():
    vec_a = np.array([2, 3])
    vec_b = np.array([5, 7])
    mat_c = np.array([[11, 13], [17, 19]])
    actual = matrix.multiply_veca_vecb_matc(vec_a, vec_b, mat_c)
    assert actual == 946

    invalid_vec_a = np.array([[11, 13], [17, 19]])
    with pytest.raises(ValueError):
        matrix.multiply_veca_vecb_matc(invalid_vec_a, vec_b, mat_c)

    invalid_vec_b = np.array([[11, 13], [17, 19]])
    with pytest.raises(ValueError):
        matrix.multiply_veca_vecb_matc(vec_a, invalid_vec_b, mat_c)

    invalid_mat_c = np.array([2, 3])
    with pytest.raises(ValueError):
        matrix.multiply_veca_vecb_matc(vec_a, vec_b, invalid_mat_c)
