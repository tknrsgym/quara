import numpy as np


def to_np(matlab_array) -> np.ndarray:
    # data = eval(str(matlab_array))
    np_array = np.array(matlab_array)
    return np_array
