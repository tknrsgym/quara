import numpy as np


def l2_norm(x: np.array, y: np.array) -> np.float64:
    norm = np.linalg.norm(x - y)
    return norm
