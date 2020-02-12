import numpy as np
import quara.utils.matrix_util as mutil


class MatrixBasis:
    def __init__(self):
        self.array: np.ndarray = None  # TODO

    def is_hermitian(self) -> bool:
        return mutil.is_hermitian(self.array)

