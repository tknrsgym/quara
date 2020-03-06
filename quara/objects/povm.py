from typing import List, Union

import numpy as np

import quara.utils.matrix_util as mutil
from quara.objects.composite_system import CompositeSystem


class Povm:
    """
    Positive Operator-Valued Measure
    """

    def __init__(self, c_sys: CompositeSystem, vecs: List[np.ndarray]):
        # 観測されうる測定値の集合
        self._measurements: list

        # Validation
        ## Validate whether `vecs` is a set of Hermitian matrices
        # TODO: Consider using VectorizedMatrixBasis
        size = vecs[0].shape
        self._dim = int(np.sqrt(size[0]))
        size = [self._dim, self._dim]
        for v in vecs:
            if not mutil.is_hermitian(np.reshape(v, size)):
                raise ValueError("povm must be a set of Hermitian matrices")

        # Whether dim of CompositeSystem equals dim of vec
        if c_sys.dim() != self._dim:
            raise ValueError(
                f"dim of CompositeSystem must equal dim of vec. dim of CompositeSystem is {c_sys.dim}. dim of vec is {self._dim}"
            )

        # Set
        self._composite_system: CompositeSystem = c_sys
        self._vecs: List[np.ndarray] = vecs

    def __getitem__(self, key: int):
        return self._vecs[key]

    @property
    def composite_system(self):
        return self._composite_system

    def is_positive_semidefinite(self, atol: float = None) -> bool:
        """Returns whether each element is positive semidifinite.

        Returns
        -------
        bool
            True where each element is positive semidifinite, False otherwise.
        """

        size = [self._dim, self._dim]
        for v in self._vecs:
            if not mutil.is_positive_semidefinite(np.reshape(v, size), atol):
                return False

        return True

    def is_identity(self) -> bool:
        """Returns whether the sum of the elements ``_vecs`` is an identity matrix.

        Returns:
            bool: Return True
            if the sum of the elements ``_vecs`` is an identity matrix,
            otherwise it returns False.
        """
        size = [self._dim, self._dim]
        sum_matrix = np.zeros(size, dtype=np.complex128)
        for v in self._vecs:
            sum_matrix += np.reshape(v, size)

        identity = np.identity(self._dim, dtype=np.complex128)
        return np.allclose(sum_matrix, identity)

    def get_eigen_values(
        self, index: int = None
    ) -> Union[List[np.ndarray], np.ndarray]:
        # 各要素の固有値を返す

        size = [self._dim, self._dim]
        if index:
            v = self._vecs[index]
            matrix = np.reshape(v, size)
            w = np.linalg.eigvals(matrix)
            return w
        else:
            w_list = []
            for v in self._vecs:
                matrix = np.reshape(v, size)
                w = np.linalg.eigvals(matrix)
                w_list.append(w)
            return w_list
