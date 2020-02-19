from typing import List

import numpy as np

from quara.objects.composite_system import CompositeSystem
from quara.objects.matrix_basis import MatrixBasis
import quara.utils.matrix_util as mutil


class State:
    def __init__(self, c_sys: CompositeSystem, vec: np.ndarray):
        self._composite_system: CompositeSystem = c_sys
        self._vec: np.ndarray = vec

        size = self._vec.shape
        # whether vec is one-dimensional array
        assert len(size) == 1
        # whether size of vec is square
        self._dim = int(np.sqrt(size[0]))
        assert self._dim ** 2 == size[0]
        # whether entries of vec are real numbers
        assert self._vec.dtype == np.float64
        # whether dim of CompositeSystem equals dim of vec
        assert self._composite_system.dim == self._dim

    @property
    def dim(self):
        """returns dim of vector.
        
        Returns
        -------
        int
            dim of matrix
        """
        return self._dim

    def get_density_matrix(self) -> np.ndarray:
        """returns density matrix.
        
        Returns
        -------
        int
            density matrix
        """
        density = np.zeros((self._dim, self._dim), dtype=np.complex128)
        for coefficient, basis in zip(self._vec, self._composite_system.basis):
            density += coefficient * basis
        return density

    def is_trace_one(self) -> bool:
        """returns whether trace of density matrix is one.
        
        Returns
        -------
        bool
            True where trace of density matrix is one, False otherwise.
        """
        tr = np.trace(self.get_density_matrix())
        return tr == 1

    def is_hermitian(self) -> bool:
        """returns whether density matrix is Hermitian.
        
        Returns
        -------
        bool
        True where density matrix, False otherwise.
        """
        return mutil.is_hermitian(self.get_density_matrix())

    def is_positive_semidefinite(self) -> bool:
        """returns whether density matrix is positive semidifinite.

        Returns
        -------
        bool
            True where density matrix is positive semidifinite, False otherwise.
        """
        return mutil.is_positive_semidefinite(self.get_density_matrix())

    def get_eigen_values(self) -> List:
        """returns eigen values of density matrix.

        this function uses numpy API.
        see this URL for details:
        https://numpy.org/doc/1.18/reference/generated/numpy.linalg.eigvals.html
        
        Returns
        -------
        List
            eigen values of density matrix
        """
        return np.linalg.eigvals(self.get_density_matrix())

    def convert_basis(self, basis: MatrixBasis) -> np.array:
        # TODO 別の行列基底に対するベクトル表現を返す
        # "converted vec"_{\alpha} = Tr["new basis"_{\alpha}^{\dagger} "old basis"_{\alpha}]
        pass
