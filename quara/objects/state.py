import itertools
from typing import List

import numpy as np

import quara.utils.matrix_util as mutil
from quara.objects.composite_system import CompositeSystem
from quara.objects.matrix_basis import MatrixBasis


class State:
    def __init__(self, c_sys: CompositeSystem, vec: np.ndarray):
        self._composite_system: CompositeSystem = c_sys
        self._vec: np.ndarray = vec
        size = self._vec.shape

        # whether vec is one-dimensional array
        if len(size) != 1:
            raise ValueError(f"vec must be one-dimensional array. shape is {size}")

        # whether size of vec is square
        self._dim = int(np.sqrt(size[0]))
        if self._dim ** 2 != size[0]:
            raise ValueError(f"size of vec must be square. dim of vec is {size[0]}")

        # whether entries of vec are real numbers
        if self._vec.dtype != np.float64:
            raise ValueError(
                f"entries of vec must be real numbers. dtype of vec is {self._vec.dtype}"
            )

        # whether dim of CompositeSystem equals dim of vec
        if self._composite_system.dim != self._dim:
            raise ValueError(
                f"dim of CompositeSystem must equal dim of vec. dim of CompositeSystem is {self._composite_system.dim}. dim of vec is {self._dim}"
            )

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

    def convert_basis(self, other_basis: MatrixBasis) -> np.array:
        """returns vector representation for ``other_basis``.
        
        Parameters
        ----------
        other_basis : MatrixBasis
            basis
        
        Returns
        -------
        np.array
            vector representation for ``other_basis``
        """
        # check length and dim
        len_basis = len(self._composite_system.basis)
        assert len_basis == len(other_basis)
        assert self._composite_system.basis.dim == other_basis.dim

        # "converted vec"_{\alpha} = \sum_{\beta} Tr["other basis"_{\beta}^{\dagger} "self basis"_{\alpha}] "self vec"_{\alpha}
        representation_matrix = [
            mutil.inner_product(val1, val2)
            for val1, val2 in itertools.product(
                other_basis.basis, self._composite_system.basis.basis
            )
        ]
        rep_mat = np.array(representation_matrix).reshape(len_basis, len_basis)
        converted_vec = rep_mat @ self._vec
        return converted_vec
