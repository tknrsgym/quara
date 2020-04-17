from typing import List, Union

import numpy as np

import quara.utils.matrix_util as mutil
from quara.objects.composite_system import CompositeSystem
from quara.objects.matrix_basis import MatrixBasis, convert_vec
from quara.settings import Settings


class Povm:
    """
    Positive Operator-Valued Measure
    """

    def __init__(
        self, c_sys: CompositeSystem, vecs: List[np.ndarray], is_physical: bool = True
    ):
        """Constructor

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem of this povm.
        vecs : List[np.ndarray]

        is_physical : bool, optional
            Check whether the povm is physically correct, by default True.
            If ``True``, the following requirements are met.

            - It is a set of Hermitian matrices.
            - The sum is the identity matrix.
            - positive semidefinite.

            If you want to ignore the above requirements and create a POVM object, set ``is_physical`` to ``False``.

        Raises
        ------
        ValueError
            If ``is_physical`` is ``True`` and it is not a set of Hermitian matrices
        ValueError
            If ``is_physical`` is ``True`` and the sum is not an identity matrix
        ValueError
            If ``is_physical`` is ``True`` and is not a positive semidefinite
        ValueError
            If the dim in the ``c_sys`` does not match the dim in the ``vecs``
        """
        # Set
        # TODO: consider make it tuple of np.ndarray
        self._vecs: List[np.ndarray] = vecs
        self._composite_system: CompositeSystem = c_sys

        # 観測されうる測定値の集合
        self._measurements: list

        self._is_physical = is_physical

        # Validation
        ## Validate whether `vecs` is a set of Hermitian matrices
        # TODO: Consider using VectorizedMatrixBasis
        size = vecs[0].shape
        self._dim = int(np.sqrt(size[0]))
        size = [self._dim, self._dim]

        if is_physical:
            # Validate to meet requirements as Povm
            if not self.is_hermitian():
                raise ValueError("POVM must be a set of Hermitian matrices")

            if not self.is_identity():
                # whether the sum of the elements is an identity matrix or not
                raise ValueError(
                    "The sum of the elements of POVM must be an identity matrix."
                )

            if not self.is_positive_semidefinite():
                raise ValueError("Eigenvalues of POVM elements must be non-negative.")

        # Whether dim of CompositeSystem equals dim of vec
        if c_sys.dim != self._dim:
            raise ValueError(
                f"dim of CompositeSystem must equal dim of vec. dim of CompositeSystem is {c_sys.dim}. dim of vec is {self._dim}"
            )

    def __getitem__(self, key: int):
        return self._vecs[key]

    @property
    def vecs(self) -> List[np.ndarray]:  # read only
        """Property to get vecs of povm.

        Returns
        -------
        List[np.ndarray]
            vecs of povm.
        """
        return self._vecs

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def composite_system(self) -> CompositeSystem:
        """Property to get composite system.

        Returns
        -------
        CompositeSystem
            composite system.
        """
        return self._composite_system

    @property
    def is_physical(self) -> bool:  # read only
        return self._is_physical

    def is_hermitian(self) -> bool:
        for m in self.matrixes():
            if not mutil.is_hermitian(m):
                return False
        return True

    def is_positive_semidefinite(self, atol: float = None) -> bool:
        """Returns whether each element is positive semidifinite.

        Returns
        -------
        bool
            True where each element is positive semidifinite, False otherwise.
        """
        atol = atol if atol else Settings.get_atol()

        size = [self.dim, self.dim]
        for m in self.matrixes():
            if not mutil.is_positive_semidefinite(m, atol):
                return False

        return True

    def is_identity(self) -> bool:
        """Returns whether the sum of the elements ``_vecs`` is an identity matrix.

        Returns
        -------
        bool
            If the sum of the elements ``_vecs`` is an identity matrix,
            otherwise it returns False.
        """
        sum_matrix = self._sum_matrix()
        identity = np.identity(self.dim, dtype=np.complex128)
        return np.allclose(sum_matrix, identity)

    def _sum_matrix(self):
        size = [self.dim, self.dim]
        sum_matrix = np.zeros(size, dtype=np.complex128)
        for m in self.matrixes():
            sum_matrix += np.reshape(m, size)

        return sum_matrix

    # TODO: あとで良い名前に変える
    def matrixes(self):
        matrix_list = []
        size = (self.dim, self.dim)
        for v in self.vecs:
            matrix = np.zeros(size, dtype=np.complex128)
            for coefficient, basis in zip(v, self.composite_system.basis()):
                matrix += coefficient * basis
            matrix_list.append(matrix)
        return matrix_list

    def calc_eigenvalues(
        self, index: int = None
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Calculates eigenvalues.

        Parameters
        ----------
        index : int, optional
            Index to obtain eigenvalues, by default None

        Returns
        -------
        Union[List[np.ndarray], np.ndarray]
            eigenvalues.
        """

        size = [self._dim, self._dim]
        if index is not None:
            v = self.matrixes()[index]
            matrix = np.reshape(v, size)
            w = np.linalg.eigvals(matrix)
            return w
        else:
            w_list = []
            for v in self.matrixes():
                matrix = np.reshape(v, size)
                w = np.linalg.eigvals(matrix)
                w_list.append(w)
            return w_list

    def convert_basis(self, other_basis: MatrixBasis) -> List[np.array]:
        """Calculate vector representation for ``other_basis``.

        Parameters
        ----------
        other_basis : MatrixBasis
            basis

        Returns
        -------
        List[np.array]
            Vector representation after conversion to ``other_basis`` .
        """

        converted_vecs = []
        for vec in self.vecs:
            converted_vecs.append(
                convert_vec(vec, self._composite_system.basis(), other_basis)
            )
        return converted_vecs
