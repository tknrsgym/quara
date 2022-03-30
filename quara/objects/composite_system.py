import copy
import itertools
from typing import List, Tuple, Union
import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
from scipy.linalg import kron

from quara.objects.elemental_system import ElementalSystem

from quara.objects.matrix_basis import SparseMatrixBasis, MatrixBasis, get_comp_basis
from quara.utils import matrix_util


class CompositeSystem:
    """Class for describing Composite system"""

    def __init__(self, systems: List[ElementalSystem]):
        """Constructor

        Parameters
        ----------
        systems : List[ElementalSystem]
            list of ElementalSystem of this CompositeSystem.

        Raises
        ------
        ValueError
            duplicate ElementalSystem instance.
        ValueError
            duplicate ElementalSystem name.
        """

        # Validation
        # Check for duplicate ElementalSystem
        names: List[int] = []
        e_sys_ids: List[int] = []

        for e_sys in systems:
            if e_sys.system_id in e_sys_ids:
                raise ValueError(
                    f"Duplicate ElementalSystem. \n system_id={e_sys.system_id}, name={e_sys.name}"
                )
            e_sys_ids.append(e_sys.system_id)

            if e_sys.name in names:
                raise ValueError(f"Duplicate ElementalSystem name. name={e_sys.name}")
            names.append(e_sys.name)

        # Sort by name of ElementalSystem
        # Copy to avoid affecting the original source.
        # ElementalSystem should remain the same instance as the original source
        # to check if the instances are the same in the tensor product calculation.
        # Therefore, use `copy.copy` instead of `copy.deepcopy`.
        sored_e_syses = copy.copy(systems)
        sored_e_syses.sort(key=lambda x: x.name)

        # Set
        self._elemental_systems: Tuple[ElementalSystem, ...] = tuple(sored_e_syses)

        is_orthonormal_hermitian_0thpropIs = [
            e_sys.is_orthonormal_hermitian_0thprop_identity
            for e_sys in self._elemental_systems
        ]
        self._is_orthonormal_hermitian_0thprop_identity = all(
            is_orthonormal_hermitian_0thpropIs
        )

        is_hermitian_list = [e_sys.is_hermitian for e_sys in self._elemental_systems]
        self._is_basis_hermitian = all(is_hermitian_list)

        # calculate tensor product of ElamentalSystem list for getting total MatrixBasis
        if len(self._elemental_systems) == 1:
            # TODO: check
            # self._total_basis = self._elemental_systems[0].basis
            self._total_basis = SparseMatrixBasis(self._elemental_systems[0].basis)
        else:
            basis_list = [e_sys.basis for e_sys in self._elemental_systems]
            temp = basis_list[0]
            for elem in basis_list[1:]:
                temp = [
                    matrix_util.kron(val1, val2)
                    for val1, val2 in itertools.product(temp, elem)
                ]
            # if type(basis_list[0]) == SparseMatrixBasis:
            #     self._total_basis = SparseMatrixBasis(temp)
            # elif type(basis_list[0]) == MatrixBasis:
            #     self._total_basis = MatrixBasis(temp)
            # else:
            #     error_message = f"The Type of basis_list[0] must be MatrixBasis or SparseMatrixBasis, not {type(basis_list[0])}"
            #     raise ValueError(error_message)
            self._total_basis = SparseMatrixBasis(temp)

        self._basis_basisconjugate = None
        self._dict_from_hs_to_choi = None
        self._dict_from_choi_to_hs = None
        self._basis_T_sparse = None
        self._basisconjugate_sparse = None
        self._basisconjugate_basis_sparse = None
        self._basis_basisconjugate_T_sparse = None
        self._basis_basisconjugate_T_sparse_from_1 = None
        self._basishermitian_basis_T_from_1 = None

    def comp_basis(self, mode: str = "row_major") -> SparseMatrixBasis:
        """returns computational basis of CompositeSystem.

        Parameters
        ----------
        mode : str, optional
            specify whether the order of basis is "row_major" or "column_major", by default "row_major".

        Returns
        -------
        MatrixBasis
            computational basis of CompositeSystem.

        Raises
        ------
        ValueError
            ``mode`` is unsupported.
        """
        # calculate tensor product of ElamentalSystem list for getting new MatrixBasis
        basis_tmp: SparseMatrixBasis

        if len(self._elemental_systems) == 1:
            basis_tmp = self._elemental_systems[0].comp_basis(mode=mode)
        else:
            basis_tmp = get_comp_basis(self.dim, mode=mode)
        return basis_tmp

    def basis(self) -> SparseMatrixBasis:
        """returns MatrixBasis of CompositeSystem.

        Returns
        -------
        SparseMatrixBasis
            SparseMatrixBasis of CompositeSystem.
        """
        return self._total_basis

    @property
    def dim(self) -> int:
        """returns dim of CompositeSystem.

        the dim of CompositeSystem equals the dim of basis.

        Returns
        -------
        int
            dim of CompositeSystem.
        """
        return self.basis()[0].shape[0]

    @property
    def num_e_sys(self) -> int:
        """returns the number of ElementalSystem.

        the number of ElementalSystem.

        Returns
        -------
        int
            num of ElementalSystem.

        """
        return len(self._elemental_systems)

    def dim_e_sys(self, i: int) -> int:
        """returns the dimension of the i-th ElementalSystem.

        the dim of the i-th ElementalSystem.

        Parameters
        ----------
        i: int
            the id of an ElementalSystem

        Returns
        -------
        int
            the dim of the i-th ElementalSystem
        """
        return self._elemental_systems[i].dim

    def get_basis(self, index: Union[int, Tuple]) -> SparseMatrixBasis:
        """returns basis specified by index.

        Parameters
        ----------
        index : Union[int, Tuple]
            index of basis.

            - if type is int, then regardes it as the index after calculating the basis of CompositeSystem.
            - if type is Tuple, then regardes it as the indices of the basis of ElementalSystems.

        Returns
        -------
        MatrixBasis
            basis specified by index.

        Raises
        ------
        ValueError
            length of tuple does not equal length of the list of the basis.
        IndexError
            specified index does not exist in the list of the basis.
        """
        if type(index) == tuple:
            # whether size of tuple equals length of the list of ElementalSystems
            if len(index) != len(self._elemental_systems):
                raise ValueError(
                    f"length of tuple must equal length of the list of ElementalSystems. length of tuple={len(index)}, length of the list of ElementalSystems={len(self._elemental_systems)}"
                )

            # calculate index in _basis by traversing the tuple from the back.
            # for example, if length of ElementalSystem is 3 and each dim are dim1, dim2, dim3,
            # then index in _basis of tuple(x1, x2, x3) can be calculated the following expression:
            #   x1 * (dim2 ** 2) * (dim3 ** 2) + x2 * (dim3 ** 2) + x3
            temp_grobal_index = 0
            temp_dim = 1
            for e_sys_position, local_index in enumerate(reversed(index)):
                temp_grobal_index += local_index * temp_dim
                temp_dim = temp_dim * (self._elemental_systems[e_sys_position].dim ** 2)
            return self.basis()[temp_grobal_index]
        else:
            return self.basis()[index]

    def basis_basisconjugate(
        self, basis_index: Union[int, Tuple[int, int]]
    ) -> np.ndarray:
        """returns :math:`B_{\\alpha} \\otimes \\bar{B_{\\beta}}`, where basis_index = :math:`(\\alpha, \\beta)` and :math:`B_{i}` are the elements of basis.

        Parameters
        ----------
        basis_index : Union[int, Tuple[int, int]]
            index of basis.

            - if type is int, then regardes it as the indices (basis_index / num_of_basis, basis_index % num_of_basis) of the basis of CompositeSystem.
            - if type is Tuple, then regardes (i, j) as the indices of the basis of CompositeSystem.

        Returns
        -------
        np.ndarray
            :math:`B_{\\alpha} \\otimes \\bar{B_{\\beta}}`
        """
        # calculate _basis_basisconjugate if it is None
        if self._basis_basisconjugate is None:
            self._basis_basisconjugate = dict()
            basis_no = len(self._total_basis.basis)
            basis = copy.deepcopy(self._total_basis.basis)

            for alpha, beta in itertools.product(range(basis_no), range(basis_no)):
                b_alpha = basis[alpha]
                b_beta_conj = np.conjugate(basis[beta])
                matrix = matrix_util.kron(b_alpha, b_beta_conj)
                self._basis_basisconjugate[(alpha, beta)] = matrix

        # return basis_basisconjugate
        if type(basis_index) == tuple:
            return self._basis_basisconjugate[(basis_index)]
        else:
            basis_index = divmod(basis_index, len(self.basis()))
            return self._basis_basisconjugate[(basis_index)]

    @property
    def dict_from_hs_to_choi(self) -> dict:
        # calculate _dict_from_hs_to_choi if it is None
        if self._dict_from_hs_to_choi is None:
            self._dict_from_hs_to_choi = dict()
            basis_no = len(self._total_basis.basis)
            basis = copy.deepcopy(self._total_basis.basis)

            for alpha, beta in itertools.product(range(basis_no), range(basis_no)):
                b_alpha = basis[alpha]
                b_beta_conj = np.conjugate(basis[beta])
                matrix = matrix_util.kron(b_alpha, b_beta_conj)

                # calc _dict_from_hs_to_choi
                row_indices, column_indices = matrix_util.where_not_zero(matrix)
                for row_index, column_index in zip(row_indices, column_indices):
                    if (row_index, column_index) in self._dict_from_hs_to_choi:
                        self._dict_from_hs_to_choi[(row_index, column_index)].append(
                            (alpha, beta, matrix[row_index, column_index])
                        )
                    else:
                        self._dict_from_hs_to_choi[(row_index, column_index)] = [
                            (alpha, beta, matrix[row_index, column_index])
                        ]

        # return _dict_from_hs_to_choi
        return self._dict_from_hs_to_choi

    def delete_dict_from_hs_to_choi(self) -> None:
        """delete ``dict_from_hs_to_choi`` property to save memory.

        If you use ``dict_from_hs_to_choi`` again, call ``dict_from_hs_to_choi`` again.
        """
        self._dict_from_hs_to_choi = None

    @property
    def dict_from_choi_to_hs(self) -> dict:
        if self._dict_from_choi_to_hs is None:
            self._dict_from_choi_to_hs = dict()
            basis_no = len(self._total_basis.basis)
            basis = copy.deepcopy(self._total_basis.basis)

            for alpha, beta in itertools.product(range(basis_no), range(basis_no)):
                b_alpha = basis[alpha]
                b_beta_conj = np.conjugate(basis[beta])
                matrix = matrix_util.kron(b_alpha, b_beta_conj)

                # calc _dict_from_choi_to_hs
                row_indices, column_indices = matrix_util.where_not_zero(matrix)
                for row_index, column_index in zip(row_indices, column_indices):
                    if (alpha, beta) in self._dict_from_choi_to_hs:
                        self._dict_from_choi_to_hs[(alpha, beta)].append(
                            (row_index, column_index, matrix[row_index, column_index])
                        )
                    else:
                        self._dict_from_choi_to_hs[(alpha, beta)] = [
                            (row_index, column_index, matrix[row_index, column_index])
                        ]

        # return _dict_from_choi_to_hs
        return self._dict_from_choi_to_hs

    def delete_dict_from_choi_to_hs(self) -> None:
        """delete ``dict_from_choi_to_hs`` property to save memory.

        If you use ``dict_from_choi_to_hs`` again, call ``dict_from_choi_to_hs`` again.
        """
        self._dict_from_choi_to_hs = None

    def _calc_basis_sparse(self) -> None:
        basis = copy.deepcopy(self._total_basis.basis)
        basis_tmp = []
        basisconjugate_tmp = []
        for b_alpha in basis:
            basis_tmp.append(matrix_util.flatten(b_alpha))
            basisconjugate_tmp.append(matrix_util.flatten(b_alpha.conjugate()))

        basis_tmp = np.array(basis_tmp)
        self._basis_T_sparse = csr_matrix(basis_tmp.T)
        self._basisconjugate_sparse = csr_matrix(basisconjugate_tmp)

    @property
    def basis_T_sparse(self) -> np.ndarray:
        if self._basis_T_sparse is None:
            self._calc_basis_sparse()
        return self._basis_T_sparse

    def delete_basis_T_sparse(self) -> None:
        """delete ``basis_T_sparse`` property to save memory.

        If you use ``basis_T_sparse`` again, call ``basis_T_sparse`` again.
        """
        self._basis_T_sparse = None

    @property
    def basisconjugate_sparse(self) -> np.ndarray:
        if self._basisconjugate_sparse is None:
            self._calc_basis_sparse()
        return self._basisconjugate_sparse

    def delete_basisconjugate_sparse(self) -> None:
        """delete ``basisconjugate_sparse`` property to save memory.

        If you use ``basisconjugate_sparse`` again, call ``basisconjugate_sparse`` again.
        """
        self._basisconjugate_sparse = None

    def _calc_basis_basisconjugate_sparse(self) -> None:
        basis_no = len(self._total_basis.basis)
        basis = copy.deepcopy(self._total_basis.basis)

        basis_basisconjugate_tmp = []
        basis_basisconjugate_tmp_from_1 = []
        basishermitian_basis_tmp_from_1 = []

        element_size = basis[0].shape[0] ** 2 ** 2
        element_size_2 = basis[0].shape[0] ** 2

        for alpha, beta in itertools.product(range(basis_no), range(basis_no)):
            b_alpha = basis[alpha]
            b_beta_conj = np.conjugate(basis[beta])
            matrix = sparse.kron(b_alpha, b_beta_conj, format="csr")
            reshaped_matrix = matrix.reshape(1, element_size)
            basis_basisconjugate_tmp.append(reshaped_matrix)

            if alpha != 0 and beta != 0:
                basis_basisconjugate_tmp_from_1.append(reshaped_matrix)

                matrix_2 = basis[beta].conj().T @ b_alpha
                basishermitian_basis_tmp_from_1.append(
                    matrix_2.reshape(1, element_size_2)
                )

        # set _basisconjugate_basis_sparse and _basis_basisconjugate_T_sparse
        basis_basisconjugate_tmp = sparse.vstack(basis_basisconjugate_tmp).reshape(
            basis_no ** 2, element_size
        )
        self._basisconjugate_basis_sparse = basis_basisconjugate_tmp.conjugate()
        self._basis_basisconjugate_T_sparse = basis_basisconjugate_tmp.T

        # set _basis_basisconjugate_T_sparse_from_1
        basis_basisconjugate_tmp_from_1 = sparse.vstack(
            basis_basisconjugate_tmp_from_1
        ).reshape((basis_no - 1) ** 2, element_size)
        self._basis_basisconjugate_T_sparse_from_1 = basis_basisconjugate_tmp_from_1.T

        # set _basishermitian_basis_T_from
        basishermitian_basis_tmp_from_1 = sparse.vstack(
            basishermitian_basis_tmp_from_1
        ).reshape((basis_no - 1) ** 2, element_size_2)
        self._basishermitian_basis_T_from_1 = basishermitian_basis_tmp_from_1.T

    @property
    def basisconjugate_basis_sparse(self) -> np.ndarray:
        if self._basisconjugate_basis_sparse is None:
            self._calc_basis_basisconjugate_sparse()
        return self._basisconjugate_basis_sparse

    def delete_basisconjugate_basis_sparse(self) -> None:
        """delete ``basisconjugate_basis_sparse`` property to save memory.

        If you use ``basisconjugate_basis_sparse`` again, call ``basisconjugate_basis_sparse`` again.
        """
        self._basisconjugate_basis_sparse = None

    @property
    def basis_basisconjugate_T_sparse(self) -> np.ndarray:
        if self._basis_basisconjugate_T_sparse is None:
            self._calc_basis_basisconjugate_sparse()
        return self._basis_basisconjugate_T_sparse

    def delete_basis_basisconjugate_T_sparse(self) -> None:
        """delete ``basis_basisconjugate_T_sparse`` property to save memory.

        If you use ``basis_basisconjugate_T_sparse`` again, call ``basis_basisconjugate_T_sparse`` again.
        """
        self._basis_basisconjugate_T_sparse = None

    @property
    def basis_basisconjugate_T_sparse_from_1(self) -> np.ndarray:
        if self._basis_basisconjugate_T_sparse_from_1 is None:
            self._calc_basis_basisconjugate_sparse()
        return self._basis_basisconjugate_T_sparse_from_1

    def delete_basis_basisconjugate_T_sparse_from_1(self) -> None:
        """delete ``basis_basisconjugate_T_sparse_from_1`` property to save memory.

        If you use ``basis_basisconjugate_T_sparse_from_1`` again, call ``basis_basisconjugate_T_sparse_from_1`` again.
        """
        self._basis_basisconjugate_T_sparse_from_1 = None

    @property
    def basishermitian_basis_T_from_1(self) -> np.ndarray:
        if self._basishermitian_basis_T_from_1 is None:
            self._calc_basis_basisconjugate_sparse()
        return self._basishermitian_basis_T_from_1

    def delete_basishermitian_basis_T_from_1(self) -> None:
        """delete ``basishermitian_basis_T_from_1`` property to save memory.

        If you use ``basishermitian_basis_T_from_1`` again, call ``basishermitian_basis_T_from_1`` again.
        """
        self._basishermitian_basis_T_from_1 = None

    @property
    def elemental_systems(self) -> Tuple[ElementalSystem]:
        """returns list of ElementalSystem of this CompositeSystem.

        Returns
        -------
        Tuple[ElementalSystem]
            list of ElementalSystem of this CompositeSystem.
        """
        return self._elemental_systems

    @property
    def is_orthonormal_hermitian_0thprop_identity(self) -> bool:
        """returns whether all ElementalSystem of this CompositeSystem are orthonormal, hermitian and 0th prop identity.

        Returns
        -------
        bool
            whether all ElementalSystem of this CompositeSystem are orthonormal, hermitian and 0th prop identity.
        """
        return self._is_orthonormal_hermitian_0thprop_identity

    @property
    def is_basis_hermitian(self) -> bool:
        return self._is_basis_hermitian

    def __len__(self) -> int:
        return len(self._elemental_systems)

    def __getitem__(self, key: int) -> ElementalSystem:
        return self._elemental_systems[key]

    def __iter__(self):
        return iter(self._elemental_systems)

    def __eq__(self, other) -> bool:
        if not isinstance(other, CompositeSystem):
            return False

        if len(self) != len(other):
            return False

        for s, o in zip(self, other):
            if s is not o:
                return False
        return True

    def __str__(self):
        desc = "elemental_systems:\n"
        for i, e_sys in enumerate(self._elemental_systems):
            desc += f"[{i}] {e_sys.name} (system_id={e_sys.system_id})\n"

        desc += "\n"
        desc += f"dim: {self.dim}\n"
        desc += f"basis:\n"
        desc += str(self.basis())
        return desc

    def __repr__(self):
        return f"{self.__class__.__name__}(systems={repr(self.elemental_systems)})"
