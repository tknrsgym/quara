import copy
import itertools
from typing import List, Tuple, Union

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import MatrixBasis, get_comp_basis


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
            self._total_basis = self._elemental_systems[0].basis
        else:
            basis_list = [e_sys.basis for e_sys in self._elemental_systems]
            temp = basis_list[0]
            for elem in basis_list[1:]:
                temp = [
                    np.kron(val1, val2) for val1, val2 in itertools.product(temp, elem)
                ]
            self._total_basis = MatrixBasis(temp)

        # calculate np.kron(basis, basisconjugate)
        basis_no = len(self._total_basis.basis)
        hs = np.zeros((basis_no, basis_no), dtype=np.float64)
        basis = copy.deepcopy(self._total_basis.basis)

        self._basis_basisconjugate = dict()
        self._dict_from_hs_to_choi = dict()
        self._dict_from_choi_to_hs = dict()
        basis_tmp = []
        basisconjugate_tmp = []
        basis_basisconjugate_tmp = []
        for b_alpha in basis:
            basis_tmp.append(b_alpha.flatten())
            basisconjugate_tmp.append(b_alpha.conjugate().flatten())
        for alpha, beta in itertools.product(range(basis_no), range(basis_no)):
            b_alpha = basis[alpha]
            b_beta_conj = np.conjugate(basis[beta])
            matrix = np.kron(b_alpha, b_beta_conj)
            self._basis_basisconjugate[(alpha, beta)] = matrix
            basis_basisconjugate_tmp.append(matrix.flatten())

            # calc _dict_from_hs_to_choi and _dict_from_choi_to_hs
            row_indices, column_indices = np.where(matrix != 0)
            for row_index, column_index in zip(row_indices, column_indices):
                # _dict_from_hs_to_choi
                if (row_index, column_index) in self._dict_from_hs_to_choi:
                    self._dict_from_hs_to_choi[(row_index, column_index)].append(
                        (alpha, beta, matrix[row_index, column_index])
                    )
                else:
                    self._dict_from_hs_to_choi[(row_index, column_index)] = [
                        (alpha, beta, matrix[row_index, column_index])
                    ]

                # _dict_from_choi_to_hs
                if (alpha, beta) in self._dict_from_choi_to_hs:
                    self._dict_from_choi_to_hs[(alpha, beta)].append(
                        (row_index, column_index, matrix[row_index, column_index])
                    )
                else:
                    self._dict_from_choi_to_hs[(alpha, beta)] = [
                        (row_index, column_index, matrix[row_index, column_index])
                    ]
        basis_tmp = np.array(basis_tmp)
        self._basis_T_sparse = csr_matrix(basis_tmp.T)
        self._basisconjugate_sparse = csr_matrix(basisconjugate_tmp)
        basis_basisconjugate_tmp = np.array(basis_basisconjugate_tmp)
        self._basisconjugate_basis_sparse = csr_matrix(
            basis_basisconjugate_tmp.conjugate()
        )
        self._basis_basisconjugate_T_sparse = csr_matrix(basis_basisconjugate_tmp.T)

    def comp_basis(self, mode: str = "row_major") -> MatrixBasis:
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
        basis_tmp: MatrixBasis

        if len(self._elemental_systems) == 1:
            basis_tmp = self._elemental_systems[0].comp_basis(mode=mode)
        else:
            basis_tmp = get_comp_basis(self.dim, mode=mode)
        return basis_tmp

    def basis(self) -> MatrixBasis:
        """returns MatrixBasis of CompositeSystem.

        Returns
        -------
        MatrixBasis
            MatrixBasis of CompositeSystem.
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

    def get_basis(self, index: Union[int, Tuple]) -> MatrixBasis:
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
        if type(basis_index) == tuple:
            return self._basis_basisconjugate[(basis_index)]
        else:
            basis_index = divmod(basis_index, len(self.basis()))
            return self._basis_basisconjugate[(basis_index)]

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
