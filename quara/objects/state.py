import itertools
from typing import List

import numpy as np

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import MatrixBasis, get_normalized_pauli_basis
import quara.utils.matrix_util as mutil


class State:
    def __init__(self, c_sys: CompositeSystem, vec: np.ndarray):
        self._composite_system: CompositeSystem = c_sys
        self._vec: np.ndarray = vec
        size = self._vec.shape

        # whether vec is one-dimensional array
        if len(size) != 1:
            raise ValueError(f"vec must be one-dimensional array. shape is {size}")

        # whether size of vec is square number
        self._dim = int(np.sqrt(size[0]))
        if self._dim ** 2 != size[0]:
            raise ValueError(
                f"size of vec must be square number. dim of vec is {size[0]}"
            )

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
    def vec(self):
        """returns vector.
        
        Returns
        -------
        np.array
            vector
        """
        return self._vec

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
        converted_vec = convert_vec(
            self._vec, self._composite_system.basis, other_basis
        )
        return converted_vec


def convert_vec(
    from_vec: np.array, from_basis: MatrixBasis, to_basis: MatrixBasis
) -> np.array:
    """converts vector representation from ``from_basis`` to ``to_basis``.
    
    Parameters
    ----------
    from_vec : np.array
        vector representation before converts vector representation
    from_basis : MatrixBasis
        basis before converts vector representation
    to_basis : MatrixBasis
        basis after converts vector representation
    
    Returns
    -------
    np.array
        vector representation after converts vector representation
        ``dtype`` is ``float64``
    """
    # whether length of from_basis equals length of to_basis
    if len(from_basis) != len(to_basis):
        raise ValueError(
            f"length of from_basis must equal length of to_basis. length of from_basis={len(from_basis)}. length of to_basis is {len(to_basis)}"
        )
    len_basis = len(from_basis)

    # whether entries of from_vec are real numbers
    if from_vec.dtype != np.float64:
        raise ValueError(
            f"entries of from_vec must be real numbers. dtype of from_vec is {from_vec._vec.dtype}"
        )

    # whether dim of from_basis equals dim of to_basis
    if from_basis.dim != to_basis.dim:
        raise ValueError(
            f"dim of from_basis must equal dim of to_basis. dim of from_basis={from_basis.dim}. dim of to_basis is {to_basis.dim}"
        )

    # "converted_vec"_{\alpha} = \sum_{\beta} Tr["to_basis"_{\beta}^{\dagger} "from_basis"_{\alpha}] "from_vec"_{\alpha}
    representation_matrix = [
        mutil.inner_product(val1, val2)
        for val1, val2 in itertools.product(to_basis, from_basis)
    ]
    rep_mat = np.array(representation_matrix).reshape(len_basis, len_basis)
    converted_vec = rep_mat @ from_vec
    # return converted_vec.real.astype(np.float64)
    return converted_vec


def get_x0_1q_with_normalized_pauli_basis(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``X_0`` with normalized pauli basis.
    
    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state
    
    Returns
    -------
    np.array
        vec of state
    """
    # convert "vec in Pauli basis" to "vec in basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, 1, 0, 0], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis)
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_x1_1q_with_normalized_pauli_basis(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``X_1`` with normalized pauli basis.
    
    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state
    
    Returns
    -------
    np.array
        vec of state
    """
    # convert "vec in Pauli basis" to "vec in basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, -1, 0, 0], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis)
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_y0_1q_with_normalized_pauli_basis(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``Y_0`` with normalized pauli basis.
    
    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state
    
    Returns
    -------
    np.array
        vec of state
    """
    # convert "vec in Pauli basis" to "vec in basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, 0, 1, 0], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis)
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_y1_1q_with_normalized_pauli_basis(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``Y_1`` with normalized pauli basis.
    
    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state
    
    Returns
    -------
    np.array
        vec of state
    """
    # convert "vec in Pauli basis" to "vec in basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, 0, -1, 0], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis)
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_z0_1q_with_normalized_pauli_basis(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``Z_0`` with normalized pauli basis.
    
    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state
    
    Returns
    -------
    np.array
        vec of state
    """
    # convert "vec in Pauli basis" to "vec in basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis)
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_z1_1q_with_normalized_pauli_basis(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``Z_1`` with normalized pauli basis.
    
    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state
    
    Returns
    -------
    np.array
        vec of state
    """
    # convert "vec in Pauli basis" to "vec in basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis)
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


# TODO 2qubit. \frac{1}{\sqrt{2}}(|00>+|11>)
