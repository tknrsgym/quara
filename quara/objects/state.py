import itertools
from typing import List

import numpy as np

import quara.utils.matrix_util as mutil
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import (
    MatrixBasis,
    convert_vec,
    get_normalized_pauli_basis,
)
from quara.settings import Settings


class State:
    def __init__(
        self, c_sys: CompositeSystem, vec: np.ndarray, is_physical: bool = True
    ):
        """Constructor

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem of this state.
        vec : np.ndarray
            vec of this state.
        is_physical : bool, optional
            checks whether the state is physically wrong, by default True.
            if at least one of the following conditions is ``False``, the state is physically wrong:

            - density matrix is positive semidefinite.
            - trace of density matrix equals 1.

        Raises
        ------
        ValueError
            vec is not one-dimensional array.
        ValueError
            size of vec is not square number.
        ValueError
            entries of vec are not real numbers.
        ValueError
            dim of CompositeSystem does not equal dim of vec.
        ValueError
            ``is_physical`` is ``True`` and density matrix is not positive semidefinite.
        ValueError
            ``is_physical`` is ``True`` and trace of density matrix does not equal 1.
        """
        self._composite_system: CompositeSystem = c_sys
        self._vec: np.ndarray = vec
        size = self._vec.shape
        self._is_physical = is_physical

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

        # whether the state is physically wrong
        if self._is_physical:
            if not self.is_positive_semidefinite():
                raise ValueError(
                    "the state is physically wrong. density matrix is not positive semidefinite."
                )
            elif not self.is_trace_one():
                raise ValueError(
                    "the state is physically wrong. trace of density matrix does not equal 1."
                )

    @property
    def vec(self):
        """returns vec of this state.

        Returns
        -------
        np.array
            vec of this state.
        """
        return self._vec

    @property
    def dim(self):
        """returns dim of this state.

        Returns
        -------
        int
            dim of this state.
        """
        return self._dim

    @property
    def is_physical(self):
        """returns argument ``is_physical`` specified in the constructor.

        Returns
        -------
        int
            argument ``is_physical`` specified in the constructor.
        """
        return self._is_physical

    def get_density_matrix(self) -> np.ndarray:
        """returns density matrix.

        Returns
        -------
        int
            density matrix.
        """
        density = np.zeros((self._dim, self._dim), dtype=np.complex128)
        for coefficient, basis in zip(self._vec, self._composite_system.basis()):
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
        return np.isclose(tr, 1, atol=Settings.get_atol())

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

    def calc_eigenvalues(self) -> List:
        """calculates eigen values of density matrix.

        this function uses numpy API.
        see this URL for details:
        https://numpy.org/doc/1.18/reference/generated/numpy.linalg.eigvals.html

        Returns
        -------
        List
            eigen values of density matrix.
        """
        return np.linalg.eigvals(self.get_density_matrix())

    def convert_basis(self, other_basis: MatrixBasis) -> np.array:
        """returns vector representation for ``other_basis``.

        Parameters
        ----------
        other_basis : MatrixBasis
            basis.

        Returns
        -------
        np.array
            vector representation for ``other_basis``.
        """
        converted_vec = convert_vec(
            self._vec, self._composite_system.basis(), other_basis
        )
        return converted_vec


def get_x0_1q(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``X_0`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.array
        vec of state.
    """
    # whether dim of CompositeSystem equals 2
    if c_sys.dim != 2:
        raise ValueError(
            f"dim of CompositeSystem must equals 2.  dim of CompositeSystem is {c_sys.dim}"
        )

    # convert "vec in Pauli basis" to "vec in the basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, 1, 0, 0], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis())
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_x1_1q(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``X_1`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.array
        vec of state.
    """
    # whether dim of CompositeSystem equals 2
    if c_sys.dim != 2:
        raise ValueError(
            f"dim of CompositeSystem must equals 2.  dim of CompositeSystem is {c_sys.dim}"
        )

    # convert "vec in Pauli basis" to "vec in the basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, -1, 0, 0], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis())
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_y0_1q(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``Y_0`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.array
        vec of state.
    """
    # whether dim of CompositeSystem equals 2
    if c_sys.dim != 2:
        raise ValueError(
            f"dim of CompositeSystem must equals 2.  dim of CompositeSystem is {c_sys.dim}"
        )

    # convert "vec in Pauli basis" to "vec in the basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, 0, 1, 0], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis())
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_y1_1q(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``Y_1`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.array
        vec of state.
    """
    # whether dim of CompositeSystem equals 2
    if c_sys.dim != 2:
        raise ValueError(
            f"dim of CompositeSystem must equals 2.  dim of CompositeSystem is {c_sys.dim}"
        )

    # convert "vec in Pauli basis" to "vec in the basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, 0, -1, 0], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis())
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_z0_1q(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``Z_0`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.array
        vec of state.
    """
    # whether dim of CompositeSystem equals 2
    if c_sys.dim != 2:
        raise ValueError(
            f"dim of CompositeSystem must equals 2.  dim of CompositeSystem is {c_sys.dim}"
        )

    # convert "vec in Pauli basis" to "vec in the basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis())
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_z1_1q(c_sys: CompositeSystem) -> np.array:
    """returns vec of state ``Z_1`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.array
        vec of state.
    """
    # whether dim of CompositeSystem equals 2
    if c_sys.dim != 2:
        raise ValueError(
            f"dim of CompositeSystem must equals 2.  dim of CompositeSystem is {c_sys.dim}"
        )

    # convert "vec in Pauli basis" to "vec in the basis of CompositeSystem"
    from_vec = 1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.float64)
    from_basis = get_normalized_pauli_basis()
    to_vec = convert_vec(from_vec, from_basis, c_sys.basis())
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state


def get_bell_2q(c_sys: CompositeSystem) -> State:
    """returns vec of Bell state, \frac{1}{2}(|00>+|11>)(<00|+<11|), with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    State
        vec of state.
    """
    # whether dim of CompositeSystem equals 4
    if c_sys.dim != 4:
        raise ValueError(
            f"dim of CompositeSystem must equals 4.  dim of CompositeSystem is {c_sys.dim}"
        )

    # \frac{1}{2}(|00>+|11>)(<00|+<11|) = \frac{1}{2}(|0><0|\otimes|0><0| + |0><1|\otimes|0><1| + |1><0|\otimes|1><0| + |1><1|\otimes|1><1|)
    # convert "vec in comp basis" to "vec in basis of CompositeSystem"
    from_vec = (
        np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=np.float64) / 2
    )
    to_vec = convert_vec(from_vec, c_sys.comp_basis(), c_sys.basis())
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state
