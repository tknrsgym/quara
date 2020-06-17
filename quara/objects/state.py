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

from quara.objects.qoperation import QOperation


class State(QOperation):
    def __init__(
        self,
        c_sys: CompositeSystem,
        vec: np.ndarray,
        is_physicality_required: bool = True,
        is_estimation_object: bool = True,
        on_para_eq_constraint: bool = True,
        on_algo_eq_constraint: bool = True,
        on_algo_ineq_constraint: bool = True,
        eps_proj_physical: float = 10 ** (-4),
    ):
        """Constructor

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem of this state.
        vec : np.ndarray
            vec of this state.
        is_physicality_required : bool, optional
            checks whether the state is physically correct, by default True.
            all of the following conditions are ``True``, the state is physically correct:

            - trace of density matrix equals 1.
            - density matrix is Hermitian.
            - density matrix is positive semidefinite.

            If you want to ignore the above requirements and create a State object, set ``is_physicality_required`` to ``False``.

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
            ``is_physicality_required`` is ``True`` and the state is not physically correct.
        """
        super().__init__(
            c_sys=c_sys,
            is_physicality_required=is_physicality_required,
            is_estimation_object=is_estimation_object,
            on_para_eq_constraint=on_para_eq_constraint,
            on_algo_eq_constraint=on_algo_eq_constraint,
            on_algo_ineq_constraint=on_algo_ineq_constraint,
            eps_proj_physical=eps_proj_physical,
        )
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

        # whether the state is physically correct
        if self.is_physicality_required and not self.is_physical():
            raise ValueError("the state is not physically correct.")

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

    def is_physical(self) -> bool:
        """returns whether the state is physically correct.

        all of the following conditions are ``True``, the state is physically correct:

        - trace of density matrix equals 1.
        - density matrix is Hermitian.
        - density matrix is positive semidefinite.

        Returns
        -------
        bool
            whether the state is physically correct.
        """
        # in `is_positive_semidefinite` function, the state is checked whether it is Hermitian.
        # therefore, do not call the `is_hermitian` function explicitly.
        return self.is_trace_one() and self.is_positive_semidefinite()

    def set_zero(self):
        self._vec = np.zeros(self._vec.shape, dtype=np.float64)
        self._is_physicality_required = False

    def zero_obj(self):
        new_vec = np.zeros(self.vec.shape, dtype=np.float64)
        state = State(
            c_sys=self._composite_system,
            vec=new_vec,
            is_physicality_required=False,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            eps_proj_physical=self.eps_proj_physical,
        )
        return state

    def to_var(self) -> np.array:
        return convert_state_to_var(
            c_sys=self._composite_system,
            vec=self.vec,
            on_para_eq_constraint=self.on_para_eq_constraint,
        )

    def to_stacked_vector(self) -> np.array:
        return self._vec

    def calc_gradient(self):
        raise NotImplementedError()

    def calc_proj_eq_constraint(self):
        raise NotImplementedError()

    def calc_proj_ineq_constraint(self):
        raise NotImplementedError()

    def calc_proj_physical(self):
        raise NotImplementedError()

    def __add__(self, other):
        raise NotImplementedError()

    def __sub__(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        # self * other
        raise NotImplementedError()

    def __rmul__(self, other):
        # other * self
        raise NotImplementedError()

    def to_density_matrix(self) -> np.ndarray:
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
        tr = np.trace(self.to_density_matrix())
        return np.isclose(tr, 1, atol=Settings.get_atol())

    def is_hermitian(self) -> bool:
        """returns whether density matrix is Hermitian.

        Returns
        -------
        bool
        True where density matrix, False otherwise.
        """
        return mutil.is_hermitian(self.to_density_matrix())

    def is_positive_semidefinite(self) -> bool:
        """returns whether density matrix is positive semidifinite.

        Returns
        -------
        bool
            True where density matrix is positive semidifinite, False otherwise.
        """
        return mutil.is_positive_semidefinite(self.to_density_matrix())

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
        return np.linalg.eigvals(self.to_density_matrix())

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


def convert_var_index_to_state_index(
    var_index: int, on_para_eq_constraint: bool = True
) -> int:
    """converts variable index to state index.

    Parameters
    ----------
    var_index : int
        variable index.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    int
        state index.
    """
    state_index = var_index + 1 if on_para_eq_constraint else var_index
    return state_index


def convert_state_index_to_var_index(
    state_index: int, on_para_eq_constraint: bool = True
) -> int:
    """converts state index to variable index.

    Parameters
    ----------
    state_index : int
        state index.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    int
        variable index.
    """
    var_index = state_index - 1 if on_para_eq_constraint else state_index
    return var_index


def convert_var_to_state(
    c_sys: CompositeSystem, var: np.ndarray, on_para_eq_constraint: bool = True
) -> State:
    """converts vec of variables to state.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this state.
    var : np.ndarray
        vec of variables.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    State
        converted state.
    """
    vec = np.insert(var, 0, 1 / np.sqrt(c_sys.dim)) if on_para_eq_constraint else var
    state = State(c_sys, vec, is_physicality_required=False)
    return state


def convert_state_to_var(
    c_sys: CompositeSystem, vec: np.ndarray, on_para_eq_constraint: bool = True
) -> np.array:
    """converts vec of state to vec of variables.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this state.
    vec : np.ndarray
        vec of state.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    np.array
        vec of variables.
    """
    var = np.delete(vec, 0) if on_para_eq_constraint else vec
    return var


def calc_gradient_from_state(
    c_sys: CompositeSystem,
    vec: np.ndarray,
    var_index: int,
    on_para_eq_constraint: bool = True,
) -> State:
    """calculates gradient from State.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this state.
    vec : np.ndarray
        vec of state.
    var_index : int
        variable index.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    State
        State with gradient as vec.
    """
    gradient = np.zeros((c_sys.dim ** 2), dtype=np.float64)
    state_index = convert_var_index_to_state_index(var_index, on_para_eq_constraint)
    gradient[state_index] = 1

    state = State(c_sys, gradient, is_physicality_required=False)
    return state


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
