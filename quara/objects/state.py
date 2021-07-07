import copy
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
        mode_proj_order: str = "eq_ineq",
        eps_proj_physical: float = None,
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
            mode_proj_order=mode_proj_order,
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
        if self.composite_system.dim != self._dim:
            raise ValueError(
                f"dim of CompositeSystem must equal dim of vec. dim of CompositeSystem is {self.composite_system.dim}. dim of vec is {self._dim}"
            )

        # whether the state is physically correct
        if self.is_physicality_required and not self.is_physical():
            raise ValueError("the state is not physically correct.")

    def _info(self):
        info = {}
        info["Type"] = self.__class__.__name__
        info["Dim"] = self.dim
        info["Vec"] = self.vec
        return info

    @property
    def vec(self):
        """returns vec of this state.

        Returns
        -------
        np.ndarray
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

    def is_eq_constraint_satisfied(self, atol: float = None):
        return self.is_trace_one(atol)

    def is_ineq_constraint_satisfied(self, atol: float = None):
        return self.is_positive_semidefinite(atol)

    def set_zero(self):
        """sets parameters to zero."""
        self._vec = np.zeros(self._vec.shape, dtype=np.float64)
        self._is_physicality_required = False

    def _generate_zero_obj(self):
        new_vec = np.zeros(self.vec.shape, dtype=np.float64)
        return new_vec

    def _generate_origin_obj(self):
        new_vec = np.zeros(self.vec.shape, dtype=np.float64)
        new_vec[0] = 1 / np.sqrt(self.dim)
        return new_vec

    def _copy(self):
        return copy.deepcopy(self.vec)

    def _add_vec(self, other):
        return self.vec + other.vec

    def _sub_vec(self, other):
        return self.vec - other.vec

    def _mul_vec(self, other):
        # self * other
        return self.vec * other

    def _truediv_vec(self, other):
        # self / other
        with np.errstate(divide="ignore"):
            return self.vec / other

    def to_var(self) -> np.ndarray:
        """converts State to variables.

        Returns
        -------
        np.ndarray
            variable representation of State.
        """
        return convert_vec_to_var(
            c_sys=self.composite_system,
            vec=self.vec,
            on_para_eq_constraint=self.on_para_eq_constraint,
        )

    def to_stacked_vector(self) -> np.ndarray:
        """converts State to stacked vector.

        Returns
        -------
        np.ndarray
            stacked vector representation of State.
        """
        return self._vec

    def calc_gradient(self, var_index: int) -> "State":
        """calculates gradient of State.

        Parameters
        ----------
        var_index : int
            index of variables to calculate gradient.

        Returns
        -------
        State
            gradient of State.
        """
        state = calc_gradient_from_state(
            self.composite_system,
            self.vec,
            var_index,
            is_estimation_object=self.is_estimation_object,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
        )
        return state

    def calc_proj_eq_constraint(self) -> "State":
        """calculates the projection of State on equal constraint.

        Returns
        -------
        State
            the projection of State on equal constraint.
        """
        vec = copy.deepcopy(self.vec)
        vec[0] = 1 / np.sqrt(self.dim)
        state = State(
            self.composite_system,
            vec,
            is_physicality_required=self.is_physicality_required,
            is_estimation_object=self.is_estimation_object,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
        )
        return state

    @staticmethod
    def calc_proj_eq_constraint_with_var(
        c_sys: CompositeSystem,
        var: np.ndarray,
        on_para_eq_constraint: bool = True,
    ) -> np.ndarray:
        """calculates the projection of State on equal constraint.

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem of this variables.
        var : np.ndarray
            variables.
        on_para_eq_constraint : bool, optional
            whether this variables is on parameter equality constraint, by default True.

        Returns
        -------
        np.ndarray
            the projection of State on equal constraint.
        """
        if on_para_eq_constraint:
            new_var = var
        else:
            new_var = copy.deepcopy(var)
            new_var[0] = 1 / np.sqrt(c_sys.dim)

        return new_var

    def calc_proj_ineq_constraint(self) -> "State":
        """calculates the projection of State on inequal constraint.

        Returns
        -------
        State
            the projection of State on inequal constraint.
        """
        # calc engenvalues and engenvectors
        density_matrix_orig = self.to_density_matrix_with_sparsity()
        eigenvals, eigenvecs = np.linalg.eigh(density_matrix_orig)

        # project
        diag = np.diag(eigenvals)
        diag[diag < 0] = 0

        # calc new vec
        new_density_matrix = eigenvecs @ diag @ eigenvecs.T.conjugate()
        vec_new = to_vec_from_density_matrix_with_sparsity(
            self.composite_system, new_density_matrix
        )

        # create new State
        state = State(
            self.composite_system,
            np.array(vec_new, dtype=np.float64),
            is_physicality_required=self.is_physicality_required,
            is_estimation_object=self.is_estimation_object,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
        )
        return state

    @staticmethod
    def calc_proj_ineq_constraint_with_var(
        c_sys: CompositeSystem,
        var: np.ndarray,
        on_para_eq_constraint: bool = True,
    ) -> np.ndarray:
        """calculates the projection of State on inequal constraint.

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem of this variables.
        var : np.ndarray
            variables.
        on_para_eq_constraint : bool, optional
            whether this variables is on parameter equality constraint, by default True.

        Returns
        -------
        np.ndarray
            the projection of State on inequal constraint.
        """
        # calc engenvalues and engenvectors
        density_matrix = to_density_matrix_from_var(c_sys, var, on_para_eq_constraint)
        eigenvals, eigenvecs = np.linalg.eigh(density_matrix)

        # project
        diag = np.diag(eigenvals)
        diag[diag < 0] = 0

        # calc new vec
        new_density_matrix = eigenvecs @ diag @ eigenvecs.T.conjugate()
        new_vec = to_vec_from_density_matrix_with_sparsity(c_sys, new_density_matrix)

        # vec to var
        new_var = convert_vec_to_var(c_sys, new_vec, on_para_eq_constraint)
        return new_var

    def to_density_matrix(self) -> np.ndarray:
        """returns density matrix.

        Returns
        -------
        int
            density matrix.
        """
        density = np.zeros((self._dim, self._dim), dtype=np.complex128)
        for coefficient, basis in zip(self._vec, self.composite_system.basis()):
            density += coefficient * basis
        return density

    def to_density_matrix_with_sparsity(self) -> np.ndarray:
        """returns density matrix.

        this function uses the scipy.sparse module.

        Returns
        -------
        int
            density matrix.
        """
        return to_density_matrix_from_vec(self.composite_system, self.vec)

    def is_trace_one(self, atol: float = None) -> bool:
        """returns whether trace of density matrix is one.

        Parameters
        ----------
        atol : float, optional
            the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.

        Returns
        -------
        bool
            True where trace of density matrix is one, False otherwise.
        """
        atol = Settings.get_atol() if atol is None else atol
        tr = np.trace(self.to_density_matrix_with_sparsity())
        return np.isclose(tr, 1, atol=atol)

    def is_hermitian(self, atol: float = None) -> bool:
        """returns whether density matrix is Hermitian.

        Parameters
        ----------
        atol : float, optional
            the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.

        Returns
        -------
        bool
        True where density matrix, False otherwise.
        """
        return mutil.is_hermitian(self.to_density_matrix_with_sparsity(), atol=atol)

    def is_positive_semidefinite(self, atol: float = None) -> bool:
        """returns whether density matrix is positive semidifinite.

        Parameters
        ----------
        atol : float, optional
            the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.

        Returns
        -------
        bool
            True where density matrix is positive semidifinite, False otherwise.
        """
        return mutil.is_positive_semidefinite(
            self.to_density_matrix_with_sparsity(), atol=atol
        )

    def calc_eigenvalues(self) -> List:
        """calculates eigen values of density matrix.

        this function uses numpy API.
        see this URL for details:
        https://numpy.org/doc/1.18/reference/generated/numpy.linalg.eigvalsh.html

        Returns
        -------
        List
            eigen values of density matrix.
        """
        if self.composite_system.is_basis_hermitian:
            values = np.linalg.eigvalsh(self.to_density_matrix_with_sparsity())
        else:
            values = np.linalg.eigvals(self.to_density_matrix_with_sparsity())
        values = sorted(values, reverse=True)
        return values

    def convert_basis(self, other_basis: MatrixBasis) -> np.ndarray:
        """returns vector representation for ``other_basis``.

        Parameters
        ----------
        other_basis : MatrixBasis
            basis.

        Returns
        -------
        np.ndarray
            vector representation for ``other_basis``.
        """
        converted_vec = convert_vec(
            self._vec, self.composite_system.basis(), other_basis
        )
        return converted_vec

    def _generate_from_var_func(self):
        return convert_var_to_state

    @staticmethod
    def convert_var_to_stacked_vector(
        c_sys: CompositeSystem,
        var: np.ndarray,
        on_para_eq_constraint: bool = True,
    ) -> np.ndarray:
        """converts variables of state to stacked vector of state.

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem of this state.
        var : np.ndarray
            variables of state.
        on_para_eq_constraint : bool, optional
            uses equal constraints, by default True.

        Returns
        -------
        np.ndarray
            stacked vector of state.
        """
        return convert_var_to_vec(c_sys, var, on_para_eq_constraint)

    @staticmethod
    def convert_stacked_vector_to_var(
        c_sys: CompositeSystem,
        stacked_vector: np.ndarray,
        on_para_eq_constraint: bool = True,
    ) -> np.ndarray:
        """converts stacked vector of state to variables of state.

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem of this state.
        stacked_vector : np.ndarray
            stacked vector of state.
        on_para_eq_constraint : bool, optional
            uses equal constraints, by default True.

        Returns
        -------
        np.ndarray
            variables of state.
        """
        return convert_vec_to_var(c_sys, stacked_vector, on_para_eq_constraint)


def to_density_matrix_from_vec(c_sys: CompositeSystem, vec: np.ndarray) -> np.ndarray:
    """converts vec to density matrix.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this state.
    vec : np.ndarray
        vec of state of this state.

    Returns
    -------
    np.ndarray
        density matrix of this state.
    """
    density_vec = c_sys._basis_T_sparse.dot(vec)
    density = density_vec.reshape((c_sys.dim, c_sys.dim))
    return density


def to_vec_from_density_matrix_with_sparsity(
    c_sys: CompositeSystem,
    density_matrix: np.ndarray,
) -> np.ndarray:
    """converts density matrix to vec.

    this function uses the scipy.sparse module.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this state.
    density_matrix : np.ndarray
        density matrix of this state.

    Returns
    -------
    np.ndarray
        vec of variables.
    """
    vec = c_sys._basisconjugate_sparse.dot(density_matrix.flatten())
    return mutil.truncate_hs(vec)


def to_density_matrix_from_var(
    c_sys: CompositeSystem,
    var: np.ndarray,
    on_para_eq_constraint: bool = True,
) -> np.ndarray:
    """converts var to density matrix.

    this function uses the scipy.sparse module.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this state.
    var : np.ndarray
        variables.
    on_para_eq_constraint : bool, optional
        whether this state is on parameter equality constraint, by default True

    Returns
    -------
    np.ndarray
        density matrix of this state.
    """
    # var to vec
    vec = convert_var_to_vec(c_sys, var, on_para_eq_constraint)

    # vec to density matrix
    density = to_density_matrix_from_vec(c_sys, vec)
    return density


def to_var_from_density_matrix(
    c_sys: CompositeSystem,
    density_matrix: np.ndarray,
    on_para_eq_constraint: bool = True,
) -> np.ndarray:
    """converts density matrix to variables.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this state.
    density_matrix : np.ndarray
        density matrix of this state.
    on_para_eq_constraint : bool, optional
        whether this state is on parameter equality constraint, by default True

    Returns
    -------
    np.ndarray
        variables.
    """
    # density matrix to vec
    vec = to_vec_from_density_matrix_with_sparsity(c_sys, density_matrix)

    # vec to var
    var = convert_vec_to_var(c_sys, vec, on_para_eq_constraint)
    return var


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


def convert_var_to_vec(
    c_sys: CompositeSystem,
    var: np.ndarray,
    on_para_eq_constraint: bool = True,
) -> np.ndarray:
    """converts variables of state to vec of state.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this state.
    var : np.ndarray
        variables of state.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    np.ndarray
        vec of state.
    """
    vec = np.insert(var, 0, 1 / np.sqrt(c_sys.dim)) if on_para_eq_constraint else var
    return vec


def convert_var_to_state(
    c_sys: CompositeSystem,
    var: np.ndarray,
    is_physicality_required: bool = True,
    is_estimation_object: bool = True,
    on_para_eq_constraint: bool = True,
    on_algo_eq_constraint: bool = True,
    on_algo_ineq_constraint: bool = True,
    mode_proj_order: str = "eq_ineq",
    eps_proj_physical: float = 10 ** (-4),
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
    vec = convert_var_to_vec(c_sys, var, on_para_eq_constraint)
    state = State(
        c_sys,
        vec,
        is_physicality_required=is_physicality_required,
        is_estimation_object=is_estimation_object,
        on_para_eq_constraint=on_para_eq_constraint,
        on_algo_eq_constraint=on_algo_eq_constraint,
        on_algo_ineq_constraint=on_algo_ineq_constraint,
        mode_proj_order=mode_proj_order,
        eps_proj_physical=eps_proj_physical,
    )
    return state


def convert_vec_to_var(
    c_sys: CompositeSystem, vec: np.ndarray, on_para_eq_constraint: bool = True
) -> np.ndarray:
    """converts vec of state to variables of state.

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
    np.ndarray
        variables of state.
    """
    var = np.delete(vec, 0) if on_para_eq_constraint else vec
    return var


def calc_gradient_from_state(
    c_sys: CompositeSystem,
    vec: np.ndarray,
    var_index: int,
    is_estimation_object: bool = True,
    on_para_eq_constraint: bool = True,
    on_algo_eq_constraint: bool = True,
    on_algo_ineq_constraint: bool = True,
    mode_proj_order: str = "eq_ineq",
    eps_proj_physical: float = 10 ** (-4),
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

    state = State(
        c_sys,
        gradient,
        is_physicality_required=False,
        is_estimation_object=is_estimation_object,
        on_para_eq_constraint=on_para_eq_constraint,
        on_algo_eq_constraint=on_algo_eq_constraint,
        on_algo_ineq_constraint=on_algo_ineq_constraint,
        mode_proj_order=mode_proj_order,
        eps_proj_physical=eps_proj_physical,
    )
    return state


def get_x0_1q(c_sys: CompositeSystem) -> np.ndarray:
    """returns vec of state ``X_0`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.ndarray
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


def get_x1_1q(c_sys: CompositeSystem) -> np.ndarray:
    """returns vec of state ``X_1`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.ndarray
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


def get_y0_1q(c_sys: CompositeSystem) -> np.ndarray:
    """returns vec of state ``Y_0`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.ndarray
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


def get_y1_1q(c_sys: CompositeSystem) -> np.ndarray:
    """returns vec of state ``Y_1`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.ndarray
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


def get_z0_1q(c_sys: CompositeSystem) -> np.ndarray:
    """returns vec of state ``Z_0`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.ndarray
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


def get_z1_1q(c_sys: CompositeSystem) -> np.ndarray:
    """returns vec of state ``Z_1`` with the basis of ``c_sys``.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    np.ndarray
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
    """returns vec of Bell state, :math:`\\frac{1}{2} (|00\\rangle + |11\\rangle)(\\langle00| + \\langle11|)`, with the basis of ``c_sys``.

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

    # \frac{1}{2}(|00>+|11>)(<00|+<11|)
    # convert "vec in comp basis" to "vec in basis of CompositeSystem"
    from_vec = (
        np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], dtype=np.float64) / 2
    )
    to_vec = convert_vec(from_vec, c_sys.comp_basis(), c_sys.basis())
    state = State(c_sys, to_vec.real.astype(np.float64))
    return state
