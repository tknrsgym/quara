import itertools
from functools import reduce
from operator import add
from typing import List, Tuple

import numpy as np

import quara.utils.matrix_util as mutil
from quara.objects.composite_system import CompositeSystem, ElementalSystem
from quara.objects.matrix_basis import (
    MatrixBasis,
    get_comp_basis,
    get_normalized_pauli_basis,
)
from quara.settings import Settings


class Gate:
    def __init__(
        self, c_sys: CompositeSystem, hs: np.ndarray, is_physical: bool = True
    ):
        """Constructor

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem of this gate.
        hs : np.ndarray
            HS representation of this gate.
        is_physical : bool, optional
            checks whether the state is physically wrong, by default True.
            if at least one of the following conditions is ``False``, the state is physically wrong:

            - gate is TP(trace-preserving map).
            - gate is CP(Complete-Positivity-Preserving).

        Raises
        ------
        ValueError
            HS representation is not square matrix.
        ValueError
            dim of HS representation is not square number.
        ValueError
            HS representation is not real matrix.
        ValueError
            dim of HS representation does not equal dim of CompositeSystem.
        ValueError
            ``is_physical`` is ``True`` and gate is not TP.
        ValueError
            ``is_physical`` is ``True`` and gate is not CP.
        """
        self._composite_system: CompositeSystem = c_sys
        self._hs: np.ndarray = hs
        self._is_physical = is_physical

        # whether HS representation is square matrix
        size = self._hs.shape
        if size[0] != size[1]:
            raise ValueError(f"HS must be square matrix. size of HS is {size}")

        # whether dim of HS representation is square number
        self._dim: int = int(np.sqrt(size[0]))
        if self._dim ** 2 != size[0]:
            raise ValueError(f"dim of HS must be square number. dim of HS is {size[0]}")

        # whether HS representation is real matrix
        if self._hs.dtype != np.float64:
            raise ValueError(f"HS must be real matrix. dtype of HS is {self._hs.dtype}")

        # whether dim of HS equals dim of CompositeSystem
        if self._dim != self._composite_system.dim:
            raise ValueError(
                f"dim of HS must equal dim of CompositeSystem.  dim of HS is {self._dim}. dim of CompositeSystem is {self._composite_system.dim}"
            )

        # whether the state is physically wrong
        if self._is_physical:
            if not self.is_tp():
                raise ValueError("the state is physically wrong. gate is not TP.")
            elif not self.is_cp():
                raise ValueError("the state is physically wrong. gate is not CP.")

    @property
    def dim(self):
        """returns dim of gate.

        Returns
        -------
        int
            dim of gate.
        """
        return self._dim

    @property
    def hs(self):
        """returns HS representation of gate.

        Returns
        -------
        np.array
            HS representation of gate.
        """
        return self._hs

    @property
    def is_physical(self):
        """returns argument ``is_physical`` specified in the constructor.

        Returns
        -------
        int
            argument ``is_physical`` specified in the constructor.
        """
        return self._is_physical

    def get_basis(self) -> MatrixBasis:
        """returns MatrixBasis of gate.

        Returns
        -------
        MatrixBasis
            MatrixBasis of gate.
        """
        return self._composite_system.basis()

    def is_tp(self, atol: float = None) -> bool:
        """returns whether the gate is TP(trace-preserving map).

        Parameters
        ----------
        atol : float, optional
            the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.
            this function checks ``absolute(trace after mapped - trace before mapped) <= atol``.

        Returns
        -------
        bool
            True where the gate is TP, False otherwise.
        """
        atol = atol if atol else Settings.get_atol()

        # if A:HS representation of gate, then A:TP <=> Tr[A(B_\alpha)] = Tr[B_\alpha] for all basis.
        for index, basis in enumerate(self._composite_system.basis()):
            # calculate Tr[B_\alpha]
            trace_before_mapped = np.trace(basis)

            # calculate Tr[A(B_\alpha)]
            vec = np.zeros((self._dim ** 2))
            vec[index] = 1
            vec_after_mapped = self.hs @ vec

            density = np.zeros((self._dim, self._dim), dtype=np.complex128)
            for coefficient, basis in zip(
                vec_after_mapped, self._composite_system.basis()
            ):
                density += coefficient * basis

            trace_after_mapped = np.trace(density)

            # check Tr[A(B_\alpha)] = Tr[B_\alpha]
            tp_for_basis = np.isclose(
                trace_after_mapped, trace_before_mapped, atol=atol, rtol=0.0
            )
            if not tp_for_basis:
                return False

        return True

    def is_cp(self, atol: float = None) -> bool:
        """returns whether gate is CP(Complete-Positivity-Preserving).

        Parameters
        ----------
        atol : float, optional
            the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.
            this function ignores eigenvalues close zero.

        Returns
        -------
        bool
            True where gate is CP, False otherwise.
        """
        atol = atol if atol else Settings.get_atol()

        # "A is CP"  <=> "C(A) >= 0"
        return mutil.is_positive_semidefinite(self.calc_choi_matrix(), atol=atol)

    def convert_basis(self, other_basis: MatrixBasis) -> np.array:
        """returns HS representation for ``other_basis``.

        Parameters
        ----------
        other_basis : MatrixBasis
            basis.

        Returns
        -------
        np.array
            HS representation for ``other_basis``.
        """
        converted_hs = convert_hs(self.hs, self._composite_system.basis(), other_basis)
        return converted_hs

    def convert_to_comp_basis(self) -> np.array:
        """returns HS representation for computational basis.

        Returns
        -------
        np.array
            HS representation for computational basis.
        """
        converted_hs = convert_hs(
            self.hs, self._composite_system.basis(), self._composite_system.comp_basis()
        )
        return converted_hs

    def calc_choi_matrix(self) -> np.array:
        """calculates Choi matrix of gate.

        Returns
        -------
        np.array
            Choi matrix of gate.
        """
        # C(A) = \sum_{\alpha, \beta} HS(A)_{\alpha, \beta} B_\alpha \otimes \overline{B_\beta}
        tmp_list = []
        basis = self._composite_system.basis()
        indexed_basis = list(zip(range(len(basis)), basis))
        for B_alpha, B_beta in itertools.product(indexed_basis, indexed_basis):
            tmp = self._hs[B_alpha[0]][B_beta[0]] * np.kron(
                B_alpha[1], B_beta[1].conj()
            )
            tmp_list.append(tmp)

        # summing
        choi = reduce(add, tmp_list)
        return choi

    def calc_kraus_matrices(self) -> List[np.array]:
        """calculates Kraus matrices of gate.

        this function returns Kraus matrices as list of ``np.array`` with ``dtype=np.complex128``.
        the list is sorted large eigenvalue order.
        if HS of gate is not CP, then returns empty list because Kraus matrices does not exist.

        Returns
        -------
        List[np.array]
            Kraus matrices of gate.
        """
        if not self.is_cp():
            return []

        # step1. calc the eigenvalue decomposition of Choi matrix.
        #   Choi = \sum_{\alpha} c_{\alpha} |c_{\alpha}><c_{\alpha}| s.t. c_{\alpha} are eigenvalues and |c_{\alpha}> are eigenvectors of orthogonal basis.
        choi = self.calc_choi_matrix()
        eigen_vals, eigen_vecs = np.linalg.eig(choi)
        eigens = [
            (eigen_vals[index], eigen_vecs[:, index])
            for index in range(len(eigen_vals))
        ]
        # filter positive eigen values
        eigens = [
            (eigen_val, eigen_vec)
            for (eigen_val, eigen_vec) in eigens
            if eigen_val > 0 and not np.isclose(eigen_val, 0, atol=Settings.get_atol())
        ]
        # sort large eigenvalue order
        eigens = sorted(eigens, key=lambda x: x[0], reverse=True)

        # step2. calc Kraus representaion.
        #   K_{\alpha} = \sqrt{c_{\alpha}} unvec(|c_{\alpha}>)
        kraus = [
            np.sqrt(eigen_val) * eigen_vec.reshape((2, 2))
            for (eigen_val, eigen_vec) in eigens
        ]

        return kraus

    def calc_process_matrix(self) -> np.array:
        """calculates process matrix of gate.

        Returns
        -------
        np.array
            process matrix of gate.
        """
        # \chi_{\alpha, \beta}(A) = Tr[(B_{\alpha}^{\dagger} \otimes B_{\beta}^T) HS(A)] for computational basis.
        hs_comp = self.convert_to_comp_basis()
        comp_basis = self._composite_system.comp_basis()
        process_matrix = [
            np.trace(np.kron(B_alpha.conj().T, B_beta.T) @ hs_comp)
            for B_alpha, B_beta in itertools.product(comp_basis, comp_basis)
        ]
        return np.array(process_matrix).reshape((4, 4))

    def to_var(self, on_eq_constraint: bool = True) -> np.array:
        return convert_gate_to_var(
            c_sys=self._composite_system,
            hs=self.hs,
            on_eq_constraint=on_eq_constraint,
        )

def convert_var_index_to_gate_index(
    c_sys: CompositeSystem, var_index: int, on_eq_constraint: bool = True
) -> Tuple[int, int]:
    """converts variable index to gate index.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    var_index : int
        variable index.
    on_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    Tuple[int, int]
        gate index.
        first value of tuple is row number of HS representation of this gate.
        second value of tuple is column number of HS representation of this gate.
    """
    dim = c_sys.dim
    (row, col) = divmod(var_index, dim ** 2)
    if on_eq_constraint:
        row += 1
    return (row, col)


def convert_gate_index_to_var_index(
    c_sys: CompositeSystem, gate_index: Tuple[int, int], on_eq_constraint: bool = True
) -> int:
    """converts gate index to variable index.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    gate_index : Tuple[int, int]
        gate index.
        first value of tuple is row number of HS representation of this gate.
        second value of tuple is column number of HS representation of this gate.
    on_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    int
        variable index.
    """
    dim = c_sys.dim
    (row, col) = gate_index
    var_index = (
        (dim ** 2) * (row - 1) + col if on_eq_constraint else (dim ** 2) * row + col
    )
    return var_index


def convert_var_to_gate(
    c_sys: CompositeSystem, var: np.ndarray, on_eq_constraint: bool = True
) -> Gate:
    """converts vec of variables to gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    var : np.ndarray
        vec of variables.
    on_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    Gate
        converted gate.
    """
    dim = c_sys.dim
    hs = np.insert(var, 0, np.eye(1, dim ** 2), axis=0) if on_eq_constraint else var
    gate = Gate(c_sys, hs, is_physical=False)
    return gate


def convert_gate_to_var(
    c_sys: CompositeSystem, hs: np.ndarray, on_eq_constraint: bool = True
) -> np.array:
    """converts hs of gate to vec of variables.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    hs : np.ndarray
        HS representation of this gate.
    on_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    np.array
        vec of variables.
    """
    var = np.delete(hs, 0, axis=0).flatten() if on_eq_constraint else hs.flatten()
    return var


def calc_gradient_from_gate(
    c_sys: CompositeSystem,
    hs: np.ndarray,
    var_index: int,
    on_eq_constraint: bool = True,
) -> Gate:
    """calculates gradient from gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    hs : np.ndarray
        HS representation of this gate.
    var_index : int
        variable index.
    on_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    Gate
        Gate with gradient as hs.
    """
    gradient = np.zeros((c_sys.dim ** 2, c_sys.dim ** 2), dtype=np.float64)
    gate_index = convert_var_index_to_gate_index(c_sys, var_index, on_eq_constraint)
    gradient[gate_index] = 1

    gate = Gate(c_sys, gradient, is_physical=False)
    return gate


def is_hp(hs: np.array, basis: MatrixBasis, atol: float = None) -> bool:
    """returns whether gate is HP(Hermiticity-Preserving).

    HP <=> HS on Hermitian basis is real matrix.
    therefore converts input basis to Pauli basis, and checks whetever converted HS is real matrix.

    Parameters
    ----------
    hs : np.array
        HS representation of gate.
    basis : MatrixBasis
        basis of HS representation.
    atol : float, optional
        the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.
        this function checks ``absolute(imaginary part of matrix - zero matrix) <= atol``.

    Returns
    -------
    bool
        True where gate is EP, False otherwise.
    """

    atol = atol if atol else Settings.get_atol()

    # convert Hermitian basis(Pauli basis)
    hs_converted = convert_hs(hs, basis, get_normalized_pauli_basis())

    # whetever converted HS is real matrix(imaginary part is zero matrix)
    zero_matrix = np.zeros(hs_converted.shape)
    return np.allclose(hs_converted.imag, zero_matrix, atol=atol, rtol=0.0)


def calc_agf(g: Gate, u: Gate) -> np.float64:
    """returns AGF(Average Gate Fidelity) and ``g`` and ``u``.

    Parameters
    ----------
    g : Gate
        L-TP-CP map.
    u : Gate
        unitary gate.

    Returns
    -------
    np.float64
        AGF.

    Raises
    ------
    ValueError
        HS representation of ``u`` is not Hermitian.
    """
    # u: unitary gate <=> HS(u) is Hermitian
    # whetever HS(u) is Hermitian
    if not mutil.is_hermitian(u.hs):
        raise ValueError("gate u must be unitary")

    # let trace = Tr[HS(u)^{\dagger}HS(g)]
    # AGF = 1-\frac{d^2-trace}{d(d+1)}
    d = u.dim
    trace = np.vdot(u.hs, g.hs)
    agf = 1 - (d ** 2 - trace) / (d * (d + 1))
    return agf


def convert_hs(
    from_hs: np.array, from_basis: MatrixBasis, to_basis: MatrixBasis
) -> np.array:
    """returns HS representation for ``to_basis``

    Parameters
    ----------
    from_hs : np.array
        HS representation before convert.
    from_basis : MatrixBasis
        basis before convert.
    to_basis : MatrixBasis
        basis after convert.

    Returns
    -------
    np.array
        HS representation for ``to_basis``.

    Raises
    ------
    ValueError
        ``from_hs`` is not square matrix.
    ValueError
        dim of ``from_hs`` is not square number.
    ValueError
        dim of ``from_basis`` does not equal dim of ``to_basis``.
    ValueError
        length of ``from_basis`` does not equal length of ``to_basis``.
    """
    ### parameter check

    # whether HS is square matrix
    size = from_hs.shape
    if size[0] != size[1]:
        raise ValueError(f"HS must be square matrix. size of HS is {size}")

    # whether dim of HS is square number
    dim: int = int(np.sqrt(size[0]))
    if dim ** 2 != size[0]:
        raise ValueError(f"dim of HS must be square number. dim of HS is {size[0]}")

    # whether dim of from_basis equals dim of to_basis
    if from_basis.dim != to_basis.dim:
        raise ValueError(
            f"dim of from_basis must equal dim of to_basis.  dim of from_basis is {from_basis.dim}. dim of to_basis is {to_basis.dim}"
        )

    # whether length of from_basis equals length of to_basis
    if len(from_basis) != len(to_basis):
        raise ValueError(
            f"length of from_basis must equal length of to_basis.  length of from_basis is {len(from_basis)}. length of to_basis is {len(to_basis)}"
        )

    ### main logic

    # U_{\alpha,\bata} := Tr[to_basis_{\alpha}^{\dagger} @ from_basis_{\beta}]
    trans_matrix = [
        np.vdot(B_alpha, B_beta)
        for B_alpha, B_beta in itertools.product(to_basis, from_basis)
    ]
    U = np.array(trans_matrix).reshape(from_basis.dim ** 2, from_basis.dim ** 2)
    to_hs = U @ from_hs @ U.conj().T
    return to_hs


def _get_1q_gate_from_hs_on_pauli_basis(
    matrix: np.array, c_sys: CompositeSystem
) -> Gate:
    # whether dim of CompositeSystem equals 2
    if c_sys.dim != 2:
        raise ValueError(
            f"dim of CompositeSystem must equals 2.  dim of CompositeSystem is {c_sys.dim}"
        )

    # convert "HS representation in Pauli basis" to "HS representation in basis of CompositeSystem"
    hs = convert_hs(matrix, get_normalized_pauli_basis(), c_sys.basis())
    gate = Gate(c_sys, hs.real.astype(np.float64))
    return gate


def get_i(c_sys: CompositeSystem) -> Gate:
    """returns identity gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing gate.

    Returns
    -------
    Gate
        identity gate.
    """
    hs = np.eye(c_sys.dim ** 2, dtype=np.float64)
    gate = Gate(c_sys, hs)
    return gate


def get_x(c_sys: CompositeSystem) -> Gate:
    """returns Pauli X gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing gate.

    Returns
    -------
    Gate
        Pauli X gate.

    Raises
    ------
    ValueError
        CompositeSystem is not 1quit.
    ValueError
        dim of CompositeSystem does not equal 2
    """
    # whether CompositeSystem is 1 qubit
    size = len(c_sys._elemental_systems)
    if size != 1:
        raise ValueError(f"CompositeSystem must be 1 qubit. it is {size} qubits")

    matrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    gate = _get_1q_gate_from_hs_on_pauli_basis(matrix, c_sys)
    return gate


def get_y(c_sys: CompositeSystem) -> Gate:
    """returns Pauli Y gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing gate.

    Returns
    -------
    Gate
        Pauli Y gate.

    Raises
    ------
    ValueError
        CompositeSystem is not 1quit.
    ValueError
        dim of CompositeSystem does not equal 2
    """
    # whether CompositeSystem is 1 qubit
    size = len(c_sys._elemental_systems)
    if size != 1:
        raise ValueError(f"CompositeSystem must be 1 qubit. it is {size} qubits")

    matrix = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    gate = _get_1q_gate_from_hs_on_pauli_basis(matrix, c_sys)
    return gate


def get_z(c_sys: CompositeSystem) -> Gate:
    """returns Pauli Z gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing gate.

    Returns
    -------
    Gate
        Pauli Z gate.

    Raises
    ------
    ValueError
        CompositeSystem is not 1quit.
    ValueError
        dim of CompositeSystem does not equal 2
    """
    # whether CompositeSystem is 1 qubit
    size = len(c_sys._elemental_systems)
    if size != 1:
        raise ValueError(f"CompositeSystem must be 1 qubit. it is {size} qubits")

    matrix = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    gate = _get_1q_gate_from_hs_on_pauli_basis(matrix, c_sys)
    return gate


def get_h(c_sys: CompositeSystem) -> Gate:
    """returns H gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing gate.

    Returns
    -------
    Gate
        H gate.

    Raises
    ------
    ValueError
        CompositeSystem is not 1quit.
    ValueError
        dim of CompositeSystem does not equal 2
    """
    # whether CompositeSystem is 1 qubit
    size = len(c_sys._elemental_systems)
    if size != 1:
        raise ValueError(f"CompositeSystem must be 1 qubit. it is {size} qubits")

    matrix = np.array(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0]], dtype=np.float64
    )
    gate = _get_1q_gate_from_hs_on_pauli_basis(matrix, c_sys)
    return gate


def get_root_x(c_sys: CompositeSystem) -> Gate:
    """returns root of X gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing gate.

    Returns
    -------
    Gate
        root of X gate.

    Raises
    ------
    ValueError
        CompositeSystem is not 1quit.
    ValueError
        dim of CompositeSystem does not equal 2
    """
    # whether CompositeSystem is 1 qubit
    size = len(c_sys._elemental_systems)
    if size != 1:
        raise ValueError(f"CompositeSystem must be 1 qubit. it is {size} qubits")

    matrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]], dtype=np.float64
    )
    gate = _get_1q_gate_from_hs_on_pauli_basis(matrix, c_sys)
    return gate


def get_root_y(c_sys: CompositeSystem) -> Gate:
    """returns root of Y gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing gate.

    Returns
    -------
    Gate
        root of Y gate.

    Raises
    ------
    ValueError
        CompositeSystem is not 1quit.
    ValueError
        dim of CompositeSystem does not equal 2
    """
    # whether CompositeSystem is 1 qubit
    size = len(c_sys._elemental_systems)
    if size != 1:
        raise ValueError(f"CompositeSystem must be 1 qubit. it is {size} qubits")

    matrix = np.array(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0]], dtype=np.float64
    )
    gate = _get_1q_gate_from_hs_on_pauli_basis(matrix, c_sys)
    return gate


def get_s(c_sys: CompositeSystem) -> Gate:
    """returns S gate(root of Z).

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing gate.

    Returns
    -------
    Gate
        S gate(root of Z).

    Raises
    ------
    ValueError
        CompositeSystem is not 1quit.
    ValueError
        dim of CompositeSystem does not equal 2
    """
    # whether CompositeSystem is 1 qubit
    size = len(c_sys._elemental_systems)
    if size != 1:
        raise ValueError(f"CompositeSystem must be 1 qubit. it is {size} qubits")

    matrix = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    gate = _get_1q_gate_from_hs_on_pauli_basis(matrix, c_sys)
    return gate


def get_sdg(c_sys: CompositeSystem) -> Gate:
    """returns dagger of S gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing gate.

    Returns
    -------
    Gate
        dagger of S gate.

    Raises
    ------
    ValueError
        CompositeSystem is not 1quit.
    ValueError
        dim of CompositeSystem does not equal 2
    """
    # whether CompositeSystem is 1 qubit
    size = len(c_sys._elemental_systems)
    if size != 1:
        raise ValueError(f"CompositeSystem must be 1 qubit. it is {size} qubits")

    matrix = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    gate = _get_1q_gate_from_hs_on_pauli_basis(matrix, c_sys)
    return gate


def get_t(c_sys: CompositeSystem) -> Gate:
    """returns T gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing gate.

    Returns
    -------
    Gate
        T gate.

    Raises
    ------
    ValueError
        CompositeSystem is not 1quit.
    ValueError
        dim of CompositeSystem does not equal 2
    """
    # whether CompositeSystem is 1 qubit
    size = len(c_sys._elemental_systems)
    if size != 1:
        raise ValueError(f"CompositeSystem must be 1 qubit. it is {size} qubits")

    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    gate = _get_1q_gate_from_hs_on_pauli_basis(matrix, c_sys)
    return gate


def get_cnot(c_sys: CompositeSystem, control: ElementalSystem) -> Gate:
    """returns CNOT gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing gate.
    control : ElementalSystem
        ElementalSystem of control qubit.

    Returns
    -------
    Gate
        CNOT gate.

    Raises
    ------
    ValueError
        CompositeSystem is not 2quits.
    ValueError
        dim of CompositeSystem does not equal 4.
    """
    # whether CompositeSystem is 2 qubits
    size = len(c_sys._elemental_systems)
    if size != 2:
        raise ValueError(f"CompositeSystem must be 2 qubits. it is {size} qubits")

    # whether dim of CompositeSystem equals 4
    if c_sys.dim != 4:
        raise ValueError(
            f"dim of CompositeSystem must equals 4.  dim of CompositeSystem is {c_sys.dim}"
        )

    if control.name == c_sys.elemental_systems[0].name:
        # control bit is 1st qubit
        hs_comp_basis = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float64,
        )
    else:
        # control bit is 2nd qubit
        hs_comp_basis = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )

    hs_for_c_sys = convert_hs(
        hs_comp_basis, c_sys.comp_basis(), c_sys.basis()
    ).real.astype(np.float64)
    gate = Gate(c_sys, hs_for_c_sys)
    return gate


def get_cz(c_sys: CompositeSystem) -> Gate:
    """returns CZ gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing gate.

    Returns
    -------
    Gate
        CZ gate.

    Raises
    ------
    ValueError
        CompositeSystem is not 2quits.
    ValueError
        dim of CompositeSystem does not equal 4.
    """
    # whether CompositeSystem is 2 qubits
    size = len(c_sys._elemental_systems)
    if size != 2:
        raise ValueError(f"CompositeSystem must be 2 qubits. it is {size} qubits")

    # whether dim of CompositeSystem equals 4
    if c_sys.dim != 4:
        raise ValueError(
            f"dim of CompositeSystem must equals 4.  dim of CompositeSystem is {c_sys.dim}"
        )

    hs_comp_basis = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )

    hs_for_c_sys = convert_hs(
        hs_comp_basis, c_sys.comp_basis(), c_sys.basis()
    ).real.astype(np.float64)
    gate = Gate(c_sys, hs_for_c_sys)
    return gate


def get_swap(c_sys: CompositeSystem) -> Gate:
    """returns SWAP gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing gate.

    Returns
    -------
    Gate
        SWAP gate.

    Raises
    ------
    ValueError
        CompositeSystem is not 2quits.
    ValueError
        dim of CompositeSystem does not equal 4
    """
    # whether CompositeSystem is 2 qubits
    size = len(c_sys._elemental_systems)
    if size != 2:
        raise ValueError(f"CompositeSystem must be 2 qubits. it is {size} qubits")

    # whether dim of CompositeSystem equals 4
    if c_sys.dim != 4:
        raise ValueError(
            f"dim of CompositeSystem must equals 4.  dim of CompositeSystem is {c_sys.dim}"
        )

    hs_comp_basis = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )

    hs_for_c_sys = convert_hs(
        hs_comp_basis, c_sys.comp_basis(), c_sys.basis()
    ).real.astype(np.float64)
    gate = Gate(c_sys, hs_for_c_sys)
    return gate
