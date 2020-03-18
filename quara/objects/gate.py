from functools import reduce
from operator import add
import itertools
from typing import List

import numpy as np

from quara.objects.composite_system import CompositeSystem
from quara.objects.matrix_basis import (
    MatrixBasis,
    get_comp_basis,
    get_normalized_pauli_basis,
)
import quara.utils.matrix_util as mutil


class Gate:
    def __init__(self, c_sys: CompositeSystem, hs: np.ndarray):
        self._composite_system: CompositeSystem = c_sys
        self._hs: np.ndarray = hs

        # whether HS is square matrix
        size = self._hs.shape
        if size[0] != size[1]:
            raise ValueError(f"HS must be square matrix. size of HS is {size}")

        # whether dim of HS is square number
        self._dim: int = int(np.sqrt(size[0]))
        if self._dim ** 2 != size[0]:
            raise ValueError(f"dim of HS must be square number. dim of HS is {size[0]}")

        # whether dtype=np.float64
        if self._hs.dtype != np.float64:
            raise ValueError(
                f"entries of HS must be real numbers. dtype of HS is {self._hs.dtype}"
            )

        # whether dim of HS equals dim of compsite system
        if self._dim != self._composite_system.dim():
            raise ValueError(
                f"dim of HS must equal dim of CompositeSystem.  dim of HS is {self._dim}. dim of CompositeSystem is {self._composite_system.dim()}"
            )

    @property
    def dim(self):
        """returns dim representation of gate.
        
        Returns
        -------
        int
            dim representation of gate
        """
        return self._dim

    @property
    def hs(self):
        """returns HS representation of gate.
        
        Returns
        -------
        np.array
            HS representation of gate
        """
        return self._hs

    def get_basis(self) -> MatrixBasis:
        """returns MatrixBasis of gate
        
        Returns
        -------
        MatrixBasis
            MatrixBasis of gate
        """
        return self._composite_system.basis()

    def is_tp(self, atol: float = 1e-13) -> bool:
        """returns whether the gate is TP(trace-preserving map).
        
        Parameters
        ----------
        atol : float, optional
            the absolute tolerance parameter, by default 1e-13.
            this function checks ``absolute(trace after mapped - trace before mapped) <= atol``.
        
        Returns
        -------
        bool
            True where the gate is TP, False otherwise.
        """
        # if A:HS representation of gate, then A:TP <=> Tr[A(B_\alpha)] = Tr[B_\alpha] for all basis.
        dim = self._composite_system.basis().dim
        for basis in self._composite_system.basis():
            trace_before_mapped = np.trace(basis)
            vec_basis = basis.reshape((-1, 1))
            trace_after_mapped = np.trace((self.hs @ vec_basis).reshape((dim, dim)))
            tp_for_basis = np.isclose(
                trace_after_mapped, trace_before_mapped, atol=atol, rtol=0.0
            )
            if not tp_for_basis:
                return False

        return True

    def is_cp(self) -> bool:
        """returns whether gate is CP(Complete-Positivity-Preserving).
        
        Returns
        -------
        bool
            True where gate is CP, False otherwise.
        """
        # "A is CP"  <=> "C(A) >= 0"
        return np.all(np.linalg.eigvals(self.calc_choi_matrix()) >= 0)

    def convert_basis(self, other_basis: MatrixBasis) -> np.array:
        """returns HS representation for ``other_basis``.
        
        Parameters
        ----------
        other_basis : MatrixBasis
            basis
        
        Returns
        -------
        np.array
            HS representation for ``other_basis``
        """
        converted_hs = convert_hs(self.hs, self._composite_system.basis(), other_basis)
        return converted_hs

    def convert_to_comp_basis(self) -> np.array:
        """returns HS representation for computational basis.
        
        Returns
        -------
        np.array
            HS representation for computational basis
        """
        converted_hs = convert_hs(
            self.hs, self._composite_system.basis(), get_comp_basis()
        )
        return converted_hs

    def calc_choi_matrix(self) -> np.array:
        """calculates Choi matrix of gate.
        
        Returns
        -------
        np.array
            Choi matrix of gate
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
            Kraus matrices of gate
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
            if eigen_val > 0 and not np.isclose(eigen_val, 0, atol=1e-14)
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
            process matrix of gate
        """
        # \chi_{\alpha, \beta}(A) = Tr[(B_{\alpha}^{\dagger} \otimes B_{\beta}^T) HS(A)] for computational basis.
        hs_comp = self.convert_to_comp_basis()
        comp_basis = get_comp_basis()
        process_matrix = [
            np.trace(np.kron(B_alpha.conj().T, B_beta.T) @ hs_comp)
            for B_alpha, B_beta in itertools.product(comp_basis, comp_basis)
        ]
        return np.array(process_matrix).reshape((4, 4))


def is_ep(hs: np.array, basis: MatrixBasis, atol: float = 1e-13) -> bool:
    """returns whether gate is EP(Hermiticity-Preserving).

    EP <=> HS on Hermitian basis is real matrix.
    therefore converts input basis to Pauli basis, and checks whetever converted HS is real matrix.

    Parameters
    ----------
    hs : np.array
        HS representation of gate
    basis : MatrixBasis
        [description]
    atol : float, optional
        the absolute tolerance parameter, by default 1e-13.
        this function checks ``absolute(imaginary part of matrix - zero matrix) <= atol``.
    
    Returns
    -------
    bool
        True where gate is EP, False otherwise.
    """
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
        L-TP-CP map
    u : Gate
        unitary gate
    
    Returns
    -------
    np.float64
        AGF
    
    Raises
    ------
    ValueError
        HS representation of ``u`` is not Hermitian
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
        HS representation before convert
    from_basis : MatrixBasis
        basis before convert
    to_basis : MatrixBasis
        basis after convert
    
    Returns
    -------
    np.array
        HS representation for ``to_basis``
    
    Raises
    ------
    ValueError
        ``from_hs`` is not square matrix
    ValueError
        dim of ``from_hs`` is not square number
    ValueError
        dim of ``from_basis`` does not equal dim of ``to_basis``
    ValueError
        length of ``from_basis`` does not equal length of ``to_basis``
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
    if c_sys.dim() != 2:
        raise ValueError(
            f"dim of CompositeSystem must equals 2.  dim of CompositeSystem is {c_sys.dim()}"
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
        CompositeSystem containing gate
    
    Returns
    -------
    Gate
        identity gate
    """
    matrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    gate = _get_1q_gate_from_hs_on_pauli_basis(matrix, c_sys)
    return gate


def get_x(c_sys: CompositeSystem) -> Gate:
    """returns Pauli X gate.
    
    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing gate
    
    Returns
    -------
    Gate
        Pauli X gate
    """
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
        CompositeSystem containing gate
    
    Returns
    -------
    Gate
        Pauli Y gate
    """
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
        CompositeSystem containing gate
    
    Returns
    -------
    Gate
        Pauli Z gate
    """
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
        CompositeSystem containing gate
    
    Returns
    -------
    Gate
        H gate
    """
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
        CompositeSystem containing gate
    
    Returns
    -------
    Gate
        root of X gate
    """
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
        CompositeSystem containing gate
    
    Returns
    -------
    Gate
        root of Y gate
    """
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
        CompositeSystem containing gate
    
    Returns
    -------
    Gate
        S gate(root of Z)
    """
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
        CompositeSystem containing gate
    
    Returns
    -------
    Gate
        dagger of S gate
    """
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
        CompositeSystem containing gate
    
    Returns
    -------
    Gate
        T gate
    """
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


def get_cnot(c_sys: CompositeSystem) -> Gate:
    # TODO implement
    comp_basis_1q = get_comp_basis()
    new_basis = [
        np.kron(val1, val2)
        for val1, val2 in itertools.product(comp_basis_1q, comp_basis_1q)
    ]
    comp_basis_2q = MatrixBasis(new_basis)

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
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    hs_for_c_sys = convert_hs(hs_comp_basis, comp_basis_2q, c_sys.basis()).real.astype(
        np.float64
    )
    gate = Gate(c_sys, hs_for_c_sys)
    return gate
