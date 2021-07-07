import copy
import itertools
from functools import reduce
from operator import add
import sys
from typing import List, Tuple, Optional

import numpy as np
from scipy.linalg import expm

import quara.utils.matrix_util as mutil
from quara.objects.composite_system import CompositeSystem, ElementalSystem
from quara.objects.gate import (
    Gate,
    convert_hs,
    convert_var_index_to_gate_index,
    convert_gate_index_to_var_index,
    convert_hs_to_var,
)
from quara.objects.matrix_basis import (
    MatrixBasis,
    get_comp_basis,
    get_normalized_pauli_basis,
)
from quara.settings import Settings
from quara.objects.qoperation import QOperation


class EffectiveLindbladian(Gate):
    def __init__(
        self,
        c_sys: CompositeSystem,
        hs: np.ndarray,
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
            CompositeSystem of this EffectiveLindbladian.
        hs : np.ndarray
            HS representation of this EffectiveLindbladian.
        is_physicality_required : bool, optional
            checks whether the EffectiveLindbladian is physically wrong, by default True.
            if at least one of the following conditions is ``False``, the EffectiveLindbladian is physically wrong:

            - EffectiveLindbladian is TP(trace-preserving map).
            - EffectiveLindbladian is CP(Complete-Positivity-Preserving).

            If you want to ignore the above requirements and create a EffectiveLindbladian object, set ``is_physicality_required`` to ``False``.

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
            ``is_physicality_required`` is ``True`` and the gate is not physically correct.
        """
        # check the basis is a orthonormal Hermitian matrix basis with B_0 = I/sqrt(d)
        if c_sys.is_orthonormal_hermitian_0thprop_identity == False:
            raise ValueError(
                "basis is not a orthonormal Hermitian matrix basis and 0th prop I."
            )
        dim = c_sys.dim
        expected_B0 = np.eye(dim) / np.sqrt(dim)
        if not np.allclose(
            c_sys.basis()[0], expected_B0, atol=Settings.get_atol(), rtol=0.0
        ):
            raise ValueError(
                "0th basis is not I/sqrt(dim). basis[0]={c_sys.basis()[0]}"
            )

        super().__init__(
            c_sys,
            hs,
            is_physicality_required=is_physicality_required,
            is_estimation_object=is_estimation_object,
            on_para_eq_constraint=on_para_eq_constraint,
            on_algo_eq_constraint=on_algo_eq_constraint,
            on_algo_ineq_constraint=on_algo_ineq_constraint,
            mode_proj_order=mode_proj_order,
            eps_proj_physical=eps_proj_physical,
        )

        # whether the EffectiveLindbladian is physically correct
        if self.is_physicality_required and not self.is_physical():
            raise ValueError("the EffectiveLindbladian is not phsically correct.")

    def calc_h_mat(self) -> np.ndarray:
        """calculates h matrix of this EffectiveLindbladian.

        Returns
        -------
        np.ndarray
            h matrix of this EffectiveLindbladian.
        """
        basis = self.composite_system.basis()
        comp_basis = self.composite_system.comp_basis()
        lindbladian_cb = convert_hs(self.hs, basis, comp_basis)
        identity = np.eye(self.dim)

        tmp_h_mat = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for B_alpha in basis:
            trace = np.trace(
                lindbladian_cb
                @ (np.kron(B_alpha, identity) - np.kron(identity, B_alpha.conj()))
            )
            h_alpha = 1j / (2 * self.dim) * trace
            tmp_h_mat += h_alpha * B_alpha

        return tmp_h_mat

    def calc_j_mat(self) -> np.ndarray:
        """calculates j matrix of this EffectiveLindbladian.

        Returns
        -------
        np.ndarray
            j matrix of this EffectiveLindbladian.
        """
        basis = self.composite_system.basis()
        comp_basis = self.composite_system.comp_basis()
        lindbladian_cb = convert_hs(self.hs, basis, comp_basis)
        identity = np.eye(self.dim)

        tmp_j_mat = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for alpha, B_alpha in enumerate(basis[1:]):
            trace = np.trace(
                lindbladian_cb
                @ (np.kron(B_alpha, identity) + np.kron(identity, B_alpha.conj()))
            )
            delta = 1 if alpha == 0 else 0
            j_alpha = 1 / (2 * self.dim * (1 + delta)) * trace
            tmp_j_mat += j_alpha * B_alpha

        return tmp_j_mat

    def calc_k_mat(self) -> np.ndarray:
        """calculates k matrix of this EffectiveLindbladian.

        Returns
        -------
        np.ndarray
            k matrix of this EffectiveLindbladian.
        """
        basis = self.composite_system.basis()
        comp_basis = self.composite_system.comp_basis()
        lindbladian_cb = convert_hs(self.hs, basis, comp_basis)

        tmp_k_mat = np.zeros(
            (self.dim ** 2 - 1, self.dim ** 2 - 1), dtype=np.complex128
        )
        for alpha, B_alpha in enumerate(basis[1:]):
            for beta, B_beta in enumerate(basis[1:]):
                tmp_k_mat[alpha, beta] = np.trace(
                    lindbladian_cb @ np.kron(B_alpha, B_beta.conj())
                )

        return tmp_k_mat

    def _check_mode_basis(self, mode_basis: str):
        if not mode_basis in ["hermitian_basis", "comp_basis"]:
            raise ValueError(f"unsupported mode_basis={mode_basis}")

    def calc_h_part(self, mode_basis: str = "hermitian_basis") -> np.ndarray:
        """calculates h part of this EffectiveLindbladian.

        mode_basis allows the following values:
        - hermitian_basis
        - comp_basis

        Parameters
        ----------
        mode_basis : str, optional
            basis for calculating h part, by default "hermitian_basis"

        Returns
        -------
        np.ndarray
            h part of this EffectiveLindbladian.
        """
        self._check_mode_basis(mode_basis)
        h_mat = self.calc_h_mat()
        h_part = _calc_h_part_from_h_mat(h_mat)

        if mode_basis == "hermitian_basis":
            h_part = convert_hs(
                h_part,
                self.composite_system.comp_basis(),
                self.composite_system.basis(),
            )
            h_part = _truncate_hs(h_part, self.eps_proj_physical)

        return h_part

    def calc_j_part(self, mode_basis: str = "hermitian_basis") -> np.ndarray:
        """calculates j part of this EffectiveLindbladian.

        mode_basis allows the following values:
        - hermitian_basis
        - comp_basis

        Parameters
        ----------
        mode_basis : str, optional
            basis for calculating j part, by default "hermitian_basis"

        Returns
        -------
        np.ndarray
            j part of this EffectiveLindbladian.
        """
        self._check_mode_basis(mode_basis)
        j_mat = self.calc_j_mat()
        j_part = _calc_j_part_from_j_mat(j_mat)

        if mode_basis == "hermitian_basis":
            j_part = convert_hs(
                j_part,
                self.composite_system.comp_basis(),
                self.composite_system.basis(),
            )
            j_part = _truncate_hs(j_part, self.eps_proj_physical)

        return j_part

    def calc_k_part(self, mode_basis: str = "hermitian_basis") -> np.ndarray:
        """calculates k part of this EffectiveLindbladian.

        mode_basis allows the following values:
        - hermitian_basis
        - comp_basis

        Parameters
        ----------
        mode_basis : str, optional
            basis for calculating k part, by default "hermitian_basis"

        Returns
        -------
        np.ndarray
            k part of this EffectiveLindbladian.
        """
        self._check_mode_basis(mode_basis)
        k_mat = self.calc_k_mat()
        k_part = _calc_k_part_from_k_mat(k_mat, self.composite_system)

        if mode_basis == "hermitian_basis":
            k_part = convert_hs(
                k_part,
                self.composite_system.comp_basis(),
                self.composite_system.basis(),
            )
            k_part = _truncate_hs(k_part, self.eps_proj_physical)

        return k_part

    def calc_d_part(self, mode_basis: str = "hermitian_basis") -> np.ndarray:
        """calculates d part of this EffectiveLindbladian.

        mode_basis allows the following values:
        - hermitian_basis
        - comp_basis

        Parameters
        ----------
        mode_basis : str, optional
            basis for calculating d part, by default "hermitian_basis"

        Returns
        -------
        np.ndarray
            d part of this EffectiveLindbladian.
        """
        self._check_mode_basis(mode_basis)
        d_part = self.calc_j_part(mode_basis="comp_basis") + self.calc_k_part(
            mode_basis="comp_basis"
        )

        if mode_basis == "hermitian_basis":
            d_part = convert_hs(
                d_part,
                self.composite_system.comp_basis(),
                self.composite_system.basis(),
            )
            d_part = _truncate_hs(d_part, self.eps_proj_physical)

        return d_part

    def _generate_origin_obj(self):
        # return HS matrix of the origin = diag(0, min, min,..,min) in R^{{dim ** 2}x{dim ** 2}}
        min = sys.float_info.min_exp
        diag_values = [0] + [min] * (self.dim ** 2 - 1)
        origin_hs = np.diag(diag_values).real.astype(np.float64)
        return origin_hs

    def calc_gradient(self, var_index: int) -> "EffectiveLindbladian":
        lindbladian = calc_gradient_from_effective_lindbladian(
            self.composite_system,
            self.hs,
            var_index,
            is_estimation_object=self.is_estimation_object,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            eps_proj_physical=self.eps_proj_physical,
        )
        return lindbladian

    def calc_proj_eq_constraint(self) -> "EffectiveLindbladian":
        new_hs = self._copy()
        new_hs[0, :] = 0
        new_lindbladian = EffectiveLindbladian(
            c_sys=self.composite_system,
            hs=new_hs,
            is_physicality_required=self.is_physicality_required,
            is_estimation_object=self.is_estimation_object,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            eps_proj_physical=self.eps_proj_physical,
        )

        return new_lindbladian

    def calc_proj_ineq_constraint(self) -> "EffectiveLindbladian":
        h_mat = self.calc_h_mat()
        j_mat = self.calc_j_mat()
        k_mat = self.calc_k_mat()

        # project k_mat
        eigenvals, eigenvecs = np.linalg.eig(k_mat)
        for index in range(len(eigenvals)):
            if eigenvals[index] < 0:
                eigenvals[index] = 0
        new_k_mat = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T.conjugate()

        new_lindbladian = generate_effective_lindbladian_from_hjk(
            self.composite_system,
            h_mat,
            j_mat,
            new_k_mat,
            is_physicality_required=self.is_physicality_required,
            is_estimation_object=self.is_estimation_object,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            eps_proj_physical=self.eps_proj_physical,
        )

        return new_lindbladian

    def is_tp(self, atol: float = None) -> bool:
        """returns whether the effective Lindbladian is TP(trace-preserving map).

        Parameters
        ----------
        atol : float, optional
            the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.
            this function checks ``absolute(trace after mapped - trace before mapped) <= atol``.

        Returns
        -------
        bool
            True where the effective Lindbladian is TP, False otherwise.
        """
        atol = Settings.get_atol() if atol is None else atol

        # for A:L^{gb}, "A is TP" <=> "1st row of A is zeros"
        return np.allclose(self.hs[0], 0, atol=atol, rtol=0.0)

    def is_cp(self, atol: float = None) -> bool:
        """returns whether effective Lindbladian is CP(Complete-Positivity-Preserving).

        Parameters
        ----------
        atol : float, optional
            the absolute tolerance parameter, uses :func:`~quara.settings.Settings.get_atol` by default.
            this function ignores eigenvalues close zero.

        Returns
        -------
        bool
            True where the effective Lindbladian is CP, False otherwise.
        """
        atol = Settings.get_atol() if atol is None else atol

        # for A:L^{gb}, "A is CP"  <=> "k >= 0"
        return mutil.is_positive_semidefinite(self.calc_k_mat(), atol=atol)

    def to_kraus_matrices(self) -> List[Tuple[np.float64, np.ndarray]]:
        """returns Kraus matrices of EffectiveLindbladian.

        if :math:`A` is Hermitian preserve matrix, then :math:`A(X) = \\sum_i a_i A_i X A_i^{\\dagger}`, where :math:`a_i` are real numbers and :math:`A_i` are complex square matrices.
        this function returns the list of :math:`(a_i, A_i)` sorted in descending order by :math:`a_i`.

        Returns
        -------
        List[Tuple[np.float64, np.ndarray]]
            Kraus matrices of EffectiveLindbladian.
        """
        # step1. calc the eigenvalue decomposition of Choi matrix.
        #   Choi = \sum_{\alpha} c_{\alpha} |c_{\alpha}><c_{\alpha}| s.t. c_{\alpha} are eigenvalues and |c_{\alpha}> are eigenvectors of orthogonal basis.
        choi = self.to_choi_matrix()
        eigen_vals, eigen_vecs = np.linalg.eig(choi)
        eigens = [
            (eigen_vals[index], eigen_vecs[:, index])
            for index in range(len(eigen_vals))
        ]
        # filter non-zero eigen values
        eigens = [
            (eigen_val, eigen_vec)
            for (eigen_val, eigen_vec) in eigens
            if not np.isclose(eigen_val, 0, atol=Settings.get_atol())
        ]
        # sort large eigenvalue order
        eigens = sorted(eigens, key=lambda x: x[0], reverse=True)

        # step2. convert to Kraus representaion.
        #   K_{\alpha} = {\sqrt{c_{\alpha}}, unvec(|c_{\alpha}>)}
        kraus = [
            (np.sqrt(eigen_val), eigen_vec.reshape((self.dim, self.dim)))
            for (eigen_val, eigen_vec) in eigens
        ]

        return kraus

    def _generate_from_var_func(self):
        return convert_var_to_effective_lindbladian

    def to_gate(self) -> Gate:
        """returns the Gate corresponding to this EffectiveLindbladian.

        Returns
        -------
        Gate
            the Gate corresponding to this EffectiveLindbladian.
        """
        new_hs = expm(self.hs)
        gate = Gate(
            self.composite_system,
            new_hs,
            is_physicality_required=self.is_physicality_required,
            is_estimation_object=self.is_estimation_object,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
        )
        return gate


def convert_var_index_to_effective_lindbladian_index(
    c_sys: CompositeSystem, var_index: int, on_para_eq_constraint: bool = True
) -> Tuple[int, int]:
    """converts variable index to EffectiveLindbladian index.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this EffectiveLindbladian.
    var_index : int
        variable index.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    Tuple[int, int]
        index of EffectiveLindbladian.
        first value of tuple is row number of HS representation of this EffectiveLindbladian.
        second value of tuple is column number of HS representation of this EffectiveLindbladian.
    """
    return convert_var_index_to_gate_index(
        c_sys, var_index, on_para_eq_constraint=on_para_eq_constraint
    )


def convert_effective_lindbladian_index_to_var_index(
    c_sys: CompositeSystem,
    effective_lindbladian_index: Tuple[int, int],
    on_para_eq_constraint: bool = True,
) -> int:
    """converts effective_lindbladian_index index to variable index.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this EffectiveLindbladian.
    effective_lindbladian_index : Tuple[int, int]
        index of EffectiveLindbladian.
        first value of tuple is row number of HS representation of this EffectiveLindbladian.
        second value of tuple is column number of HS representation of this EffectiveLindbladian.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    int
        variable index.
    """
    return convert_gate_index_to_var_index(
        c_sys, effective_lindbladian_index, on_para_eq_constraint=on_para_eq_constraint
    )


def convert_var_to_effective_lindbladian(
    c_sys: CompositeSystem,
    var: np.ndarray,
    is_physicality_required: bool = True,
    is_estimation_object: bool = True,
    on_para_eq_constraint: bool = True,
    on_algo_eq_constraint: bool = True,
    on_algo_ineq_constraint: bool = True,
    eps_proj_physical: float = None,
) -> EffectiveLindbladian:
    """converts vec of variables to EffectiveLindbladian.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this EffectiveLindbladian.
    var : np.ndarray
        vec of variables.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    EffectiveLindbladian
        converted EffectiveLindbladian.
    """
    dim = c_sys.dim

    size = (dim ** 2 - 1, dim ** 2) if on_para_eq_constraint else (dim ** 2, dim ** 2)
    reshaped = var.reshape(size)

    hs = (
        np.insert(reshaped, 0, np.eye(1, dim ** 2), axis=0)
        if on_para_eq_constraint
        else reshaped
    )
    lindbladian = EffectiveLindbladian(
        c_sys,
        hs,
        is_physicality_required=is_physicality_required,
        is_estimation_object=is_estimation_object,
        on_para_eq_constraint=on_para_eq_constraint,
        on_algo_eq_constraint=on_algo_eq_constraint,
        on_algo_ineq_constraint=on_algo_ineq_constraint,
        eps_proj_physical=eps_proj_physical,
    )
    return lindbladian


def convert_effective_lindbladian_to_var(
    c_sys: CompositeSystem, hs: np.ndarray, on_para_eq_constraint: bool = True
) -> np.ndarray:
    """converts hs of EffectiveLindbladian to vec of variables.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this EffectiveLindbladian.
    hs : np.ndarray
        HS representation of this EffectiveLindbladian.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    np.ndarray
        vec of variables.
    """
    return convert_hs_to_var(c_sys, hs, on_para_eq_constraint=on_para_eq_constraint)


def calc_gradient_from_effective_lindbladian(
    c_sys: CompositeSystem,
    hs: np.ndarray,
    var_index: int,
    is_estimation_object: bool = True,
    on_para_eq_constraint: bool = True,
    on_algo_eq_constraint: bool = True,
    on_algo_ineq_constraint: bool = True,
    eps_proj_physical: float = None,
) -> EffectiveLindbladian:
    """calculates gradient from EffectiveLindbladian.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    hs : np.ndarray
        HS representation of this gate.
    var_index : int
        variable index.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    EffectiveLindbladian
        EffectiveLindbladian with gradient as hs.
    """
    gradient = np.zeros((c_sys.dim ** 2, c_sys.dim ** 2), dtype=np.float64)
    gate_index = convert_var_index_to_effective_lindbladian_index(
        c_sys, var_index, on_para_eq_constraint
    )
    gradient[gate_index] = 1

    lindbladian = EffectiveLindbladian(
        c_sys,
        gradient,
        is_physicality_required=False,
        is_estimation_object=is_estimation_object,
        on_para_eq_constraint=on_para_eq_constraint,
        on_algo_eq_constraint=on_algo_eq_constraint,
        on_algo_ineq_constraint=on_algo_ineq_constraint,
        eps_proj_physical=eps_proj_physical,
    )
    return lindbladian


def _check_h_mat(h_mat: np.ndarray, dim: int) -> None:
    # whetever h_mat is Hermitian
    if not mutil.is_hermitian(h_mat):
        raise ValueError("h_mat must be Hermitian. h_mat={h_mat}")

    # whether dim of h_mat equals dim of CompositeSystem
    size = h_mat.shape[0]
    if dim != size:
        raise ValueError(
            f"dim of h_mat must equal dim of CompositeSystem.  dim of h_mat is {size}. dim of CompositeSystem is {dim}"
        )


def _calc_h_part_from_h_mat(h_mat: np.ndarray) -> np.ndarray:
    identity = np.eye(h_mat.shape[0])
    return -1j * (np.kron(h_mat, identity) - np.kron(identity, h_mat.conj()))


def _check_j_mat(j_mat: np.ndarray, dim: int) -> None:
    # whetever j_mat is Hermitian
    if not mutil.is_hermitian(j_mat):
        raise ValueError("j_mat must be Hermitian. j_mat={j_mat}")

    # whether dim of j_mat equals dim of CompositeSystem
    size = j_mat.shape[0]
    if dim != size:
        raise ValueError(
            f"dim of j_mat must equal dim of CompositeSystem.  dim of j_mat is {size}. dim of CompositeSystem is {dim}"
        )


def _calc_j_mat_from_k_mat(k_mat: np.ndarray, c_sys: CompositeSystem) -> None:
    basis = c_sys.basis()
    j_mat = np.zeros((c_sys.dim, c_sys.dim), dtype=np.complex128)
    for row in range(k_mat.shape[0]):
        for col in range(k_mat.shape[1]):
            term = k_mat[row, col] * (basis[col + 1].T.conj() @ basis[row + 1])
            j_mat += term

    return -1 / 2 * j_mat


def _calc_j_part_from_j_mat(j_mat: np.ndarray) -> np.ndarray:
    identity = np.eye(j_mat.shape[0])
    return np.kron(j_mat, identity) + np.kron(identity, j_mat.conj())


def _check_k_mat(k_mat: np.ndarray, dim: int) -> None:
    # whetever k_mat is Hermitian
    if not mutil.is_hermitian(k_mat):
        raise ValueError("k_mat must be Hermitian. k_mat={k_mat}")

    # whether dim of k_mat equals dim of CompositeSystem
    size = k_mat.shape[0]
    if dim ** 2 - 1 != size:
        raise ValueError(
            f"dim of k_mat must equal 'dim of CompositeSystem' ** 2 -1 .  dim of k_mat is {size}. dim of CompositeSystem is {dim}"
        )


def _calc_k_part_from_k_mat(k_mat: np.ndarray, c_sys: CompositeSystem) -> np.ndarray:
    basis = c_sys.basis()
    k_part = np.zeros((c_sys.dim ** 2, c_sys.dim ** 2), dtype=np.complex128)
    for row in range(k_mat.shape[0]):
        for col in range(k_mat.shape[0]):
            term = k_mat[row, col] * np.kron(basis[row + 1], basis[col + 1].conj())
            k_part += term

    return k_part


def _truncate_hs(
    hs: np.ndarray,
    eps_proj_physical: float = None,
    is_zero_imaginary_part_required: bool = True,
) -> np.ndarray:
    tmp_hs = mutil.truncate_imaginary_part(hs, eps_proj_physical)

    if is_zero_imaginary_part_required == True and np.any(tmp_hs.imag != 0):
        raise ValueError(
            f"some imaginary parts of entries of matrix != 0. converted hs={tmp_hs}"
        )

    if is_zero_imaginary_part_required == True:
        tmp_hs = tmp_hs.real.astype(np.float64)

    truncated_hs = mutil.truncate_computational_fluctuation(tmp_hs, eps_proj_physical)
    return truncated_hs


def generate_hs_from_hjk(
    c_sys: CompositeSystem,
    h_mat: np.ndarray,
    j_mat: np.ndarray,
    k_mat: np.ndarray,
    eps_proj_physical: float = None,
) -> np.ndarray:
    """generates HS matrix of EffectiveLindbladian from h matrix, j matrix and k matrix.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this EffectiveLindbladian.
    h_mat : np.ndarray
        h matrix.
    j_mat : np.ndarray
        j matrix.
    k_mat : np.ndarray
        k matrix.

    Returns
    -------
    np.ndarray
        HS matrix of EffectiveLindbladian.
    """
    dim = c_sys.dim

    # calculate h_part
    _check_h_mat(h_mat, dim)
    h_part = _calc_h_part_from_h_mat(h_mat)

    # calculate j_part
    _check_j_mat(j_mat, dim)
    j_part = _calc_j_part_from_j_mat(j_mat)

    # calculate k_part
    _check_k_mat(k_mat, dim)
    k_part = _calc_k_part_from_k_mat(k_mat, c_sys)

    # calculate hs(=Lindbladian for Hermitian basis)
    lindbladian_comp_basis = h_part + j_part + k_part
    lindbladian_tmp = convert_hs(
        lindbladian_comp_basis, c_sys.comp_basis(), c_sys.basis()
    )
    lindbladian_hermitian_basis = _truncate_hs(lindbladian_tmp, eps_proj_physical)

    return lindbladian_hermitian_basis


def generate_effective_lindbladian_from_hjk(
    c_sys: CompositeSystem,
    h_mat: np.ndarray,
    j_mat: np.ndarray,
    k_mat: np.ndarray,
    is_physicality_required: bool = True,
    is_estimation_object: bool = True,
    on_para_eq_constraint: bool = True,
    on_algo_eq_constraint: bool = True,
    on_algo_ineq_constraint: bool = True,
    mode_proj_order: str = "eq_ineq",
    eps_proj_physical: float = None,
):
    """generates EffectiveLindbladian from h matrix, j matrix and k matrix.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this EffectiveLindbladian.
    h_mat : np.ndarray
        h matrix.
    j_mat : np.ndarray
        j matrix.
    k_mat : np.ndarray
        k matrix.
    is_physicality_required : bool, optional
        whether this QOperation is physicality required, by default True
    is_estimation_object : bool, optional
        whether this QOperation is estimation object, by default True
    on_para_eq_constraint : bool, optional
        whether this QOperation is on parameter equality constraint, by default True
    on_algo_eq_constraint : bool, optional
        whether this QOperation is on algorithm equality constraint, by default True
    on_algo_ineq_constraint : bool, optional
        whether this QOperation is on algorithm inequality constraint, by default True
    mode_proj_order : str, optional
        the order in which the projections are performed, by default "eq_ineq"
    eps_proj_physical : float, optional
        epsilon that is projection algorithm error threshold for being physical, by default :func:`~quara.settings.Settings.get_atol` / 10.0

    Returns
    -------
    np.ndarray
        EffectiveLindbladian.
    """
    # generate HS
    hs = generate_hs_from_hjk(c_sys, h_mat, j_mat, k_mat)

    # init
    effective_lindbladian = EffectiveLindbladian(
        c_sys,
        hs,
        is_physicality_required=is_physicality_required,
        is_estimation_object=is_estimation_object,
        on_para_eq_constraint=on_para_eq_constraint,
        on_algo_eq_constraint=on_algo_eq_constraint,
        on_algo_ineq_constraint=on_algo_ineq_constraint,
        mode_proj_order=mode_proj_order,
        eps_proj_physical=eps_proj_physical,
    )
    return effective_lindbladian


def generate_hs_from_h(
    c_sys: CompositeSystem,
    h_mat: np.ndarray,
    eps_proj_physical: float = None,
) -> np.ndarray:
    """generates HS matrix of EffectiveLindbladian from h matrix.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this EffectiveLindbladian.
    h_mat : np.ndarray
        h matrix.

    Returns
    -------
    np.ndarray
        HS matrix of EffectiveLindbladian.
    """
    dim = c_sys.dim

    # calculate h_part
    _check_h_mat(h_mat, dim)
    h_part = _calc_h_part_from_h_mat(h_mat)

    # calculate hs(=Lindbladian for Hermitian basis)
    lindbladian_comp_basis = h_part
    lindbladian_tmp = convert_hs(
        lindbladian_comp_basis, c_sys.comp_basis(), c_sys.basis()
    )
    lindbladian_hermitian_basis = _truncate_hs(lindbladian_tmp, eps_proj_physical)

    return lindbladian_hermitian_basis


def generate_effective_lindbladian_from_h(
    c_sys: CompositeSystem,
    h_mat: np.ndarray,
    is_physicality_required: bool = True,
    is_estimation_object: bool = True,
    on_para_eq_constraint: bool = True,
    on_algo_eq_constraint: bool = True,
    on_algo_ineq_constraint: bool = True,
    mode_proj_order: str = "eq_ineq",
    eps_proj_physical: float = None,
):
    """generates EffectiveLindbladian from h matrix.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this EffectiveLindbladian.
    h_mat : np.ndarray
        h matrix.
    is_physicality_required : bool, optional
        whether this QOperation is physicality required, by default True
    is_estimation_object : bool, optional
        whether this QOperation is estimation object, by default True
    on_para_eq_constraint : bool, optional
        whether this QOperation is on parameter equality constraint, by default True
    on_algo_eq_constraint : bool, optional
        whether this QOperation is on algorithm equality constraint, by default True
    on_algo_ineq_constraint : bool, optional
        whether this QOperation is on algorithm inequality constraint, by default True
    mode_proj_order : str, optional
        the order in which the projections are performed, by default "eq_ineq"
    eps_proj_physical : float, optional
        epsilon that is projection algorithm error threshold for being physical, by default :func:`~quara.settings.Settings.get_atol` / 10.0

    Returns
    -------
    np.ndarray
        EffectiveLindbladian.
    """
    # generate HS
    hs = generate_hs_from_h(c_sys, h_mat)

    # init
    effective_lindbladian = EffectiveLindbladian(
        c_sys,
        hs,
        is_physicality_required=is_physicality_required,
        is_estimation_object=is_estimation_object,
        on_para_eq_constraint=on_para_eq_constraint,
        on_algo_eq_constraint=on_algo_eq_constraint,
        on_algo_ineq_constraint=on_algo_ineq_constraint,
        mode_proj_order=mode_proj_order,
        eps_proj_physical=eps_proj_physical,
    )
    return effective_lindbladian


def generate_hs_from_hk(
    c_sys: CompositeSystem,
    h_mat: np.ndarray,
    k_mat: np.ndarray,
    eps_proj_physical: float = None,
) -> np.ndarray:
    """generates HS matrix of EffectiveLindbladian from h matrix and k matrix.

    j matrix is calculated from k matrix.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this EffectiveLindbladian.
    h_mat : np.ndarray
        h matrix.
    k_mat : np.ndarray
        k matrix.

    Returns
    -------
    np.ndarray
        HS matrix of EffectiveLindbladian.
    """
    dim = c_sys.dim

    # calculate h_part
    _check_h_mat(h_mat, dim)
    h_part = _calc_h_part_from_h_mat(h_mat)

    # calculate k_part
    _check_k_mat(k_mat, dim)
    k_part = _calc_k_part_from_k_mat(k_mat, c_sys)

    # calculate j_part
    j_mat = _calc_j_mat_from_k_mat(k_mat, c_sys)
    j_part = _calc_j_part_from_j_mat(j_mat)

    # calculate hs(=Lindbladian for Hermitian basis)
    lindbladian_comp_basis = h_part + j_part + k_part
    lindbladian_tmp = convert_hs(
        lindbladian_comp_basis, c_sys.comp_basis(), c_sys.basis()
    )
    lindbladian_hermitian_basis = _truncate_hs(lindbladian_tmp, eps_proj_physical)

    return lindbladian_hermitian_basis


def generate_effective_lindbladian_from_hk(
    c_sys: CompositeSystem,
    h_mat: np.ndarray,
    k_mat: np.ndarray,
    is_physicality_required: bool = True,
    is_estimation_object: bool = True,
    on_para_eq_constraint: bool = True,
    on_algo_eq_constraint: bool = True,
    on_algo_ineq_constraint: bool = True,
    mode_proj_order: str = "eq_ineq",
    eps_proj_physical: float = None,
):
    """generates EffectiveLindbladian from h matrix and k matrix.

    j matrix is calculated from k matrix.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this EffectiveLindbladian.
    h_mat : np.ndarray
        h matrix.
    k_mat : np.ndarray
        k matrix.
    is_physicality_required : bool, optional
        whether this QOperation is physicality required, by default True
    is_estimation_object : bool, optional
        whether this QOperation is estimation object, by default True
    on_para_eq_constraint : bool, optional
        whether this QOperation is on parameter equality constraint, by default True
    on_algo_eq_constraint : bool, optional
        whether this QOperation is on algorithm equality constraint, by default True
    on_algo_ineq_constraint : bool, optional
        whether this QOperation is on algorithm inequality constraint, by default True
    mode_proj_order : str, optional
        the order in which the projections are performed, by default "eq_ineq"
    eps_proj_physical : float, optional
        epsilon that is projection algorithm error threshold for being physical, by default :func:`~quara.settings.Settings.get_atol` / 10.0

    Returns
    -------
    np.ndarray
        EffectiveLindbladian.
    """
    # generate HS
    hs = generate_hs_from_hk(c_sys, h_mat, k_mat)

    # init
    effective_lindbladian = EffectiveLindbladian(
        c_sys,
        hs,
        is_physicality_required=is_physicality_required,
        is_estimation_object=is_estimation_object,
        on_para_eq_constraint=on_para_eq_constraint,
        on_algo_eq_constraint=on_algo_eq_constraint,
        on_algo_ineq_constraint=on_algo_ineq_constraint,
        mode_proj_order=mode_proj_order,
        eps_proj_physical=eps_proj_physical,
    )
    return effective_lindbladian


def generate_hs_from_k(
    c_sys: CompositeSystem,
    k_mat: np.ndarray,
    eps_proj_physical: float = None,
) -> np.ndarray:
    """generates HS matrix of EffectiveLindbladian from k matrix.

    j matrix is calculated from k matrix.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this EffectiveLindbladian.
    k_mat : np.ndarray
        k matrix.

    Returns
    -------
    np.ndarray
        HS matrix of EffectiveLindbladian.
    """
    dim = c_sys.dim

    # calculate k_part
    _check_k_mat(k_mat, dim)
    k_part = _calc_k_part_from_k_mat(k_mat, c_sys)

    # calculate j_part
    j_mat = _calc_j_mat_from_k_mat(k_mat, c_sys)
    j_part = _calc_j_part_from_j_mat(j_mat)

    # calculate hs(=Lindbladian for Hermitian basis)
    lindbladian_comp_basis = j_part + k_part
    lindbladian_tmp = convert_hs(
        lindbladian_comp_basis, c_sys.comp_basis(), c_sys.basis()
    )
    lindbladian_hermitian_basis = _truncate_hs(lindbladian_tmp, eps_proj_physical)

    return lindbladian_hermitian_basis


def generate_effective_lindbladian_from_k(
    c_sys: CompositeSystem,
    k_mat: np.ndarray,
    is_physicality_required: bool = True,
    is_estimation_object: bool = True,
    on_para_eq_constraint: bool = True,
    on_algo_eq_constraint: bool = True,
    on_algo_ineq_constraint: bool = True,
    mode_proj_order: str = "eq_ineq",
    eps_proj_physical: float = None,
):
    """generates EffectiveLindbladian from k matrix.

    j matrix is calculated from k matrix.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this EffectiveLindbladian.
    k_mat : np.ndarray
        k matrix.
    is_physicality_required : bool, optional
        whether this QOperation is physicality required, by default True
    is_estimation_object : bool, optional
        whether this QOperation is estimation object, by default True
    on_para_eq_constraint : bool, optional
        whether this QOperation is on parameter equality constraint, by default True
    on_algo_eq_constraint : bool, optional
        whether this QOperation is on algorithm equality constraint, by default True
    on_algo_ineq_constraint : bool, optional
        whether this QOperation is on algorithm inequality constraint, by default True
    mode_proj_order : str, optional
        the order in which the projections are performed, by default "eq_ineq"
    eps_proj_physical : float, optional
        epsilon that is projection algorithm error threshold for being physical, by default :func:`~quara.settings.Settings.get_atol` / 10.0

    Returns
    -------
    np.ndarray
        EffectiveLindbladian.
    """
    # generate HS
    hs = generate_hs_from_k(c_sys, k_mat)

    # init
    effective_lindbladian = EffectiveLindbladian(
        c_sys,
        hs,
        is_physicality_required=is_physicality_required,
        is_estimation_object=is_estimation_object,
        on_para_eq_constraint=on_para_eq_constraint,
        on_algo_eq_constraint=on_algo_eq_constraint,
        on_algo_ineq_constraint=on_algo_ineq_constraint,
        mode_proj_order=mode_proj_order,
        eps_proj_physical=eps_proj_physical,
    )
    return effective_lindbladian


def generate_j_part_cb_from_jump_operators(
    jump_operators: List[np.ndarray],
) -> np.ndarray:
    """generates j part of EffectiveLindbladian from jump operators.

    this j part is represented by computational basis.

    Parameters
    ----------
    jump_operators : List[np.ndarray]
        jump operators to generate j part.

    Returns
    -------
    np.ndarray
        j part of EffectiveLindbladian.
    """
    dim = jump_operators[0].shape[0]
    identity = np.eye(dim)
    terms = [
        np.kron(opertor, identity) + np.kron(identity, opertor.conj())
        for opertor in jump_operators
    ]
    j_part_cb = -1 / 2 * reduce(add, terms)
    return j_part_cb


def generate_j_part_gb_from_jump_operators(
    jump_operators: List[np.ndarray],
    basis: MatrixBasis,
    eps_proj_physical: float = None,
) -> np.ndarray:
    """generates j part of EffectiveLindbladian from jump operators.

    this j part is represented by general basis.

    Parameters
    ----------
    jump_operators : List[np.ndarray]
        jump operators to generate j part.
    basis : MatrixBasis
        MatrixBasis to present j part.
    eps_proj_physical : float, optional
        error threshold to truncate, by default :func:`~quara.settings.Settings.get_atol`

    Returns
    -------
    np.ndarray
        j part of EffectiveLindbladian.
    """
    j_part_cb = generate_j_part_cb_from_jump_operators(jump_operators)
    j_part_gb = convert_hs(j_part_cb, get_comp_basis(basis.dim), basis)
    j_part_gb = _truncate_hs(j_part_gb, eps_proj_physical)
    return j_part_gb


def generate_k_part_cb_from_jump_operators(
    jump_operators: List[np.ndarray],
) -> np.ndarray:
    """generates k part of EffectiveLindbladian from jump operators.

    this k part is represented by computational basis.

    Parameters
    ----------
    jump_operators : List[np.ndarray]
        jump operators to generate k part.

    Returns
    -------
    np.ndarray
        k part of EffectiveLindbladian.
    """
    terms = [np.kron(opertor, opertor.conj()) for opertor in jump_operators]
    k_part_cb = reduce(add, terms)
    return k_part_cb


def generate_k_part_gb_from_jump_operators(
    jump_operators: List[np.ndarray],
    basis: MatrixBasis,
    eps_proj_physical: float = None,
) -> np.ndarray:
    """generates k part of EffectiveLindbladian from jump operators.

    this k part is represented by general basis.

    Parameters
    ----------
    jump_operators : List[np.ndarray]
        jump operators to generate k part.
    basis : MatrixBasis
        MatrixBasis to present k part.
    eps_proj_physical : float, optional
        error threshold to truncate, by default :func:`~quara.settings.Settings.get_atol`

    Returns
    -------
    np.ndarray
        k part of EffectiveLindbladian.
    """
    k_part_cb = generate_k_part_cb_from_jump_operators(jump_operators)
    k_part_gb = convert_hs(k_part_cb, get_comp_basis(basis.dim), basis)
    k_part_gb = _truncate_hs(k_part_gb, eps_proj_physical)
    return k_part_gb


def generate_d_part_cb_from_jump_operators(
    jump_operators: List[np.ndarray],
) -> np.ndarray:
    """generates d part of EffectiveLindbladian from jump operators.

    this d part is represented by computational basis.

    Parameters
    ----------
    jump_operators : List[np.ndarray]
        jump_operators to generate d part.

    Returns
    -------
    np.ndarray
        d part of EffectiveLindbladian.
    """
    d_part_cb = generate_j_part_cb_from_jump_operators(
        jump_operators
    ) + generate_k_part_cb_from_jump_operators(jump_operators)
    return d_part_cb


def generate_d_part_gb_from_jump_operators(
    jump_operators: List[np.ndarray],
    basis: MatrixBasis,
    eps_proj_physical: float = None,
) -> np.ndarray:
    """generates d part of EffectiveLindbladian from jump operators.

    this d part is represented by general basis.

    Parameters
    ----------
    jump_operators : List[np.ndarray]
        jump operators to generate d part.
    basis : MatrixBasis
        MatrixBasis to present d part.
    eps_proj_physical : float, optional
        threshold to truncate, by default :func:`~quara.settings.Settings.get_atol`

    Returns
    -------
    np.ndarray
        d part of EffectiveLindbladian.
    """
    d_part_cb = generate_d_part_cb_from_jump_operators(jump_operators)
    d_part_gb = convert_hs(d_part_cb, get_comp_basis(basis.dim), basis)
    d_part_gb = _truncate_hs(d_part_gb, eps_proj_physical)
    return d_part_gb


def generate_effective_lindbladian_from_jump_operators(
    c_sys: CompositeSystem,
    jump_operators: List[np.ndarray],
    is_physicality_required: bool = True,
    is_estimation_object: bool = True,
    on_para_eq_constraint: bool = True,
    on_algo_eq_constraint: bool = True,
    on_algo_ineq_constraint: bool = True,
    mode_proj_order: str = "eq_ineq",
    eps_proj_physical: float = None,
):
    """generates EffectiveLindbladian from jump operators.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this EffectiveLindbladian.
    jump_operators : List[np.ndarray]
        jump operators to generate EffectiveLindbladian.
    is_physicality_required : bool, optional
        whether this QOperation is physicality required, by default True
    is_estimation_object : bool, optional
        whether this QOperation is estimation object, by default True
    on_para_eq_constraint : bool, optional
        whether this QOperation is on parameter equality constraint, by default True
    on_algo_eq_constraint : bool, optional
        whether this QOperation is on algorithm equality constraint, by default True
    on_algo_ineq_constraint : bool, optional
        whether this QOperation is on algorithm inequality constraint, by default True
    mode_proj_order : str, optional
        the order in which the projections are performed, by default "eq_ineq"
    eps_proj_physical : float, optional
        epsilon that is projection algorithm error threshold for being physical, by default :func:`~quara.settings.Settings.get_atol` / 10.0

    Returns
    -------
    np.ndarray
        EffectiveLindbladian.
    """
    # calculate hs(=Lindbladian for Hermitian basis)
    lindbladian_tmp = generate_d_part_gb_from_jump_operators(
        jump_operators, c_sys.basis()
    )
    lindbladian_hermitian_basis = _truncate_hs(lindbladian_tmp, eps_proj_physical)

    # init
    effective_lindbladian = EffectiveLindbladian(
        c_sys,
        lindbladian_hermitian_basis,
        is_physicality_required=is_physicality_required,
        is_estimation_object=is_estimation_object,
        on_para_eq_constraint=on_para_eq_constraint,
        on_algo_eq_constraint=on_algo_eq_constraint,
        on_algo_ineq_constraint=on_algo_ineq_constraint,
        mode_proj_order=mode_proj_order,
        eps_proj_physical=eps_proj_physical,
    )
    return effective_lindbladian
