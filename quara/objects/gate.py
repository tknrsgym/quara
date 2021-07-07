import copy
import itertools
from functools import reduce
from operator import add
from typing import List, Tuple, Optional

import numpy as np

import quara.utils.matrix_util as mutil
from quara.objects.composite_system import CompositeSystem, ElementalSystem
from quara.objects.matrix_basis import (
    MatrixBasis,
    get_comp_basis,
    get_normalized_pauli_basis,
)
from quara.settings import Settings
from quara.objects.qoperation import QOperation


class Gate(QOperation):
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
            CompositeSystem of this gate.
        hs : np.ndarray
            HS representation of this gate.
        is_physicality_required : bool, optional
            checks whether the gate is physically wrong, by default True.
            if at least one of the following conditions is ``False``, the gate is physically wrong:

            - gate is TP(trace-preserving map).
            - gate is CP(Complete-Positivity-Preserving).

            If you want to ignore the above requirements and create a Gate object, set ``is_physicality_required`` to ``False``.

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
        self._hs: np.ndarray = hs

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
        if self._dim != self.composite_system.dim:
            raise ValueError(
                f"dim of HS must equal dim of CompositeSystem.  dim of HS is {self._dim}. dim of CompositeSystem is {self.composite_system.dim}"
            )

        # whether the gate is physically correct
        if self.is_physicality_required and not self.is_physical():
            raise ValueError("the gate is not physically correct.")

    def _info(self):
        info = {}
        info["Type"] = self.__class__.__name__
        info["Dim"] = self.dim
        info["HS"] = self.hs
        return info

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
        np.ndarray
            HS representation of gate.
        """
        return self._hs

    def is_eq_constraint_satisfied(self, atol: float = None):
        return self.is_tp(atol)

    def is_ineq_constraint_satisfied(self, atol: float = None):
        return self.is_cp(atol)

    def set_zero(self):
        self._hs = np.zeros(self._hs.shape, dtype=np.float64)
        self._is_physicality_required = False

    def _generate_zero_obj(self):
        new_hs = np.zeros(self.hs.shape, dtype=np.float64)
        return new_hs

    def _generate_origin_obj(self):
        size = self.hs.shape
        new_hs = np.zeros(size)
        new_hs[0][0] = 1
        return new_hs

    def to_var(self) -> np.ndarray:
        return convert_hs_to_var(
            c_sys=self.composite_system,
            hs=self.hs,
            on_para_eq_constraint=self.on_para_eq_constraint,
        )

    def to_stacked_vector(self) -> np.ndarray:
        return self.hs.flatten()

    def calc_gradient(self, var_index: int) -> "Gate":
        gate = calc_gradient_from_gate(
            self.composite_system,
            self.hs,
            var_index,
            is_estimation_object=self.is_estimation_object,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
        )
        return gate

    def calc_proj_eq_constraint(self) -> "Gate":
        hs = copy.deepcopy(self.hs)
        hs[0][0] = 1
        hs[0][1:] = 0
        new_gate = Gate(
            c_sys=self.composite_system,
            hs=hs,
            is_physicality_required=self.is_physicality_required,
            is_estimation_object=self.is_estimation_object,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
        )

        return new_gate

    @staticmethod
    def calc_proj_eq_constraint_with_var(
        c_sys: CompositeSystem,
        var: np.ndarray,
        on_para_eq_constraint: bool = True,
    ) -> np.ndarray:
        """calculates the projection of Gate on equal constraint.

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
            the projection of Gate on equal constraint.
        """
        if on_para_eq_constraint:
            new_var = var
        else:
            new_var = copy.deepcopy(var)
            new_var[0] = 1
            new_var[1 : c_sys.dim ** 2] = 0

        return new_var

    def calc_proj_ineq_constraint(self) -> "Gate":
        # calc engenvalues and engenvectors
        choi_matrix = self.to_choi_matrix_with_sparsity()
        eigenvals, eigenvecs = np.linalg.eigh(choi_matrix)

        # project
        diag = np.diag(eigenvals)
        diag[diag < 0] = 0

        # calc new HS
        new_choi_matrix = eigenvecs @ diag @ eigenvecs.T.conjugate()
        new_hs = to_hs_from_choi_with_sparsity(self.composite_system, new_choi_matrix)

        # create new Gate
        new_gate = Gate(
            c_sys=self.composite_system,
            hs=new_hs,
            is_physicality_required=self.is_physicality_required,
            is_estimation_object=self.is_estimation_object,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
        )

        return new_gate

    @staticmethod
    def calc_proj_ineq_constraint_with_var(
        c_sys: CompositeSystem,
        var: np.ndarray,
        on_para_eq_constraint: bool = True,
    ) -> np.ndarray:
        """calculates the projection of Gate on inequal constraint.

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
            the projection of Gate on equal constraint.
        """
        # calc engenvalues and engenvectors
        choi_matrix = to_choi_from_var(c_sys, var, on_para_eq_constraint)
        eigenvals, eigenvecs = np.linalg.eigh(choi_matrix)

        # project
        diag = np.diag(eigenvals)
        diag[diag < 0] = 0

        # calc new HS
        new_choi_matrix = eigenvecs @ diag @ eigenvecs.T.conjugate()
        new_hs = to_hs_from_choi_with_sparsity(c_sys, new_choi_matrix)

        # HS to var
        new_var = convert_hs_to_var(c_sys, new_hs, on_para_eq_constraint)
        return new_var

    @staticmethod
    def convert_var_to_stacked_vector(
        c_sys: CompositeSystem,
        var: np.ndarray,
        on_para_eq_constraint: bool = True,
    ) -> np.ndarray:
        """converts variables of gate to stacked vector of gate.

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem of this gate.
        var : np.ndarray
            variables of gate.
        on_para_eq_constraint : bool, optional
            uses equal constraints, by default True.

        Returns
        -------
        np.ndarray
            stacked vector of gate.
        """
        return convert_var_to_vec(c_sys, var, on_para_eq_constraint)

    @staticmethod
    def convert_stacked_vector_to_var(
        c_sys: CompositeSystem, vec: np.ndarray, on_para_eq_constraint: bool = True
    ) -> np.ndarray:
        """converts stacked vector of gate to variables of gate.

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem of this gate.
        vec : np.ndarray
            stacked_vector of gate.
        on_para_eq_constraint : bool, optional
            uses equal constraints, by default True.

        Returns
        -------
        np.ndarray
            variables of gate.
        """
        return convert_vec_to_var(c_sys, vec, on_para_eq_constraint)

    def _add_vec(self, other) -> np.ndarray:
        new_hs = self.hs + other.hs
        return new_hs

    def _sub_vec(self, other) -> np.ndarray:
        new_hs = self.hs - other.hs
        return new_hs

    def _mul_vec(self, other):
        new_hs = self.hs * other
        return new_hs

    def _truediv_vec(self, other):
        new_hs = self.hs / other
        return new_hs

    def get_basis(self) -> MatrixBasis:
        """returns MatrixBasis of gate.

        Returns
        -------
        MatrixBasis
            MatrixBasis of gate.
        """
        return self.composite_system.basis()

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
        atol = Settings.get_atol() if atol is None else atol

        # if A:HS representation of gate, then A:TP <=> Tr[A(B_\alpha)] = Tr[B_\alpha] for all basis.
        for index, basis in enumerate(self.composite_system.basis()):
            # calculate Tr[B_\alpha]
            trace_before_mapped = np.trace(basis)

            # calculate Tr[A(B_\alpha)]
            vec = np.zeros((self._dim ** 2))
            vec[index] = 1
            vec_after_mapped = self.hs @ vec

            density = np.zeros((self._dim, self._dim), dtype=np.complex128)
            for coefficient, basis in zip(
                vec_after_mapped, self.composite_system.basis()
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
        atol = Settings.get_atol() if atol is None else atol

        # "A is CP"  <=> "C(A) >= 0"
        return mutil.is_positive_semidefinite(
            self.to_choi_matrix_with_sparsity(), atol=atol
        )

    def convert_basis(self, other_basis: MatrixBasis) -> np.ndarray:
        """returns HS representation for ``other_basis``.

        Parameters
        ----------
        other_basis : MatrixBasis
            basis.

        Returns
        -------
        np.ndarray
            HS representation for ``other_basis``.
        """
        converted_hs = convert_hs(self.hs, self.composite_system.basis(), other_basis)
        return converted_hs

    def convert_to_comp_basis(self, mode: str = "row_major") -> np.ndarray:
        """returns HS representation for computational basis.

        Parameters
        ----------
        mode : str, optional
            specify whether the order of basis is "row_major" or "column_major", by default "row_major".

        Returns
        -------
        np.ndarray
            HS representation for computational basis.
        """
        converted_hs = convert_hs(
            self.hs,
            self.composite_system.basis(),
            self.composite_system.comp_basis(mode=mode),
        )
        return converted_hs

    def to_choi_matrix(self) -> np.ndarray:
        """returns Choi matrix of gate.

        Returns
        -------
        np.ndarray
            Choi matrix of gate.
        """
        # C(A) = \sum_{\alpha, \beta} HS(A)_{\alpha, \beta} B_\alpha \otimes \overline{B_\beta}
        c_sys = self.composite_system
        num_basis = len(c_sys.basis())
        tmp_list = []
        for alpha, beta in itertools.product(range(num_basis), range(num_basis)):
            tmp = self._hs[alpha][beta] * c_sys.basis_basisconjugate((alpha, beta))
            tmp_list.append(tmp)

        # summing
        choi = reduce(add, tmp_list)
        return choi

    def to_choi_matrix_with_dict(self) -> np.ndarray:
        """returns Choi matrix of gate.

        this function uses the scipy.sparse module.

        Returns
        -------
        np.ndarray
            Choi matrix of gate.
        """
        c_sys = self.composite_system
        num_basis = len(c_sys.basis())
        choi = np.zeros((num_basis, num_basis), dtype=np.complex128)
        for i, j in itertools.product(range(num_basis), range(num_basis)):
            non_zeros = c_sys._dict_from_hs_to_choi.get((i, j), [])
            for alpha, beta, coefficient in non_zeros:
                choi[i, j] += self.hs[alpha, beta] * coefficient

        return choi

    def to_choi_matrix_with_sparsity(self) -> np.ndarray:
        """returns Choi matrix of gate.

        this function uses the scipy.sparse module.

        Returns
        -------
        np.ndarray
            Choi matrix of gate.
        """
        return to_choi_from_hs(self.composite_system, self._hs)

    def to_kraus_matrices(self) -> List[np.ndarray]:
        """returns Kraus matrices of gate.

        this function returns Kraus matrices as list of ``np.ndarray`` with ``dtype=np.complex128``.
        the list is sorted large eigenvalue order.
        if HS of gate is not CP, then returns empty list because Kraus matrices does not exist.

        Returns
        -------
        List[np.ndarray]
            Kraus matrices of gate.
        """
        if not self.is_cp():
            return []

        # step1. calc the eigenvalue decomposition of Choi matrix.
        #   Choi = \sum_{\alpha} c_{\alpha} |c_{\alpha}><c_{\alpha}| s.t. c_{\alpha} are eigenvalues and |c_{\alpha}> are eigenvectors of orthogonal basis.
        choi = self.to_choi_matrix_with_sparsity()
        eigen_vals, eigen_vecs = np.linalg.eigh(choi)
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

        # step2. calc Kraus representaion.
        #   K_{\alpha} = \sqrt{c_{\alpha}} unvec(|c_{\alpha}>)
        _kraus = [
            np.sqrt(eigen_val) * eigen_vec.reshape((self.dim, self.dim))
            for (eigen_val, eigen_vec) in eigens
        ]

        # step3: fix phase
        kraus = []
        for k in _kraus:
            # k_00 = k[0][0]

            # ang = np.angle(k_00)
            # _k = (np.e ** (-1j * ang)) * k
            # kraus.append(_k)
            for i, value in enumerate(k.flatten()):
                if value == 0:
                    continue
                elif value < 0:
                    print(f"debug: k[{i}] value < 0")
                    e_i_theta = value / abs(value)
                    _k = (1 / e_i_theta) * k

                    # _k = (np.e ** (-1j * ang)) * k
                    kraus.append(_k)
                    break
                else:
                    kraus.append(k)
                    break
            else:
                kraus.append(k)
        return kraus

    def to_process_matrix(self) -> np.ndarray:
        """returns process matrix of gate.

        Returns
        -------
        np.ndarray
            process matrix of gate.
        """
        # \chi_{\alpha, \beta}(A) = Tr[(B_{\alpha}^{\dagger} \otimes B_{\beta}^T) HS(A)] for computational basis.
        hs_comp = self.convert_to_comp_basis()
        comp_basis = self.composite_system.comp_basis()
        process_matrix = [
            np.trace(np.kron(B_alpha.conj().T, B_beta.T) @ hs_comp)
            for B_alpha, B_beta in itertools.product(comp_basis, comp_basis)
        ]
        return np.array(process_matrix).reshape((4, 4))

    def _generate_from_var_func(self):
        return convert_var_to_gate

    def _copy(self):
        return copy.deepcopy(self.hs)

    @staticmethod
    def convert_var_to_stacked_vector(
        c_sys: CompositeSystem,
        var: np.ndarray,
        on_para_eq_constraint: bool = True,
    ) -> np.ndarray:
        """converts variables of gate to stacked vector of gate.

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem of this gate.
        var : np.ndarray
            variables of gate.
        on_para_eq_constraint : bool, optional
            uses equal constraints, by default True.

        Returns
        -------
        np.ndarray
            stacked vector of gate.
        """
        if on_para_eq_constraint:
            head = np.zeros(c_sys.dim ** 2)
            head[0] = 1
            stacked_vector = np.insert(var, 0, head)
        else:
            stacked_vector = var

        return stacked_vector

    @staticmethod
    def convert_stacked_vector_to_var(
        c_sys: CompositeSystem,
        stacked_vector: np.ndarray,
        on_para_eq_constraint: bool = True,
    ) -> np.ndarray:
        """converts stacked vector of gate to variables of gate.

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem of this gate.
        stacked_vector : np.ndarray
            stacked vector of gate.
        on_para_eq_constraint : bool, optional
            uses equal constraints, by default True.

        Returns
        -------
        np.ndarray
            variables of gate.
        """
        return (
            np.delete(stacked_vector, np.s_[:c_sys.dim ** 2])
            if on_para_eq_constraint
            else stacked_vector
        )


def to_choi_from_hs(c_sys: CompositeSystem, hs: np.ndarray) -> np.ndarray:
    """converts HS representation to Choi matrix of this gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    hs : np.ndarray
        HS representation of this gate.

    Returns
    -------
    np.ndarray
        Choi matrix of this gate.
    """
    choi_vec = c_sys._basis_basisconjugate_T_sparse.dot(hs.flatten())
    choi = choi_vec.reshape((c_sys.dim ** 2, c_sys.dim ** 2))
    return choi


def to_hs_from_choi(c_sys: CompositeSystem, choi: np.ndarray) -> np.ndarray:
    """converts Choi matrix to HS representation of this gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    choi : np.ndarray
        Choi matrix of this gate.

    Returns
    -------
    np.ndarray
        HS representation of this gate.
    """
    num_basis = len(c_sys.basis().basis)
    hs = np.zeros((num_basis, num_basis), dtype=np.float64)

    for alpha, beta in itertools.product(range(num_basis), range(num_basis)):
        b_bc = c_sys.basis_basisconjugate((alpha, beta))
        b_bc_dag = np.conjugate(b_bc.T)
        hs[alpha, beta] = (np.trace(b_bc_dag @ choi)).real.astype(np.float64)

    return hs


def to_hs_from_choi_with_dict(c_sys: CompositeSystem, choi: np.ndarray) -> np.ndarray:
    """converts Choi matrix to HS representation of this gate.

    this function uses dict to calculate fast.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    choi : np.ndarray
        Choi matrix of this gate.

    Returns
    -------
    np.ndarray
        HS representation of this gate.
    """
    num_basis = len(c_sys.basis())
    hs = np.zeros((num_basis, num_basis), dtype=np.complex128)

    for alpha, beta in itertools.product(range(num_basis), range(num_basis)):
        non_zeros = c_sys._dict_from_choi_to_hs.get((alpha, beta), [])
        for i, j, coefficient in non_zeros:
            hs[alpha, beta] += coefficient * choi[j, i]

    return mutil.truncate_hs(hs)


def to_hs_from_choi_with_sparsity(
    c_sys: CompositeSystem, choi: np.ndarray
) -> np.ndarray:
    """converts Choi matrix to HS representation of this gate.

    this function uses the scipy.sparse module.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    choi : np.ndarray
        Choi matrix of this gate.

    Returns
    -------
    np.ndarray
        HS representation of this gate.
    """
    hs_vec = c_sys._basisconjugate_basis_sparse.dot(choi.flatten())
    hs = hs_vec.reshape((c_sys.dim ** 2, c_sys.dim ** 2))

    return mutil.truncate_hs(hs)


def to_choi_from_var(
    c_sys: CompositeSystem,
    var: np.ndarray,
    on_para_eq_constraint: bool = True,
) -> np.ndarray:
    """converts variables to Choi matrix.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    var : np.ndarray
        variables.
    on_para_eq_constraint : bool, optional
        whether this gate is on parameter equality constraint, by default True

    Returns
    -------
    np.ndarray
        Choi matrix of this gate.
    """
    # var to hs
    hs = convert_var_to_hs(c_sys, var, on_para_eq_constraint)

    # hs to Choi matrix
    choi = to_choi_from_hs(c_sys, hs)
    return choi


def to_var_from_choi(
    c_sys: CompositeSystem,
    choi: np.ndarray,
    on_para_eq_constraint: bool = True,
) -> np.ndarray:
    """converts Choi matrix to variables.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    choi : np.ndarray
        Choi matrix of this gate.
    on_para_eq_constraint : bool, optional
        whether this gate is on parameter equality constraint, by default True

    Returns
    -------
    np.ndarray
        variables.
    """
    # Choi matrix to hs
    hs = to_choi_from_hs(c_sys, choi)

    # hs to var
    var = convert_hs_to_var(c_sys, hs, on_para_eq_constraint)
    return var


def convert_var_index_to_gate_index(
    c_sys: CompositeSystem, var_index: int, on_para_eq_constraint: bool = True
) -> Tuple[int, int]:
    """converts variable index to gate index.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    var_index : int
        variable index.
    on_para_eq_constraint : bool, optional
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
    if on_para_eq_constraint:
        row += 1
    return (row, col)


def convert_gate_index_to_var_index(
    c_sys: CompositeSystem,
    gate_index: Tuple[int, int],
    on_para_eq_constraint: bool = True,
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
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    int
        variable index.
    """
    dim = c_sys.dim
    (row, col) = gate_index
    var_index = (
        (dim ** 2) * (row - 1) + col
        if on_para_eq_constraint
        else (dim ** 2) * row + col
    )
    return var_index


def convert_var_to_hs(
    c_sys: CompositeSystem,
    var: np.ndarray,
    on_para_eq_constraint: bool = True,
) -> np.ndarray:
    """converts variables of gate to HS representation.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    var : np.ndarray
        variables of gate.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    np.ndarray
        HS representation of this gate.
    """
    dim = c_sys.dim

    size = (dim ** 2 - 1, dim ** 2) if on_para_eq_constraint else (dim ** 2, dim ** 2)
    reshaped = var.reshape(size)

    hs = (
        np.insert(reshaped, 0, np.eye(1, dim ** 2), axis=0)
        if on_para_eq_constraint
        else reshaped
    )
    return hs


def convert_var_to_gate(
    c_sys: CompositeSystem,
    var: np.ndarray,
    is_physicality_required: bool = True,
    is_estimation_object: bool = True,
    on_para_eq_constraint: bool = True,
    on_algo_eq_constraint: bool = True,
    on_algo_ineq_constraint: bool = True,
    mode_proj_order: str = "eq_ineq",
    eps_proj_physical: float = None,
) -> Gate:
    """converts vec of variables to gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    var : np.ndarray
        vec of variables.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    Gate
        converted gate.
    """
    hs = convert_var_to_hs(c_sys, var, on_para_eq_constraint)
    gate = Gate(
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
    return gate


def convert_hs_to_var(
    c_sys: CompositeSystem, hs: np.ndarray, on_para_eq_constraint: bool = True
) -> np.ndarray:
    """converts hs of gate to vec of variables.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    hs : np.ndarray
        HS representation of this gate.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    np.ndarray
        vec of variables.
    """
    var = np.delete(hs, 0, axis=0).flatten() if on_para_eq_constraint else hs.flatten()
    return var


def calc_gradient_from_gate(
    c_sys: CompositeSystem,
    hs: np.ndarray,
    var_index: int,
    is_estimation_object: bool = True,
    on_para_eq_constraint: bool = True,
    on_algo_eq_constraint: bool = True,
    on_algo_ineq_constraint: bool = True,
    mode_proj_order: str = "eq_ineq",
    eps_proj_physical: float = None,
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
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    Gate
        Gate with gradient as hs.
    """
    gradient = np.zeros((c_sys.dim ** 2, c_sys.dim ** 2), dtype=np.float64)
    gate_index = convert_var_index_to_gate_index(
        c_sys, var_index, on_para_eq_constraint
    )
    gradient[gate_index] = 1

    gate = Gate(
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
    return gate


def is_hp(hs: np.ndarray, basis: MatrixBasis, atol: float = None) -> bool:
    """returns whether gate is HP(Hermiticity-Preserving).

    HP <=> HS on Hermitian basis is real matrix.
    therefore converts input basis to Pauli basis, and checks whetever converted HS is real matrix.

    Parameters
    ----------
    hs : np.ndarray
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

    atol = Settings.get_atol() if atol is None else atol

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
    # check type
    if type(g) != Gate or type(u) != Gate:
        raise ValueError(
            f"type of g and u must be Gate. type of g={type(g)}, type of u={type(u)}"
        )

    # u: unitary gate <=> HS(u) is unitary
    # whetever HS(u) is unitary
    if not mutil.is_unitary(u.hs):
        raise ValueError("gate u must be unitary")

    # let trace = Tr[HS(u)^{\dagger}HS(g)]
    # AGF = 1-\frac{d^2-trace}{d(d+1)}
    d = u.dim
    trace = np.vdot(u.hs, g.hs)
    agf = 1 - (d ** 2 - trace) / (d * (d + 1))
    return agf


def convert_hs(
    from_hs: np.ndarray, from_basis: MatrixBasis, to_basis: MatrixBasis
) -> np.ndarray:
    """returns HS representation for ``to_basis``

    Parameters
    ----------
    from_hs : np.ndarray
        HS representation before convert.
    from_basis : MatrixBasis
        basis before convert.
    to_basis : MatrixBasis
        basis after convert.

    Returns
    -------
    np.ndarray
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
    matrix: np.ndarray, c_sys: CompositeSystem
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
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
    else:
        # control bit is 2nd qubit
        hs_comp_basis = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
            [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
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
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )

    hs_for_c_sys = convert_hs(
        hs_comp_basis, c_sys.comp_basis(), c_sys.basis()
    ).real.astype(np.float64)
    gate = Gate(c_sys, hs_for_c_sys)
    return gate


def get_depolarizing_channel(p: float, c_sys: Optional[CompositeSystem] = None) -> Gate:
    if not (0 <= p <= 1):
        message = "`p` must be between 0 and 1."
        raise ValueError(message)
    if not c_sys:
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
    # 1, 1-p, 1-p, ... 1-p
    source = np.array([1] + [1 - p] * (c_sys.dim ** 2 - 1), dtype=np.float64)
    hs = np.diag(source)
    gate = Gate(hs=hs, c_sys=c_sys)
    return gate


def get_x_rotation(theta: float, c_sys: Optional[CompositeSystem] = None) -> Gate:
    hs = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.cos(theta), -np.sin(theta)],
            [0, 0, np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float64,
    )
    if not c_sys:
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
    gate = Gate(hs=hs, c_sys=c_sys)
    return gate


def get_amplitutde_damping_channel(
    gamma: float, c_sys: Optional[CompositeSystem] = None
) -> Gate:
    hs = np.array(
        [
            [1, 0, 0, 0],
            [0, np.sqrt(1 - gamma), 0, 0],
            [0, 0, np.sqrt(1 - gamma), 0],
            [gamma, 0, 0, 1 - gamma],
        ],
        dtype=np.float64,
    )
    if not c_sys:
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
    gate = Gate(hs=hs, c_sys=c_sys)
    return gate
