import copy
from functools import reduce
from operator import mul
from typing import List, Tuple, Union

import numpy as np

from quara.objects.composite_system import CompositeSystem
from quara.objects import gate
from quara.objects.gate import Gate,to_hs_from_kraus_matrices
from quara.objects.matrix_basis import (
    SparseMatrixBasis,
)
from quara.objects.povm import Povm
from quara.objects.qoperation import QOperation
from quara.utils.index_util import index_serial_from_index_multi_dimensional
from quara.utils.number_util import to_stream


class MProcess(QOperation):
    def __init__(
        self,
        c_sys: CompositeSystem,
        hss: List[np.ndarray],
        shape: Tuple[int] = None,
        mode_sampling: bool = False,
        random_seed_or_generator: Union[int, np.random.Generator] = None,
        is_physicality_required: bool = True,
        is_estimation_object: bool = True,
        on_para_eq_constraint: bool = True,
        on_algo_eq_constraint: bool = True,
        on_algo_ineq_constraint: bool = True,
        mode_proj_order: str = "eq_ineq",
        eps_proj_physical: float = None,
        eps_truncate_imaginary_part: float = None,
        eps_zero: Union[float, np.float64] = 10 ** -8,
    ):
        super().__init__(
            c_sys=c_sys,
            is_physicality_required=is_physicality_required,
            is_estimation_object=is_estimation_object,
            on_para_eq_constraint=on_para_eq_constraint,
            on_algo_eq_constraint=on_algo_eq_constraint,
            on_algo_ineq_constraint=on_algo_ineq_constraint,
            mode_proj_order=mode_proj_order,
            eps_proj_physical=eps_proj_physical,
            eps_truncate_imaginary_part=eps_truncate_imaginary_part,
        )
        self._hss: List[np.ndarray] = hss
        self._num_outcomes = len(self._hss)

        # whether all ElementalSystem of this CompositeSystem are orthonormal, hermitian and 0th prop identity
        if c_sys.is_orthonormal_hermitian_0thprop_identity == False:
            raise ValueError(
                "all ElementalSystem of this CompositeSystem must be orthonormal, hermitian and 0th prop identity."
            )

        for i, hs in enumerate(self._hss):
            # whether HS representation is square matrix
            size = hs.shape
            if size[0] != size[1]:
                raise ValueError(
                    f"HS must be square matrix. size of hss[{i}] is {size}"
                )

            # whether dim of HS representation is square number
            self._dim: int = int(np.sqrt(size[0]))
            if self._dim ** 2 != size[0]:
                raise ValueError(
                    f"dim of HS must be square number. dim of hss[{i}] is {size[0]}"
                )

            # whether HS representation is real matrix
            if hs.dtype != np.float64:
                raise ValueError(
                    f"HS must be real matrix. dtype of hss[{i}] is {hs.dtype}"
                )

        # whether dim of HS equals dim of CompositeSystem
        if self._dim != self.composite_system.dim:
            raise ValueError(
                f"dim of HS must equal dim of CompositeSystem.  dim of HS is {self._dim}. dim of CompositeSystem is {self.composite_system.dim}"
            )

        if shape is None:
            self._shape = (len(self._hss),)
        else:
            # validation about data size
            if len(self._hss) != reduce(mul, shape):
                raise ValueError(
                    f"the size of ps({len(self._hss)}) and shape({shape}) do not match."
                )
            self._shape = shape

        self.set_mode_sampling(mode_sampling, random_seed_or_generator)

        # whether eps_zero is a non-negative value.
        if eps_zero < 0:
            raise ValueError(
                "eps_zero must be a non-negative value. eps_zero is {eps_zero}"
            )
        self._eps_zero: Union[float, np.float64] = eps_zero

        # whether the mprocess is physically correct
        if self.is_physicality_required and not self.is_physical():
            raise ValueError("the mprocess is not physically correct.")

    def _info(self):
        info = {}
        info["Type"] = self.__class__.__name__
        info["Dim"] = self.dim
        info["HSs"] = self.hss
        info["ModeSampling"] = self.mode_sampling
        return info

    @property  # read only
    def dim(self):
        """returns dim of gate.

        Returns
        -------
        int
            dim of gate.
        """
        return self._dim

    @property  # read only
    def num_outcomes(self) -> int:
        """Property to get the number of HSs.

        Returns
        -------
        int
            the number of HSs.
        """
        return self._num_outcomes

    @property  # read only
    def hss(self):
        """returns HS representations of MProcess.

        Returns
        -------
        np.ndarray
            HS representations of MProcess.
        """
        return self._hss

    def hs(self, index: Union[int, Tuple[int]]) -> np.ndarray:
        """returns HS representations of MProcess by index.

        Parameters
        ----------
        index : Union[int, Tuple[int]]
            index of HS of MProcess.
            If type is an int, access is one-dimensional.
            If type is tuple, access is multi-dimensional.

        Returns
        -------
        np.ndarray
            HS of MProcess by index.

        Raises
        ------
        ValueError
            length of tuple does not equal length of the list of HS.
        IndexError
            specified index does not exist in the list of HS.
        """
        if type(index) == tuple:
            # whether size of tuple equals length of the list of HS
            if len(index) != len(self.shape):
                raise ValueError(
                    f"length of tuple must equal length of the list of HS. length of tuple={len(index)}, length of the list of HS={len(self.shape)}"
                )

            serial_index = index_serial_from_index_multi_dimensional(self.shape, index)
            return self.hss[serial_index]
        else:
            return self.hss[index]

    @property  # read only
    def shape(self) -> Tuple[int]:
        """returns shape.

        Returns
        -------
        Tuple[int]
            the shape of MProcess.
        """
        return self._shape

    @property  # read only
    def mode_sampling(self) -> bool:
        """returns the mode of sampling.

        if mode_sampling is True, samples to determine one HS.

        Returns
        -------
        bool
            the mode of sampling.
        """
        return self._mode_sampling

    @property  # read only
    def random_seed_or_generator(self) -> Union[int, np.random.Generator]:
        """returns the random seed or state to sample HS.

        Returns
        -------
        Union[int, np.random.Generator]
            the random seed or state to sample HS.
        """
        return self._random_seed_or_generator

    @property  # read only
    def random_state(self) -> np.random.Generator:
        """returns the random state to sample HS.

        Returns
        -------
        np.random.Generator
            the random state to sample HS.
        """
        return self._random_state

    def set_mode_sampling(
        self,
        mode_sampling: bool,
        random_seed_or_generator: Union[int, np.random.Generator] = None,
    ) -> None:
        # although mode_sampling is False, random_seed_or_generator is not None
        if mode_sampling == False and random_seed_or_generator is not None:
            raise ValueError(
                "although mode_sampling is False, random_seed_or_generator is not None."
            )

        self._mode_sampling: bool = mode_sampling
        if self.mode_sampling == True:
            self._random_seed_or_generator = random_seed_or_generator
            self._random_state = to_stream(self._random_seed_or_generator)
        else:
            self._random_seed_or_generator = None
            self._random_state = None

    @property  # read only
    def eps_zero(self):
        return self._eps_zero

    def is_eq_constraint_satisfied(self, atol: float = None) -> bool:
        return self.is_sum_tp(atol=atol)

    def is_ineq_constraint_satisfied(self, atol: float = None) -> bool:
        return self.is_cp(atol=atol)

    def set_zero(self):
        hs = np.zeros(self.hs(0).shape, dtype=np.float64)
        self._hss = [hs.copy() for _ in self.hss]
        self._is_physicality_required = False

    def _generate_zero_obj(self) -> np.ndarray:
        hs = np.zeros(self.hs(0).shape, dtype=np.float64)
        new_hss = [hs.copy() for _ in self.hss]
        return new_hss

    def generate_zero_obj(self) -> "MProcess":
        """returns zero object of QOperation.

        Returns
        -------
        QOperation
            zero object of QOperation.
        """
        hss = self._generate_zero_obj()
        new_qoperation = MProcess(
            self.composite_system,
            hss,
            shape=self.shape,
            mode_sampling=self.mode_sampling,
            random_seed_or_generator=self.random_seed_or_generator,
            is_physicality_required=False,
            is_estimation_object=False,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
            eps_truncate_imaginary_part=self.eps_truncate_imaginary_part,
        )
        return new_qoperation

    def _generate_origin_obj(self) -> np.ndarray:
        hs = np.zeros(self.hs(0).shape, dtype=np.float64)
        hs[0][0] = 1 / len(self.hss)
        new_hss = [hs.copy() for _ in self.hss]
        return new_hss

    def generate_origin_obj(self) -> "MProcess":
        """returns origin object of MProcess.

        Returns
        -------
        MProcess
            origin object of MProcess.
        """
        hss = self._generate_origin_obj()
        new_qoperation = MProcess(
            self.composite_system,
            hss,
            shape=self.shape,
            mode_sampling=self.mode_sampling,
            random_seed_or_generator=self.random_seed_or_generator,
            is_physicality_required=False,
            is_estimation_object=False,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
            eps_truncate_imaginary_part=self.eps_truncate_imaginary_part,
        )
        return new_qoperation

    def to_var(self) -> np.ndarray:
        return convert_hss_to_var(
            c_sys=self.composite_system,
            hss=self.hss,
            on_para_eq_constraint=self.on_para_eq_constraint,
        )

    def to_stacked_vector(self) -> np.ndarray:
        stacked_vec = np.array(self.hss).flatten()
        return stacked_vec

    def _embed_qoperation_from_qutrits_to_qubits(
        self, perm_matrix, c_sys_qubits
    ) -> QOperation:
        num_qutrits = self.composite_system.num_e_sys

        mats_qutrits_list = []
        num_kraus = 0
        for index in range(len(self.hss)):
            mats_qutrits = self.to_kraus_matrices(index)
            mats_qutrits_list.append(mats_qutrits)
            num_kraus += len(mats_qutrits)
        coeff = 1 / np.sqrt(num_kraus)

        # calc matrices for qubits
        hss = []
        for mats_qutrits in mats_qutrits_list:
            kraus_qubits = []
            for mat_qutrits in mats_qutrits:
                mat_qubits = QOperation._calc_matrix_from_qutrits_to_qubits(
                    num_qutrits, perm_matrix, mat_qutrits, coeff
                )
                kraus_qubits.append(mat_qubits)

            # convert to hss
            hs = to_hs_from_kraus_matrices(
                c_sys_qubits,
                kraus_qubits,
                eps_truncate_imaginary_part=self.eps_truncate_imaginary_part,
            )
            hss.append(hs)

        # gerenera qoperation for qubits
        new_qope = MProcess(
            c_sys_qubits,
            hss,
            shape=self.shape,
            mode_sampling=self.mode_sampling,
            random_seed_or_generator=self.random_seed_or_generator,
            is_physicality_required=self.is_physicality_required,
            is_estimation_object=self.is_estimation_object,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
            eps_truncate_imaginary_part=self.eps_truncate_imaginary_part,
            eps_zero=self.eps_zero,
        )
        return new_qope

    def calc_gradient(self, var_index: int) -> "MProcess":
        mprocess = calc_gradient_from_mprocess(
            self.composite_system,
            self.hss,
            var_index,
            shape=self.shape,
            mode_sampling=self.mode_sampling,
            random_seed_or_generator=self.random_seed_or_generator,
            is_estimation_object=self.is_estimation_object,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
            eps_truncate_imaginary_part=self.eps_truncate_imaginary_part,
        )
        return mprocess

    def calc_proj_eq_constraint(self) -> "MProcess":
        dim = self.composite_system.dim
        hss = copy.deepcopy(self.hss)

        # calc new var
        vec = np.zeros((dim ** 2))
        for hs in hss:
            vec += hs[0]
        vec[0] -= 1

        new_hss = []
        for hs in hss:
            hs[0] -= vec / len(hss)
            new_hss.append(hs)

        # create new MProcess
        new_mprocess = MProcess(
            c_sys=self.composite_system,
            hss=new_hss,
            shape=self.shape,
            mode_sampling=self.mode_sampling,
            random_seed_or_generator=self.random_seed_or_generator,
            is_physicality_required=self.is_physicality_required,
            is_estimation_object=self.is_estimation_object,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
            eps_truncate_imaginary_part=self.eps_truncate_imaginary_part,
        )

        return new_mprocess

    @staticmethod
    def calc_proj_eq_constraint_with_var(
        c_sys: CompositeSystem,
        var: np.ndarray,
        on_para_eq_constraint: bool = True,
    ) -> np.ndarray:
        dim = c_sys.dim

        # var to hss
        hss = convert_var_to_hss(
            c_sys, var, on_para_eq_constraint=on_para_eq_constraint
        )

        # calc new var
        vec = np.zeros((dim ** 2))
        for hs in hss:
            vec += hs[0]
        vec[0] -= 1

        new_hss = []
        for hs in hss:
            hs[0] -= vec / len(hss)
            new_hss.append(hs)

        # hss to var
        new_var = convert_hss_to_var(c_sys, new_hss, on_para_eq_constraint)

        return new_var

    def calc_proj_ineq_constraint(self) -> "MProcess":
        new_hss = []
        dim = self.composite_system.dim
        for hs in self.hss:
            proj_hs = Gate.calc_proj_ineq_constraint_with_var(
                self.composite_system,
                hs.flatten(),
                on_para_eq_constraint=False,
                eps_truncate_imaginary_part=self.eps_truncate_imaginary_part,
            ).reshape((dim ** 2, dim ** 2))
            new_hss.append(proj_hs)

        # create new MProcess
        new_mprocess = MProcess(
            c_sys=self.composite_system,
            hss=new_hss,
            shape=self.shape,
            mode_sampling=self.mode_sampling,
            random_seed_or_generator=self.random_seed_or_generator,
            is_physicality_required=self.is_physicality_required,
            is_estimation_object=self.is_estimation_object,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
            eps_truncate_imaginary_part=self.eps_truncate_imaginary_part,
        )

        return new_mprocess

    @staticmethod
    def calc_proj_ineq_constraint_with_var(
        c_sys: CompositeSystem,
        var: np.ndarray,
        on_para_eq_constraint: bool = True,
        eps_truncate_imaginary_part: float = None,
    ) -> np.ndarray:
        # var to hss
        hss = convert_var_to_hss(
            c_sys, var, on_para_eq_constraint=on_para_eq_constraint
        )

        # calc new var
        new_var = np.array([], dtype=np.float64)
        for hs_index, hs in enumerate(hss):
            proj_hs = Gate.calc_proj_ineq_constraint_with_var(
                c_sys,
                hs.flatten(),
                on_para_eq_constraint=False,
                eps_truncate_imaginary_part=eps_truncate_imaginary_part,
            )
            if on_para_eq_constraint is True and hs_index == len(hss) - 1:
                proj_hs = np.delete(proj_hs, np.s_[0 : c_sys.dim ** 2])
            new_var = np.append(new_var, proj_hs)

        return new_var

    def generate_from_var(
        self,
        var: np.ndarray,
        is_physicality_required: bool = None,
        is_estimation_object: bool = None,
        on_para_eq_constraint: bool = None,
        on_algo_eq_constraint: bool = None,
        on_algo_ineq_constraint: bool = None,
        mode_proj_order: str = "eq_ineq",
        eps_proj_physical: float = None,
        eps_truncate_imaginary_part: float = None,
    ) -> "QOperation":
        """
        generates QOperation from variables.

        Parameters
        ----------
        var : np.ndarray
            variables.
        is_physicality_required : bool, optional
            whether this QOperation is physicality required, by default None.
            if this parameter is None, the value of this instance is set.
        is_estimation_object : bool, optional
            whether this QOperation is estimation object, by default None.
            if this parameter is None, the value of this instance is set.
        on_para_eq_constraint : bool, optional
            whether this QOperation is on parameter equality constraint, by default None.
            if this parameter is None, the value of this instance is set.
        on_algo_eq_constraint : bool, optional
            whether this QOperation is on algorithm equality constraint, by default None.
            if this parameter is None, the value of this instance is set.
        on_algo_ineq_constraint : bool, optional
            whether this QOperation is on algorithm inequality constraint, by default None.
            if this parameter is None, the value of this instance is set.
        mode_proj_order : str, optional
            the order in which the projections are performed, by default "eq_ineq".
        eps_proj_physical : float, optional
            epsiron that is projection algorithm error threshold for being physical, by default None.
            if this parameter is None, the value of this instance is set.
        eps_truncate_imaginary_part : float, optional
            threshold to truncate imaginary part, by default :func:`~quara.settings.Settings.get_atol`

        Returns
        -------
        QOperation
            generated QOperation.
        """
        is_physicality_required = (
            self.is_physicality_required
            if is_physicality_required is None
            else is_physicality_required
        )
        is_estimation_object = (
            self.is_estimation_object
            if is_estimation_object is None
            else is_estimation_object
        )
        on_para_eq_constraint = (
            self.on_para_eq_constraint
            if on_para_eq_constraint is None
            else on_para_eq_constraint
        )
        on_algo_eq_constraint = (
            self.on_algo_eq_constraint
            if on_algo_eq_constraint is None
            else on_algo_eq_constraint
        )
        on_algo_ineq_constraint = (
            self.on_algo_ineq_constraint
            if on_algo_ineq_constraint is None
            else on_algo_ineq_constraint
        )
        eps_proj_physical = (
            self.eps_proj_physical if eps_proj_physical is None else eps_proj_physical
        )
        c_sys = self.composite_system
        hss = convert_var_to_hss(
            c_sys, var, on_para_eq_constraint=on_para_eq_constraint
        )

        new_qoperation = MProcess(
            c_sys=c_sys,
            hss=hss,
            shape=self.shape,
            mode_sampling=self.mode_sampling,
            random_seed_or_generator=self.random_seed_or_generator,
            is_physicality_required=is_physicality_required,
            is_estimation_object=is_estimation_object,
            on_para_eq_constraint=on_para_eq_constraint,
            on_algo_eq_constraint=on_algo_eq_constraint,
            on_algo_ineq_constraint=on_algo_ineq_constraint,
            mode_proj_order=mode_proj_order,
            eps_proj_physical=eps_proj_physical,
            eps_truncate_imaginary_part=eps_truncate_imaginary_part,
        )
        return new_qoperation

    def _check_shape(self, shape_left: Tuple[int], shape_right: Tuple[int]):
        if shape_left != shape_right:
            raise ValueError(
                f"shape of MProcess of operands don't equal. shape of left={shape_left.shape}, shape of right={shape_right.shape}"
            )

    def _add_vec(self, other) -> List[np.ndarray]:
        self._check_shape(self.shape, other.shape)
        new_hss = [self_hs + other_hs for self_hs, other_hs in zip(self.hss, other.hss)]
        return new_hss

    def _sub_vec(self, other) -> List[np.ndarray]:
        self._check_shape(self.shape, other.shape)
        new_hss = [self_hs - other_hs for self_hs, other_hs in zip(self.hss, other.hss)]
        return new_hss

    def _mul_vec(self, other) -> List[np.ndarray]:
        new_hss = [hs * other for hs in self.hss]
        return new_hss

    def _truediv_vec(self, other) -> List[np.ndarray]:
        new_hss = [hs / other for hs in self.hss]
        return new_hss

    def get_basis(self) -> SparseMatrixBasis:
        """returns MatrixBasis of gate.
        Returns
        -------
        MatrixBasis
            MatrixBasis of gate.
        """
        return self.composite_system.basis()

    def is_sum_tp(self, atol: float = None) -> bool:
        sum_hss = np.sum(self._hss, axis=0)
        return gate.is_tp(self.composite_system, sum_hss, atol)

    def is_cp(self, atol: float = None) -> bool:
        for hs in self.hss:
            if gate.is_cp(self.composite_system, hs, atol) == False:
                return False
        return True

    def convert_basis(self, other_basis: SparseMatrixBasis) -> List[np.ndarray]:
        """returns list of HS representations for ``other_basis``.
        Parameters
        ----------
        other_basis : MatrixBasis
            basis.
        Returns
        -------
        List[np.ndarray]
            list of HS representations for ``other_basis``.
        """
        converted_hss = [
            gate.convert_hs(hs, self.composite_system.basis(), other_basis)
            for hs in self.hss
        ]
        return converted_hss

    def convert_to_comp_basis(self, mode: str = "row_major") -> List[np.ndarray]:
        """returns list of HS representations for computational basis.
        Parameters
        ----------
        mode : str, optional
            specify whether the order of basis is "row_major" or "column_major", by default "row_major".
        Returns
        -------
        List[np.ndarray]
            list of HS representations for computational basis.
        """
        converted_hss = [
            gate.convert_hs(
                hs,
                self.composite_system.basis(),
                self.composite_system.comp_basis(mode=mode),
            )
            for hs in self.hss
        ]
        return converted_hss

    def to_choi_matrix(self, outcome: Union[int, Tuple[int]]) -> np.ndarray:
        """returns Choi matrix of gate.

        Parameters
        ----------
        outcome : Union[int, Tuple[int]]
            index of HS of MProcess.
            If type is an int, access is one-dimensional.
            If type is tuple, access is multi-dimensional.

        Returns
        -------
        np.ndarray
            Choi matrix of gate.
        """
        hs = self.hs(outcome)
        return gate.to_choi_from_hs(self.composite_system, hs)

    def to_choi_matrix_with_dict(self, outcome: Union[int, Tuple[int]]) -> np.ndarray:
        """returns Choi matrix of gate.

        Parameters
        ----------
        outcome : Union[int, Tuple[int]]
            index of HS of MProcess.
            If type is an int, access is one-dimensional.
            If type is tuple, access is multi-dimensional.

        Returns
        -------
        np.ndarray
            Choi matrix of gate.
        """
        hs = self.hs(outcome)
        return gate.to_choi_from_hs_with_dict(self.composite_system, hs)

    def to_choi_matrix_with_sparsity(
        self, outcome: Union[int, Tuple[int]]
    ) -> np.ndarray:
        """returns Choi matrix of gate.

        Parameters
        ----------
        outcome : Union[int, Tuple[int]]
            index of HS of MProcess.
            If type is an int, access is one-dimensional.
            If type is tuple, access is multi-dimensional.

        Returns
        -------
        np.ndarray
            Choi matrix of gate.
        """
        hs = self.hs(outcome)
        return gate.to_choi_from_hs_with_sparsity(self.composite_system, hs)

    def to_kraus_matrices(self, outcome: Union[int, Tuple[int]]) -> List[np.ndarray]:
        """returns Kraus matrices of gate.

        this function returns Kraus matrices as list of ``np.ndarray`` with ``dtype=np.complex128``.
        the list is sorted large eigenvalue order.
        if HS of gate is not CP, then returns empty list because Kraus matrices does not exist.

        Parameters
        ----------
        outcome : Union[int, Tuple[int]]
            index of HS of MProcess.
            If type is an int, access is one-dimensional.
            If type is tuple, access is multi-dimensional.

        Returns
        -------
        List[np.ndarray]
            Kraus matrices of gate.
        """
        hs = self.hs(outcome)
        return gate.to_kraus_matrices_from_hs(self.composite_system, hs)

    def to_process_matrix(self, outcome: Union[int, Tuple[int]]) -> np.ndarray:
        """returns process matrix of gate.

        Parameters
        ----------
        outcome : Union[int, Tuple[int]]
            index of HS of MProcess.
            If type is an int, access is one-dimensional.
            If type is tuple, access is multi-dimensional.

        Returns
        -------
        np.ndarray
            process matrix of gate.
        """
        hs = self.hs(outcome)
        return gate.to_process_matrix_from_hs(self.composite_system, hs)

    def _copy(self):
        return (
            copy.deepcopy(self.hss),
            copy.deepcopy(self.shape),
            copy.deepcopy(self.mode_sampling),
            copy.deepcopy(self.random_seed_or_generator),
        )

    def copy(self) -> "MProcess":
        """returns copy of MProcess.

        Returns
        -------
        MProcess
            copy of MProcess.
        """
        hss, shape, mode_sampling, random_seed_or_generator = self._copy()
        new_qoperation = self.__class__(
            self.composite_system,
            hss,
            shape=shape,
            mode_sampling=mode_sampling,
            random_seed_or_generator=random_seed_or_generator,
            is_physicality_required=self.is_physicality_required,
            is_estimation_object=self.is_estimation_object,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
            eps_truncate_imaginary_part=self.eps_truncate_imaginary_part,
        )
        return new_qoperation

    @staticmethod
    def convert_var_to_stacked_vector(
        c_sys: CompositeSystem,
        var: np.ndarray,
        on_para_eq_constraint: bool = True,
    ) -> np.ndarray:
        """converts variables of MProcess to stacked vector of MProcess.

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem of this MProcess.
        var : np.ndarray
            variables of MProcess.
        on_para_eq_constraint : bool, optional
            uses equal constraints, by default True.

        Returns
        -------
        np.ndarray
            stacked vector of MProcess.
        """
        if on_para_eq_constraint:

            vector = copy.copy(var)
            dim = c_sys.dim
            hs_size = dim ** 2 * dim ** 2
            num_outcomes = vector.shape[0] // hs_size + 1
            one = np.zeros(dim ** 2, dtype=np.float64)
            one[0] = 1
            sum_first_row = np.zeros(dim ** 2, dtype=np.float64)
            for outcome in range(num_outcomes - 1):
                sum_first_row += vector[
                    hs_size * outcome : hs_size * outcome + dim ** 2
                ]
            first_row_of_last_hs = one - sum_first_row
            stacked_vector = np.insert(
                vector, hs_size * (num_outcomes - 1), first_row_of_last_hs
            )

        else:
            stacked_vector = var

        return stacked_vector

    @staticmethod
    def convert_stacked_vector_to_var(
        c_sys: CompositeSystem,
        stacked_vector: np.ndarray,
        on_para_eq_constraint: bool = True,
    ) -> np.ndarray:
        """converts stacked vector of MProcess to variables of MProcess.

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem of this MProcess.
        stacked_vector : np.ndarray
            stacked vector of MProcess.
        on_para_eq_constraint : bool, optional
            uses equal constraints, by default True.

        Returns
        -------
        np.ndarray
            variables of MProcess.
        """
        if on_para_eq_constraint:
            dim = c_sys.dim
            hs_size = dim ** 2 * dim ** 2
            num_outcomes = stacked_vector.shape[0] // hs_size
            var = np.delete(
                stacked_vector,
                np.s_[
                    hs_size * (num_outcomes - 1) : hs_size * (num_outcomes - 1)
                    + c_sys.dim ** 2
                ],
            )
        else:
            var = stacked_vector

        return var

    def to_povm(self) -> Povm:
        vecs = [np.sqrt(self.dim) * hs[0] for hs in self.hss]
        povm = Povm(
            c_sys=self.composite_system,
            vecs=vecs,
            is_physicality_required=self.is_physicality_required,
            is_estimation_object=self.is_estimation_object,
            on_para_eq_constraint=self.on_para_eq_constraint,
            on_algo_eq_constraint=self.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.on_algo_ineq_constraint,
            mode_proj_order=self.mode_proj_order,
            eps_proj_physical=self.eps_proj_physical,
        )
        return povm


def convert_var_index_to_mprocess_index(
    c_sys: CompositeSystem,
    hss: List[np.ndarray],
    var_index: int,
    on_para_eq_constraint: bool = True,
) -> Tuple[int, int, int]:
    """converts variable index to MProcess index.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this MProcess.
    hss : List[np.ndarray]
        list of HS representation.
    var_index : int
        variable index.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    Tuple[int, int, int]
        MProcess index.
        first value of tuple is index of list of HS representation of this MProcess.
        second value of tuple is row number of HS representation of this MProcess.
        third value of tuple is column number of HS representation of this MProcess.
    """
    dim = c_sys.dim
    hs_size = dim ** 2 * dim ** 2

    (hs_index, matrix_index) = divmod(var_index, hs_size)
    (row, col) = divmod(matrix_index, dim ** 2)
    if on_para_eq_constraint:
        if hs_index == len(hss) - 1:
            row += 1
    return (hs_index, row, col)


def convert_mprocess_index_to_var_index(
    c_sys: CompositeSystem,
    mprocess_index: Tuple[int, int, int],
    hss: List[np.ndarray],
    on_para_eq_constraint: bool = True,
) -> int:
    """converts MProcess index to variable index.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this MProcess.
    mprocess_index : Tuple[int, int, int]
        MProcess index.
        first value of tuple is index of list of HS representation of this MProcess.
        second value of tuple is row number of HS representation of this MProcess.
        third value of tuple is column number of HS representation of this MProcess.
    hss : List[np.ndarray]
        list of HS representation.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    int
        variable index.
    """
    dim = c_sys.dim
    hs_size = dim ** 2 * dim ** 2

    (hs_index, row, col) = mprocess_index
    var_index = hs_index * hs_size + row * dim ** 2 + col
    if on_para_eq_constraint:
        if hs_index == len(hss) - 1:
            var_index -= dim ** 2
    return var_index


def convert_hss_to_var(
    c_sys: CompositeSystem, hss: List[np.ndarray], on_para_eq_constraint: bool = True
) -> np.ndarray:
    """converts hss of MProcess to variables.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this MProcess.
    hss : List[np.ndarray]
        list of HS representation of this MProcess.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    np.ndarray
        variables.
    """
    if on_para_eq_constraint:
        tmp_hss = []
        for index, hs in enumerate(hss):
            if index == len(hss) - 1:
                tmp_hss.append(np.delete(hs, 0, axis=0).flatten())
            else:
                tmp_hss.append(hs.flatten())
        var = np.hstack(tmp_hss)
    else:
        var = np.reshape(hss, -1)
    return var


def convert_var_to_hss(
    c_sys: CompositeSystem,
    var: np.ndarray,
    on_para_eq_constraint: bool = True,
) -> List[np.ndarray]:
    """converts variables of MProcess to list of HS representation.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this MProcess.
    var : np.ndarrayd@y
        variables of gate.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    List[np.ndarray]
        list of HS representation of this MProcess.
    """
    dim = c_sys.dim
    hs_size = dim ** 2 * dim ** 2

    if on_para_eq_constraint:

        vector = copy.copy(var)
        num_outcomes = vector.shape[0] // hs_size + 1

        one = np.zeros(dim ** 2, dtype=np.float64)
        one[0] = 1

        sum_first_row = np.zeros(dim ** 2, dtype=np.float64)
        for outcome in range(num_outcomes - 1):
            sum_first_row += vector[hs_size * outcome : hs_size * outcome + dim ** 2]
        first_row_of_last_hs = one - sum_first_row

        vector = np.insert(vector, hs_size * (num_outcomes - 1), first_row_of_last_hs)

    else:
        vector = var
        num_outcomes = vector.shape[0] // hs_size

    vec_list = []
    reshaped_vecs = vector.reshape((num_outcomes, dim ** 2, dim ** 2))
    # convert np.ndarray to list of np.ndarray
    for vec in reshaped_vecs:
        vec_list.append(vec)
    return vec_list


def calc_gradient_from_mprocess(
    c_sys: CompositeSystem,
    hss: List[np.ndarray],
    var_index: int,
    shape: Tuple[int] = None,
    mode_sampling: bool = False,
    random_seed_or_generator: Union[int, np.random.Generator] = None,
    is_estimation_object: bool = True,
    on_para_eq_constraint: bool = True,
    on_algo_eq_constraint: bool = True,
    on_algo_ineq_constraint: bool = True,
    mode_proj_order: str = "eq_ineq",
    eps_proj_physical: float = None,
    eps_truncate_imaginary_part: float = None,
) -> MProcess:
    """calculates gradient from MProcess.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this MProcess.
    hss : List[np.ndarray]
        list of HS representation of this MProcess.
    var_index : int
        variable index.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    MProcess
        MProcess with gradient as hss.
    """
    gradient = []
    for _ in hss:
        gradient.append(np.zeros((c_sys.dim ** 2, c_sys.dim ** 2), dtype=np.float64))

    (hs_index, row, col) = convert_var_index_to_mprocess_index(
        c_sys, hss, var_index, on_para_eq_constraint
    )
    gradient[hs_index][row][col] = 1

    mprocess = MProcess(
        c_sys,
        gradient,
        shape=shape,
        mode_sampling=mode_sampling,
        random_seed_or_generator=random_seed_or_generator,
        is_physicality_required=False,
        is_estimation_object=is_estimation_object,
        on_para_eq_constraint=on_para_eq_constraint,
        on_algo_eq_constraint=on_algo_eq_constraint,
        on_algo_ineq_constraint=on_algo_ineq_constraint,
        mode_proj_order=mode_proj_order,
        eps_proj_physical=eps_proj_physical,
        eps_truncate_imaginary_part=eps_truncate_imaginary_part,
    )
    return mprocess
