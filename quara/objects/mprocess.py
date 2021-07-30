import copy
import itertools
from functools import reduce
from operator import add, mul
from typing import List, Optional, Tuple, Union

import numpy as np

from quara.objects.composite_system import CompositeSystem, ElementalSystem
from quara.objects import gate
from quara.objects.matrix_basis import (
    MatrixBasis,
    get_comp_basis,
    get_normalized_pauli_basis,
)
from quara.objects.qoperation import QOperation
from quara.settings import Settings
from quara.utils.index_util import index_serial_from_index_multi_dimensional
import quara.utils.matrix_util as mutil
from quara.utils.number_util import to_stream


class MProcess(QOperation):
    def __init__(
        self,
        c_sys: CompositeSystem,
        hss: List[np.ndarray],
        shape: Tuple[int] = None,
        mode_sampling: bool = False,
        random_seed_or_state: Union[int, np.random.RandomState] = None,
        is_physicality_required: bool = True,
        is_estimation_object: bool = True,
        on_para_eq_constraint: bool = True,
        on_algo_eq_constraint: bool = True,
        on_algo_ineq_constraint: bool = True,
        mode_proj_order: str = "eq_ineq",
        eps_proj_physical: float = None,
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
        )
        self._hss: List[np.ndarray] = hss
        self._num_outcomes = len(self._hss)

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
                print(f"hs={hs}")
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

        self.set_mode_sampling(mode_sampling, random_seed_or_state)

        # whether the gate is physically correct
        if self.is_physicality_required and not self.is_physical():
            raise ValueError("the gate is not physically correct.")

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

    def hs(self, index: Union[int, Tuple]) -> np.ndarray:
        """returns HS representations of MProcess by index.

        Parameters
        ----------
        index : Union[int, Tuple]
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
    def random_seed_or_state(self) -> Union[int, np.random.RandomState]:
        """returns the random seed or state to sample HS.

        Returns
        -------
        Union[int, np.random.RandomState]
            the random seed or state to sample HS.
        """
        return self._random_seed_or_state

    @property  # read only
    def random_state(self) -> np.random.RandomState:
        """returns the random state to sample HS.

        Returns
        -------
        np.random.RandomState
            the random state to sample HS.
        """
        return self._random_state

    def set_mode_sampling(
        self,
        mode_sampling: bool,
        random_seed_or_state: Union[int, np.random.RandomState] = None,
    ) -> None:
        # although mode_sampling is False, random_seed_or_state is not None
        if mode_sampling == False and random_seed_or_state is not None:
            raise ValueError(
                "although mode_sampling is False, random_seed_or_state is not None."
            )

        self._mode_sampling: bool = mode_sampling
        if self.mode_sampling == True:
            self._random_seed_or_state = random_seed_or_state
            self._random_state = to_stream(self._random_seed_or_state)
        else:
            self._random_seed_or_state = None
            self._random_state = None

    def is_eq_constraint_satisfied(self, atol: float = None) -> bool:
        return self.is_sum_tp()

    def is_ineq_constraint_satisfied(self, atol: float = None) -> bool:
        return self.is_cp()

    def set_zero(self):
        hs = np.zeros(self.hs(0).shape, dtype=np.float64)
        self._hss = [hs.copy() for _ in self.hss]
        self._is_physicality_required = False

    def _generate_zero_obj(self):
        hs = np.zeros(self.hs(0).shape, dtype=np.float64)
        new_hss = [hs.copy() for _ in self.hss]
        return new_hss

    def _generate_origin_obj(self):
        hs = np.zeros(hs.shape(0), dtype=np.float64)
        hs[0][0] = 1 / len(self.hss)
        new_hss = [hs.copy() for _ in self.hss]
        return new_hss

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

    def get_basis(self) -> MatrixBasis:
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

    def convert_basis(self, other_basis: MatrixBasis) -> List[np.ndarray]:
        """returns HS representations for ``other_basis``.
        Parameters
        ----------
        other_basis : MatrixBasis
            basis.
        Returns
        -------
        List[np.ndarray]
            HS representations for ``other_basis``.
        """
        converted_hss = [
            gate.convert_hs(hs, self.composite_system.basis(), other_basis)
            for hs in self.hss
        ]
        return converted_hss

    def convert_to_comp_basis(self, mode: str = "row_major") -> List[np.ndarray]:
        """returns HS representations for computational basis.
        Parameters
        ----------
        mode : str, optional
            specify whether the order of basis is "row_major" or "column_major", by default "row_major".
        Returns
        -------
        List[np.ndarray]
            HS representations for computational basis.
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
