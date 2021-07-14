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


class MProcess(QOperation):
    def __init__(
        self,
        c_sys: CompositeSystem,
        hss: np.ndarray,
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
        self._hss: np.ndarray = hss

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

        # whether the gate is physically correct
        if self.is_physicality_required and not self.is_physical():
            raise ValueError("the gate is not physically correct.")

    def _info(self):
        info = {}
        info["Type"] = self.__class__.__name__
        info["Dim"] = self.dim
        info["HSs"] = self.hss
        return info

    def is_physical(self):
        # TODO implement
        return True

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
    def hss(self):
        """returns HS representation of gate.

        Returns
        -------
        np.ndarray
            HS representation of gate.
        """
        return self._hss
