import itertools
from typing import List

import numpy as np

from quara.objects.composite_system import CompositeSystem
from quara.objects.matrix_basis import MatrixBasis, get_normalized_pauli_basis
import quara.utils.matrix_util as mutil


class Gate:
    def __init__(self, c_sys: CompositeSystem, HS: np.ndarray):
        self._composite_system: CompositeSystem = c_sys
        self._HS: np.ndarray = HS
        # TODO check HS is square matrix
        # TODO check dim of HS is square number
        # TODO check dtype=np.float64
        # TODO check dim of HS equals dim of compsite system

    @property
    def HS(self):
        """returns HS representation of gate.
        
        Returns
        -------
        np.array
            HS representation of gate
        """
        return self._HS

    def is_tp(self):

        # if A:HS representation of gate, then A:TP <=> Tr[A(B_\alpha)] = Tr[B_\alpha] for all basis.
        pass


def convert_HS(
    from_HS: np.array, from_basis: MatrixBasis, to_basis: MatrixBasis
) -> np.array:
    # TODO implement

    # TODO parameter check
    representation_matrix = [
        mutil.inner_product(
            B_alpha.reshape(1, -1)[0], from_HS @ B_beta.reshape(1, -1)[0]
        )
        for B_alpha, B_beta in itertools.product(from_basis, to_basis)
    ]
    HS = np.array(representation_matrix).reshape(
        from_basis.dim ** 2, from_basis.dim ** 2
    )
    return HS


def get_HS(matrix: np.array, basis: MatrixBasis) -> np.array:
    # TODO parameter check

    # HS_{\alpha\beta} = <<B_{\alpha}| A(B_{\beta})>>
    representation_matrix = [
        mutil.inner_product(
            B_alpha.reshape(1, -1)[0], matrix @ B_beta.reshape(1, -1)[0]
        )
        for B_alpha, B_beta in itertools.product(basis, basis)
    ]
    HS = np.array(representation_matrix).reshape(basis.dim ** 2, basis.dim ** 2)
    return HS


def get_X(c_sys: CompositeSystem) -> Gate:
    # TODO check dim of CompositeSystem = 2
    # convert "HS representation in Pauli basis" to "HS representation in basis of CompositeSystem"
    matrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    HS = convert_HS(matrix, get_normalized_pauli_basis(), c_sys.basis)
    gate = Gate(c_sys, HS)
    return gate


def get_Y(c_sys: CompositeSystem) -> Gate:
    # TODO check dim of CompositeSystem = 2
    # convert "HS representation in Pauli basis" to "HS representation in basis of CompositeSystem"
    matrix = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.float64
    )
    HS = convert_HS(matrix, get_normalized_pauli_basis(), c_sys.basis)
    gate = Gate(c_sys, HS)
    return gate


def get_Z(c_sys: CompositeSystem) -> Gate:
    # TODO check dim of CompositeSystem = 2
    # convert "HS representation in Pauli basis" to "HS representation in basis of CompositeSystem"
    matrix = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    HS = convert_HS(matrix, get_normalized_pauli_basis(), c_sys.basis)
    gate = Gate(c_sys, HS)
    return gate
