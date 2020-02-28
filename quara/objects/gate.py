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
        # TODO check HS is square matrix
        # TODO check dim of HS is square number
        # TODO check dtype=np.float64
        # TODO check dim of HS equals dim of compsite system

    @property
    def hs(self):
        """returns HS representation of gate.
        
        Returns
        -------
        np.array
            HS representation of gate
        """
        return self._hs

    def is_tp(self):

        # if A:HS representation of gate, then A:TP <=> Tr[A(B_\alpha)] = Tr[B_\alpha] for all basis.
        pass

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
        converted_hs = convert_hs(self.hs, self._composite_system.basis, other_basis)
        return converted_hs

    def convert_to_comp_basis(self) -> np.array:
        """returns HS representation for computational basis.
        
        Returns
        -------
        np.array
            HS representation for computational basis
        """
        converted_hs = convert_hs(
            self.hs, self._composite_system.basis, get_comp_basis()
        )
        return converted_hs


def convert_hs(
    from_hs: np.array, from_basis: MatrixBasis, to_basis: MatrixBasis
) -> np.array:
    # TODO parameter check
    # U_{\alpha,\bata} := Tr[to_basis_{\alpha}^{\dagger} @ from_basis_{\beta}]
    trans_matrix = [
        mutil.inner_product(B_alpha.reshape(1, -1)[0], B_beta.reshape(1, -1)[0])
        for B_alpha, B_beta in itertools.product(to_basis, from_basis)
    ]
    U = np.array(trans_matrix).reshape(from_basis.dim ** 2, from_basis.dim ** 2)
    to_hs = U @ from_hs @ U.conj().T
    return to_hs


def _get_1q_gate_from_hs_on_pauli_basis(
    matrix: np.array, c_sys: CompositeSystem
) -> Gate:
    # TODO check dim of CompositeSystem = 2
    # convert "HS representation in Pauli basis" to "HS representation in basis of CompositeSystem"
    hs = convert_hs(matrix, get_normalized_pauli_basis(), c_sys.basis)
    gate = Gate(c_sys, hs)
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
