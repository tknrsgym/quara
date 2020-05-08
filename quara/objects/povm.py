import itertools
from typing import List, Tuple, Union

import numpy as np

import quara.utils.matrix_util as mutil
from quara.objects.composite_system import CompositeSystem
from quara.objects.matrix_basis import (
    MatrixBasis,
    convert_vec,
    get_normalized_pauli_basis,
)
from quara.settings import Settings


class Povm:
    """
    Positive Operator-Valued Measure
    """

    def __init__(
        self, c_sys: CompositeSystem, vecs: List[np.ndarray], is_physical: bool = True
    ):
        """Constructor

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem of this povm.
        vecs : List[np.ndarray]

        is_physical : bool, optional
            Check whether the povm is physically correct, by default True.
            If ``True``, the following requirements are met.

            - It is a set of Hermitian matrices.
            - The sum is the identity matrix.
            - positive semidefinite.

            If you want to ignore the above requirements and create a POVM object, set ``is_physical`` to ``False``.

        Raises
        ------
        ValueError
            If ``is_physical`` is ``True`` and it is not a set of Hermitian matrices
        ValueError
            If ``is_physical`` is ``True`` and the sum is not an identity matrix
        ValueError
            If ``is_physical`` is ``True`` and is not a positive semidefinite
        ValueError
            If the dim in the ``c_sys`` does not match the dim in the ``vecs``
        """
        # Set
        # TODO: consider make it tuple of np.ndarray
        self._vecs: List[np.ndarray] = vecs
        self._composite_system: CompositeSystem = c_sys

        # 観測されうる測定値の集合
        m_length = len(bin(len(vecs) - 1).replace("0b", ""))
        m_format = "0" + str(m_length) + "b"
        measurements = [format(i, m_format) for i in range(len(vecs))]
        self._measurements: Tuple = tuple(measurements)

        # TODO: 今のところ未使用。なくても済むなら削除する
        self._measurements_map: dict = {
            i: format(i, m_format) for i in range(len(vecs))
        }

        self._is_physical = is_physical

        # Validation
        ## Validate whether `vecs` is a set of Hermitian matrices
        # TODO: Consider using VectorizedMatrixBasis
        size = vecs[0].shape
        self._dim = int(np.sqrt(size[0]))
        size = [self._dim, self._dim]

        if is_physical:
            # Validate to meet requirements as Povm
            if not self.is_hermitian():
                raise ValueError("POVM must be a set of Hermitian matrices")

            if not self.is_identity():
                # whether the sum of the elements is an identity matrix or not
                raise ValueError(
                    "The sum of the elements of POVM must be an identity matrix."
                )

            if not self.is_positive_semidefinite():
                raise ValueError("Eigenvalues of POVM elements must be non-negative.")

        # Whether dim of CompositeSystem equals dim of vec
        if c_sys.dim != self._dim:
            raise ValueError(
                f"dim of CompositeSystem must equal dim of vec. dim of CompositeSystem is {c_sys.dim}. dim of vec is {self._dim}"
            )

    def measurement(self, key: str) -> np.ndarray:
        # |0> -> 0
        # |1> -> 1
        # |00> -> 0
        # |01> -> 1
        # |10> -> 2
        # |11> -> 3
        if key not in self._measurements:
            raise ValueError(
                "That measurement does not exist. See the list of measurements by 'measurement' property."
            )

        return self._vecs[int(key, 2)]

    def __getitem__(self, key) -> np.ndarray:
        # 通し番号でvecsをとる
        return self._vecs[key]

    # TODO: 他に良い名前があったらそちらに変える
    def matrixes(self) -> List[np.ndarray]:
        matrix_list = []
        size = (self.dim, self.dim)
        for v in self.vecs:
            matrix = np.zeros(size, dtype=np.complex128)
            for coefficient, basis in zip(v, self.composite_system.basis()):
                matrix += coefficient * basis
            matrix_list.append(matrix)
        return matrix_list

    def matrix(self, key: Union[int, str]) -> np.ndarray:
        if type(key) == int:
            # 通し番号
            vec = self.vecs[key]
        elif type(key) == str:
            # '00', '01', '10', '11'など
            vec = self.measurement(key)
        else:
            # TODO: message
            raise ValueError("")

        size = (self.dim, self.dim)
        matrix = np.zeros(size, dtype=np.complex128)
        for coefficient, basis in zip(vec, self.composite_system.basis()):
            matrix += coefficient * basis

        return matrix

    @property
    def measurements(self) -> List[str]:
        return list(self._measurements)

    @property
    def vecs(self) -> List[np.ndarray]:  # read only
        """Property to get vecs of povm.

        Returns
        -------
        List[np.ndarray]
            vecs of povm.
        """
        return self._vecs

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def composite_system(self) -> CompositeSystem:
        """Property to get composite system.

        Returns
        -------
        CompositeSystem
            composite system.
        """
        return self._composite_system

    @property
    def is_physical(self) -> bool:  # read only
        return self._is_physical

    def e_sys_dims(self) -> List[int]:
        # vecs_size = [len(vec) for vec in self._vecs]
        e_sys_dims = [e_sys.dim ** 2 for e_sys in self._composite_system]
        return e_sys_dims

    def is_hermitian(self) -> bool:
        for m in self.matrixes():
            if not mutil.is_hermitian(m):
                return False
        return True

    def is_positive_semidefinite(self, atol: float = None) -> bool:
        """Returns whether each element is positive semidifinite.

        Returns
        -------
        bool
            True where each element is positive semidifinite, False otherwise.
        """
        atol = atol if atol else Settings.get_atol()

        size = [self.dim, self.dim]
        for m in self.matrixes():
            if not mutil.is_positive_semidefinite(m, atol):
                return False

        return True

    def is_identity(self) -> bool:
        """Returns whether the sum of the elements ``_vecs`` is an identity matrix.

        Returns
        -------
        bool
            If the sum of the elements ``_vecs`` is an identity matrix,
            otherwise it returns False.
        """
        sum_matrix = self._sum_matrix()
        identity = np.identity(self.dim, dtype=np.complex128)
        return np.allclose(sum_matrix, identity)

    def _sum_matrix(self):
        size = [self.dim, self.dim]
        sum_matrix = np.zeros(size, dtype=np.complex128)
        for m in self.matrixes():
            sum_matrix += np.reshape(m, size)

        return sum_matrix

    def calc_eigenvalues(
        self, index: int = None
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Calculates eigenvalues.

        Parameters
        ----------
        index : int, optional
            Index to obtain eigenvalues, by default None

        Returns
        -------
        Union[List[np.ndarray], np.ndarray]
            eigenvalues.
        """

        size = [self._dim, self._dim]
        if index is not None:
            v = self.matrixes()[index]
            matrix = np.reshape(v, size)
            w = np.linalg.eigvals(matrix)
            return w
        else:
            w_list = []
            for v in self.matrixes():
                matrix = np.reshape(v, size)
                w = np.linalg.eigvals(matrix)
                w_list.append(w)
            return w_list

    def convert_basis(self, other_basis: MatrixBasis) -> List[np.array]:
        """Calculate vector representation for ``other_basis``.

        Parameters
        ----------
        other_basis : MatrixBasis
            basis

        Returns
        -------
        List[np.array]
            Vector representation after conversion to ``other_basis`` .
        """

        converted_vecs = []
        for vec in self.vecs:
            converted_vecs.append(
                convert_vec(vec, self._composite_system.basis(), other_basis)
            )
        return converted_vecs


def _get_1q_povm_from_vecs_on_pauli_basis(
    c_sys: CompositeSystem, vecs: np.array
) -> Povm:
    # whether CompositeSystem is 1 qubit
    size = len(c_sys._elemental_systems)
    if size != 1:
        raise ValueError(f"CompositeSystem must be 1 qubit. it is {size} qubits")

    # whether dim of CompositeSystem equals 2
    if c_sys.dim != 2:
        raise ValueError(
            f"dim of CompositeSystem must equals 2.  dim of CompositeSystem is {c_sys.dim}"
        )

    # convert "vecs in Pauli basis" to "vecs in basis of CompositeSystem"
    to_vecs = [
        convert_vec(vec, get_normalized_pauli_basis(), c_sys.basis()) for vec in vecs
    ]
    povm = Povm(c_sys, to_vecs)
    return povm


def _get_x_measurement_vecs() -> List[np.array]:
    vecs = [
        1 / np.sqrt(2) * np.array([1, 1, 0, 0], dtype=np.float64),
        1 / np.sqrt(2) * np.array([1, -1, 0, 0], dtype=np.float64),
    ]
    return vecs


def _get_y_measurement_vecs() -> List[np.array]:
    vecs = [
        1 / np.sqrt(2) * np.array([1, 0, 1, 0], dtype=np.float64),
        1 / np.sqrt(2) * np.array([1, 0, -1, 0], dtype=np.float64),
    ]
    return vecs


def _get_z_measurement_vecs() -> List[np.array]:
    vecs = [
        1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.float64),
        1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.float64),
    ]
    return vecs


def get_x_measurement(c_sys: CompositeSystem) -> Povm:
    """returns POVM of X measurement.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing POVM.

    Returns
    -------
    Povm
        X measurement.

    Raises
    ------
    ValueError
        CompositeSystem is not 1quit.
    ValueError
        dim of CompositeSystem does not equal 2
    """
    povm = _get_1q_povm_from_vecs_on_pauli_basis(c_sys, _get_x_measurement_vecs())
    return povm


def get_y_measurement(c_sys: CompositeSystem) -> Povm:
    """returns POVM of Y measurement.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing POVM.

    Returns
    -------
    Povm
        Y measurement.

    Raises
    ------
    ValueError
        CompositeSystem is not 1quit.
    ValueError
        dim of CompositeSystem does not equal 2
    """
    povm = _get_1q_povm_from_vecs_on_pauli_basis(c_sys, _get_y_measurement_vecs())
    return povm


def get_z_measurement(c_sys: CompositeSystem) -> Povm:
    """returns POVM of Z measurement.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing POVM.

    Returns
    -------
    Povm
        Z measurement.

    Raises
    ------
    ValueError
        CompositeSystem is not 1quit.
    ValueError
        dim of CompositeSystem does not equal 2
    """
    povm = _get_1q_povm_from_vecs_on_pauli_basis(c_sys, _get_z_measurement_vecs())
    return povm


def _get_2q_povm_from_vecs_on_pauli_basis(
    c_sys: CompositeSystem, vecs1: np.array, vecs2: np.array
) -> Povm:
    # whether CompositeSystem is 2 qubit
    size = len(c_sys._elemental_systems)
    if size != 2:
        raise ValueError(f"CompositeSystem must be 2 qubit. it is {size} qubits")

    # whether dim of CompositeSystem equals 4
    if c_sys.dim != 4:
        raise ValueError(
            f"dim of CompositeSystem must equals 4.  dim of CompositeSystem is {c_sys.dim}"
        )

    # calculate tensor products of vecs
    vecs = [np.kron(val1, val2) for val1, val2 in itertools.product(vecs1, vecs2)]

    # convert "vecs in Pauli basis" to "vecs in basis of CompositeSystem"
    to_vecs = [
        convert_vec(vec, get_normalized_pauli_basis(n_qubit=2), c_sys.basis())
        for vec in vecs
    ]
    povm = Povm(c_sys, to_vecs)
    return povm


def get_xx_measurement(c_sys: CompositeSystem) -> Povm:
    """returns POVM of XX measurement.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing POVM.

    Returns
    -------
    Povm
        XX measurement.

    Raises
    ------
    ValueError
        CompositeSystem is not 2quit.
    ValueError
        dim of CompositeSystem does not equal 4
    """
    povm = _get_2q_povm_from_vecs_on_pauli_basis(
        c_sys, _get_x_measurement_vecs(), _get_x_measurement_vecs()
    )
    return povm


def get_xy_measurement(c_sys: CompositeSystem) -> Povm:
    """returns POVM of XY measurement.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing POVM.

    Returns
    -------
    Povm
        XY measurement.

    Raises
    ------
    ValueError
        CompositeSystem is not 2quit.
    ValueError
        dim of CompositeSystem does not equal 4
    """
    povm = _get_2q_povm_from_vecs_on_pauli_basis(
        c_sys, _get_x_measurement_vecs(), _get_y_measurement_vecs()
    )
    return povm


def get_xz_measurement(c_sys: CompositeSystem) -> Povm:
    """returns POVM of XZ measurement.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing POVM.

    Returns
    -------
    Povm
        XZ measurement.

    Raises
    ------
    ValueError
        CompositeSystem is not 2quit.
    ValueError
        dim of CompositeSystem does not equal 4
    """
    povm = _get_2q_povm_from_vecs_on_pauli_basis(
        c_sys, _get_x_measurement_vecs(), _get_z_measurement_vecs()
    )
    return povm


def get_yx_measurement(c_sys: CompositeSystem) -> Povm:
    """returns POVM of YX measurement.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing POVM.

    Returns
    -------
    Povm
        YX measurement.

    Raises
    ------
    ValueError
        CompositeSystem is not 2quit.
    ValueError
        dim of CompositeSystem does not equal 4
    """
    povm = _get_2q_povm_from_vecs_on_pauli_basis(
        c_sys, _get_y_measurement_vecs(), _get_x_measurement_vecs()
    )
    return povm


def get_yy_measurement(c_sys: CompositeSystem) -> Povm:
    """returns POVM of YY measurement.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing POVM.

    Returns
    -------
    Povm
        YY measurement.

    Raises
    ------
    ValueError
        CompositeSystem is not 2quit.
    ValueError
        dim of CompositeSystem does not equal 4
    """
    povm = _get_2q_povm_from_vecs_on_pauli_basis(
        c_sys, _get_y_measurement_vecs(), _get_y_measurement_vecs()
    )
    return povm


def get_yz_measurement(c_sys: CompositeSystem) -> Povm:
    """returns POVM of YZ measurement.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing POVM.

    Returns
    -------
    Povm
        YZ measurement.

    Raises
    ------
    ValueError
        CompositeSystem is not 2quit.
    ValueError
        dim of CompositeSystem does not equal 4
    """
    povm = _get_2q_povm_from_vecs_on_pauli_basis(
        c_sys, _get_y_measurement_vecs(), _get_z_measurement_vecs()
    )
    return povm


def get_zx_measurement(c_sys: CompositeSystem) -> Povm:
    """returns POVM of ZX measurement.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing POVM.

    Returns
    -------
    Povm
        ZX measurement.

    Raises
    ------
    ValueError
        CompositeSystem is not 2quit.
    ValueError
        dim of CompositeSystem does not equal 4
    """
    povm = _get_2q_povm_from_vecs_on_pauli_basis(
        c_sys, _get_z_measurement_vecs(), _get_x_measurement_vecs()
    )
    return povm


def get_zy_measurement(c_sys: CompositeSystem) -> Povm:
    """returns POVM of ZY measurement.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing POVM.

    Returns
    -------
    Povm
        ZY measurement.

    Raises
    ------
    ValueError
        CompositeSystem is not 2quit.
    ValueError
        dim of CompositeSystem does not equal 4
    """
    povm = _get_2q_povm_from_vecs_on_pauli_basis(
        c_sys, _get_z_measurement_vecs(), _get_y_measurement_vecs()
    )
    return povm


def get_zz_measurement(c_sys: CompositeSystem) -> Povm:
    """returns POVM of ZZ measurement.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem containing POVM.

    Returns
    -------
    Povm
        ZZ measurement.

    Raises
    ------
    ValueError
        CompositeSystem is not 2quit.
    ValueError
        dim of CompositeSystem does not equal 4
    """
    povm = _get_2q_povm_from_vecs_on_pauli_basis(
        c_sys, _get_z_measurement_vecs(), _get_z_measurement_vecs()
    )
    return povm
