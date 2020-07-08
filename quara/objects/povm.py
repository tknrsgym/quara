import copy
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
from quara.objects.qoperation import QOperation


class Povm(QOperation):
    """
    Positive Operator-Valued Measure
    """

    def __init__(
        self,
        c_sys: CompositeSystem,
        vecs: List[np.ndarray],
        is_physicality_required: bool = True,
        is_estimation_object: bool = True,
        on_para_eq_constraint: bool = True,
        on_algo_eq_constraint: bool = True,
        on_algo_ineq_constraint: bool = True,
        eps_proj_physical: float = 10 ** (-4),
    ):
        """Constructor

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem of this povm.
        vecs : List[np.ndarray]
            list of vec of povm elements.
        is_physicality_required : bool, optional
            checks whether the POVM is physically wrong, by default True.
            all of the following conditions are ``True``, the POVM is physically correct:

            - It is a set of Hermitian matrices.
            - It is a set of positive semidefinite matrices.
            - The sum the elements of is the identity matrix.

            If you want to ignore the above requirements and create a POVM object, set ``is_physicality_required`` to ``False``.

        Raises
        ------
        ValueError
            entries of all vecs are not real numbers.
        ValueError
            If the dim in the ``c_sys`` does not match the dim in the ``vecs``
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
            eps_proj_physical=eps_proj_physical,
        )

        # Set
        self._vecs: Tuple[np.ndarray, ...] = tuple(copy.deepcopy(vecs))
        for b in self._vecs:
            b.setflags(write=False)

        self._composite_system: CompositeSystem = c_sys
        self._measurements = [len(self._vecs)]

        # Validation
        size = vecs[0].shape
        self._dim = int(np.sqrt(size[0]))
        size = [self._dim, self._dim]

        # whether entries of vec are real numbers
        for vec in self._vecs:
            if vec.dtype != np.float64:
                raise ValueError(
                    f"entries of all vecs must be real numbers. some dtype of vecs are {vec.dtype}"
                )

        # Whether dim of CompositeSystem equals dim of vec
        if c_sys.dim != self._dim:
            raise ValueError(
                f"dim of CompositeSystem must equal dim of vec. dim of CompositeSystem is {c_sys.dim}. dim of vec is {self._dim}"
            )

        # whether the POVM is physically correct
        if self.is_physicality_required and not self.is_physical():
            raise ValueError("the POVM is not physically correct.")

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
        """returns dim of Povm.

        Returns
        -------
        int
            dim of Povm.
        """
        return self._dim

    @property
    def measurements(self) -> List[int]:
        """Property to get numbers of measurements for each ElementalSystem.

        Returns
        -------
        List[int]
            numbers of measurements for each ElementalSystem.
        """
        return self._measurements

    def is_physical(self) -> bool:
        """returns whether the POVM is physically correct.

        all of the following conditions are ``True``, the POVM is physically correct:

        - It is a set of Hermitian matrices.
        - It is a set of positive semidefinite matrices.
        - The sum the elements of is the identity matrix.

        Returns
        -------
        bool
            whether the POVM is physically correct.
        """
        # in `is_positive_semidefinite` function, the state is checked whether it is Hermitian.
        # therefore, do not call the `is_hermitian` function explicitly.
        return self.is_positive_semidefinite() and self.is_identity()

    def set_zero(self):
        size = self.dim ** 2
        new_vecs = [np.zeros(size, dtype=np.float64) for _ in range(len(self.vecs))]
        self._vecs = new_vecs
        self._is_physicality_required = False

    def _generate_zero_obj(self):
        size = self.dim ** 2
        new_vecs = [np.zeros(size, dtype=np.float64) for _ in range(len(self.vecs))]
        return new_vecs

    def _generate_origin_obj(self):
        size = self.dim ** 2

        def generate_vec() -> np.array:
            return np.hstack(
                [
                    np.array([np.sqrt(self.dim) / len(self.vecs)], dtype=np.float64),
                    np.zeros(size - 1, dtype=np.float64),
                ]
            )

        new_vecs = [generate_vec() for _ in range(len(self.vecs))]

        return new_vecs

    def to_var(self) -> np.array:
        return convert_povm_to_var(
            c_sys=self._composite_system,
            vecs=list(self.vecs),
            on_para_eq_constraint=self.on_para_eq_constraint,
        )

    def to_stacked_vector(self) -> np.array:
        stacked_vecs = [vec.flatten() for vec in self.vecs]
        return stacked_vecs

    def calc_gradient(self):
        raise NotImplementedError()

    def calc_proj_eq_constraint(self):
        size = self.dim ** 2
        m = len(self.vecs)
        c = np.hstack(
            [
                np.array([np.sqrt(self.dim) / m], dtype=np.float64),
                np.zeros(size - 1, dtype=np.float64),
            ]
        )
        a_bar = np.sum(np.array(self.vecs), axis=0) / m

        new_vecs = []
        for vec in self.vecs:
            new_vec = vec - a_bar + c
            new_vecs.append(new_vec)
        new_povm = Povm(
            c_sys=self.composite_system, vecs=new_vecs, is_physicality_required=False
        )
        return new_povm

    def calc_proj_ineq_constraint(self) -> "Povm":
        size = (self._dim, self._dim)
        eigenvalues = []
        for vec in self.vecs:
            eigh, _ = np.linalg.eigh(vec.reshape(size))
            eigenvalues.append(eigh)
        diags = [np.diag(e) for e in eigenvalues]
        processed = []
        for diag in diags:
            diag[diag < 0] = 0
            processed.append(diag.flatten())

        # eigenvalues = self.calc_eigenvalues()
        # diags = [np.diag(e) for e in eigenvalues]
        # processed = []
        # for diag in diags:
        #     diag[diag < 0] = 0
        #     processed.append(diag.flatten())

        new_povm = Povm(
            c_sys=self.composite_system, vecs=processed, is_physicality_required=False
        )
        return new_povm

    def _generate_from_var_func(self):
        return convert_var_to_povm

    def calc_proj_physical(self):
        raise NotImplementedError()

    def calc_stopping_criterion_birgin_raydan_vectors(self):
        raise NotImplementedError()

    def is_satisfied_stopping_criterion_birgin_raydan_vectors(self):
        raise NotImplementedError()

    def is_satisfied_stopping_criterion_birgin_raydan_qoperations(self):
        raise NotImplementedError()

    # def _set_measurements(self, measurements: List[int]):
    #     self._measurements = measurements

    def get_measurement(self, index: Union[int, Tuple]) -> np.ndarray:
        """returns vec of measurement by index.

        Parameters
        ----------
        index : Union[int, Tuple]
            index of vec of measurement.
            if type is int, then regardes it as the index for CompositeSystem.
            if type is Tuple, then regardes it as the indices for earch ElementalSystems.
        Returns
        -------
        np.ndarray
            vec of measurement by index.

        Raises
        ------
        ValueError
            length of tuple does not equal length of the list of measurements.
        IndexError
            specified index does not exist in the list of measurements.
        """
        if type(index) == tuple:
            # whether size of tuple equals length of the list of measurements
            if len(index) != len(self._measurements):
                raise ValueError(
                    f"length of tuple must equal length of the list of measurements. length of tuple={len(index)}, length of the list of measurements={len(self._measurements)}"
                )

            # calculate index in _vecs by traversing the tuple from the back.
            # for example, if length of _measurements is 3 and each numbers are len1, len2, len3,
            # then index in _basis of tuple(x1, x2, x3) can be calculated the following expression:
            #   x1 * (len2 * len3) + x2 * len3 + x3
            temp_grobal_index = 0
            temp_len = 1
            for position, local_index in enumerate(reversed(index)):
                temp_grobal_index += local_index * temp_len
                temp_len = temp_len * (self._measurements[position])
            return self._vecs[temp_grobal_index]
        else:
            return self._vecs[index]

    def matrices(self) -> List[np.ndarray]:
        """returns matrices of measurements.

        Returns
        -------
        List[np.ndarray]
            matrices of measurements.
        """
        matrix_list = []
        size = (self.dim, self.dim)
        for v in self.vecs:
            matrix = np.zeros(size, dtype=np.complex128)
            for coefficient, basis in zip(v, self.composite_system.basis()):
                matrix += coefficient * basis
            matrix_list.append(matrix)
        return matrix_list

    def matrix(self, index: Union[int, Tuple]) -> np.ndarray:
        """returns matrix of measurement.

        Parameters
        ----------
        index : Union[int, Tuple]
            index of vec of measurement.
            if type is int, then regardes it as the index for CompositeSystem.
            if type is Tuple, then regardes it as the indices for earch ElementalSystems.
        Returns
        -------
        np.ndarray
            matrix of measurement.
        """
        vec = self.get_measurement(index)

        size = (self.dim, self.dim)
        matrix = np.zeros(size, dtype=np.complex128)
        for coefficient, basis in zip(vec, self.composite_system.basis()):
            matrix += coefficient * basis

        return matrix

    def is_hermitian(self) -> bool:
        """Returns whether the povm is a set of Hermit matrices.

        Returns
        -------
        bool
            If `True`, the povm is a set of Hermit matrices.
        """
        for m in self.matrices():
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

        for m in self.matrices():
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
        for m in self.matrices():
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
            v = self.matrices()[index]
            matrix = np.reshape(v, size)
            w = np.linalg.eigvals(matrix)
            return w
        else:
            w_list = []
            for v in self.matrices():
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

    def __getitem__(self, key) -> np.ndarray:
        # get vec with a serial number.
        return self._vecs[key]

    def _add_vec(self, other):
        if len(self.vecs) != len(other.vecs):
            # TODO: error_message
            raise ValueError()

        new_vecs = [s + o for s, o in zip(self, other)]
        return new_vecs

    def _sub_vec(self, other):
        if len(self.vecs) != len(other.vecs):
            # TODO: error_message
            raise ValueError()

        new_vecs = [s - o for s, o in zip(self, other)]
        return new_vecs

    def _mul_vec(self, other):
        new_vecs = [vec * other for vec in self.vecs]
        return new_vecs

    def _truediv_vec(self, other):
        new_vecs = [vec / other for vec in self.vecs]
        return new_vecs


def convert_var_index_to_povm_index(
    c_sys: CompositeSystem,
    vecs: List[np.ndarray],
    var_index: int,
    on_para_eq_constraint: bool = True,
) -> Tuple[int, int]:
    """converts variable index to povm index.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    vecs : List[np.ndarray]
        list of vec of povm elements.
    var_index : int
        variable index.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    Tuple[int, int]
        povm index.
        first value of tuple is an index of the number of measurements.
        second value of tuple is an index in specific measurement.
    """
    size = vecs[0].shape[0]
    (num_measurement, measurement_index) = divmod(var_index, size)
    return (num_measurement, measurement_index)


def convert_povm_index_to_var_index(
    c_sys: CompositeSystem,
    vecs: List[np.ndarray],
    povm_index: Tuple[int, int],
    on_para_eq_constraint: bool = True,
) -> int:
    """converts povm index to variable index.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    vecs : List[np.ndarray]
        list of vec of povm elements.
    povm_index : Tuple[int, int]
        povm index.
        first value of tuple is an index of the number of measurements.
        second value of tuple is an index in specific measurement.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    int
        variable index.
    """
    size = vecs[0].shape[0]
    (num_measurement, measurement_index) = povm_index
    var_index = size * num_measurement + measurement_index
    return var_index


def convert_var_to_povm(
    c_sys: CompositeSystem,
    var: np.ndarray,
    is_physicality_required: bool = True,
    is_estimation_object: bool = True,
    on_para_eq_constraint: bool = True,
    on_algo_eq_constraint: bool = True,
    on_algo_ineq_constraint: bool = True,
    eps_proj_physical: float = 10 ** (-4),
) -> Povm:
    """converts vec of variables to povm.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this povm.
    var : List[np.ndarray]
        list of vec of povm elements.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    Povm
        converted povm.
    """
    vecs = copy.copy(var)
    dim = int(np.sqrt(var.shape[0]))
    if on_para_eq_constraint:
        last_vec = np.eye(dim).flatten()
        var = vecs.reshape(dim - 1, dim * dim)
        for vec in var:
            last_vec -= vec.flatten()
        vecs = np.hstack([vecs, last_vec])

    vec_list = []
    reshaped_vecs = vecs.reshape(dim, dim ** dim)
    for vec in reshaped_vecs:
        vec_list.append(vec)
    povm = Povm(
        c_sys,
        vec_list,
        is_physicality_required=is_physicality_required,
        is_estimation_object=is_estimation_object,
        on_para_eq_constraint=on_para_eq_constraint,
        on_algo_eq_constraint=on_algo_eq_constraint,
        on_algo_ineq_constraint=on_algo_ineq_constraint,
        eps_proj_physical=eps_proj_physical,
    )
    return povm


def convert_povm_to_var(
    c_sys: CompositeSystem, vecs: List[np.ndarray], on_para_eq_constraint: bool = True
) -> np.array:
    """converts hs of povm to vec of variables.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this state.
    vecs : List[np.ndarray]
        list of vec of povm elements.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    np.array
        list of vec of variables.
    """
    var = copy.copy(vecs)
    if on_para_eq_constraint:
        del var[-1]
    var = np.hstack(var)
    return var


def calc_gradient_from_povm(
    c_sys: CompositeSystem,
    vecs: List[np.ndarray],
    var_index: int,
    on_para_eq_constraint: bool = True,
) -> Povm:
    """calculates gradient from gate.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of this gate.
    vecs : List[np.ndarray]
        list of vec of povm elements.
    var_index : int
        variable index.
    on_para_eq_constraint : bool, optional
        uses equal constraints, by default True.

    Returns
    -------
    Povm
        Povm with gradient as vecs.
    """
    gradient = []
    for _ in vecs:
        gradient.append(np.zeros(c_sys.dim ** 2, dtype=np.float64))

    (num_measurement, measurement_index) = convert_var_index_to_povm_index(
        c_sys, vecs, var_index, on_para_eq_constraint
    )
    gradient[num_measurement][measurement_index] = 1

    povm = Povm(c_sys, gradient, is_physicality_required=False)
    return povm


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
        convert_vec(vec, get_normalized_pauli_basis(), c_sys.basis()).real.astype(
            np.float64
        )
        for vec in vecs
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
        convert_vec(
            vec, get_normalized_pauli_basis(n_qubit=2), c_sys.basis()
        ).real.astype(np.float64)
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
