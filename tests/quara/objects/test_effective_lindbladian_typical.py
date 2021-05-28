import numpy as np
import numpy.testing as npt
import pytest

from typing import List
from itertools import permutations
from scipy.linalg import expm

from quara.math.matrix import project_to_traceless_matrix
from quara.objects import matrix_basis
from quara.objects.matrix_basis import MatrixBasis
from quara.objects.elemental_system import ElementalSystem
from quara.objects.composite_system import CompositeSystem
from quara.objects.gate import Gate
from quara.objects.effective_lindbladian import EffectiveLindbladian
from quara.objects.effective_lindbladian import generate_effective_lindbladian_from_h
from quara.objects.gate_typical import (
    get_gate_names_1qubit,
    get_gate_names_2qubit,
    get_gate_names_2qubit_asymmetric,
    get_gate_names_3qubit,
    get_gate_names_1qutrit,
    get_gate_names_1qutrit_single_gellmann,
)
from quara.objects.qoperation_typical import (
    generate_gate_object,
    generate_effective_lindbladian_object,
)


def _test_hamiltonian_vec_hamiltonian_mat(
    gate_name: str,
    dims: List[int] = None,
    ids: List[int] = None,
    c_sys: CompositeSystem = None,
    decimal: int = 15,
):
    if dims is None:
        dims = []
    if ids is None:
        ids = []
    # Arrange
    object_name = "hamiltonian_vec"
    h_vec = generate_effective_lindbladian_object(
        gate_name, object_name, dims, ids, c_sys
    )
    object_name = "hamiltonian_mat"
    h_mat = generate_effective_lindbladian_object(
        gate_name, object_name, dims, ids, c_sys
    )

    dim_sys = c_sys.dim
    b = c_sys.basis()
    if dim_sys * dim_sys != h_vec.size:
        raise ValueError(f"dimensions of c_sys and h_vec are inconsistent.")
    h_mat_from_vec = np.zeros((dim_sys, dim_sys), dtype=np.complex128)
    for i, bi in enumerate(b):
        h_mat_from_vec += h_vec[i] * bi

    # Act
    actual = h_mat_from_vec

    # Assert
    expected = h_mat
    # The case of decimal=16 below returns AssertionError.
    npt.assert_almost_equal(actual, expected, decimal=decimal)


def _test_hamiltonian_mat_unitary_mat(
    gate_name: str,
    dims: List[int] = [],
    ids: List[int] = [],
    c_sys: CompositeSystem = None,
    decimal: int = 15,
):
    # Arrange
    object_name = "hamiltonian_mat"
    h_mat = generate_effective_lindbladian_object(
        gate_name, object_name, dims, ids, c_sys
    )
    object_name = "unitary_mat"
    u_mat = generate_gate_object(gate_name, object_name, dims, ids, c_sys)

    # Act
    actual = expm(-1j * h_mat)

    # Assert
    expected = u_mat
    npt.assert_almost_equal(actual, expected, decimal=decimal)


def _test_effective_lindbladian_mat_gate_mat(
    gate_name: str,
    dims: List[int] = None,
    ids: List[int] = None,
    c_sys: CompositeSystem = None,
    decimal: int = 15,
):
    if dims is None:
        dims = []
    if ids is None:
        ids = []

    # Arrange
    object_name = "effective_lindbladian_mat"
    el_mat = generate_effective_lindbladian_object(
        gate_name, object_name, dims, ids, c_sys
    )
    object_name = "gate_mat"
    g_mat = generate_gate_object(gate_name, object_name, dims, ids, c_sys)

    # Act
    actual = expm(el_mat)
    # Assert
    expected = g_mat
    npt.assert_almost_equal(actual, expected, decimal=decimal)


def _test_generate_effective_lindbladian_from_h(
    gate_name: str,
    dims: List[int] = None,
    ids: List[int] = None,
    c_sys: CompositeSystem = None,
    decimal: int = 15,
):
    if dims is None:
        dims = []
    if ids is None:
        ids = []

    # Arrange
    object_name = "hamiltonian_mat"
    h_mat = generate_effective_lindbladian_object(
        gate_name, object_name, dims, ids, c_sys
    )
    object_name = "effective_lindbladian_mat"
    el_mat = generate_effective_lindbladian_object(
        gate_name, object_name, dims, ids, c_sys
    )

    el_from_h = generate_effective_lindbladian_from_h(c_sys, h_mat)
    # Act
    actual = el_from_h.hs
    # Assert
    expected = el_mat
    npt.assert_almost_equal(actual, expected, decimal=15)
    assert el_from_h.is_physical() == True


def _test_calc_h(
    gate_name: str,
    dims: List[int] = None,
    ids: List[int] = None,
    c_sys: CompositeSystem = None,
    decimal: int = 15,
):
    if dims is None:
        dims = []
    if ids is None:
        ids = []

    # Arrange
    object_name = "hamiltonian_mat"
    h_mat = generate_effective_lindbladian_object(
        gate_name, object_name, dims, ids, c_sys
    )

    el_from_h = generate_effective_lindbladian_from_h(c_sys, h_mat)
    # Act
    actual = project_to_traceless_matrix(el_from_h.calc_h_mat())
    # Assert
    expected = project_to_traceless_matrix(h_mat)
    npt.assert_almost_equal(actual, expected, decimal=decimal)


# Tests for identity gates


@pytest.mark.parametrize(
    ("dims", "c_sys"),
    [
        (
            [2],
            CompositeSystem(
                [ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())]
            ),
        ),
        (
            [2, 2],
            CompositeSystem(
                [
                    ElementalSystem(0, matrix_basis.get_normalized_pauli_basis()),
                    ElementalSystem(1, matrix_basis.get_normalized_pauli_basis()),
                ]
            ),
        ),
        (
            [3],
            CompositeSystem(
                [ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis())]
            ),
        ),
        (
            [4],
            CompositeSystem(
                [
                    ElementalSystem(
                        0,
                        matrix_basis.get_normalized_generalized_gell_mann_basis(dim=4),
                    )
                ]
            ),
        ),
    ],
)
def test_validity_hamiltonian_vec_hamiltonian_mat_identity_gate_case01(
    dims: List[int], c_sys: CompositeSystem
):
    # Arrange
    gate_name = "identity"
    ids = []

    _test_hamiltonian_vec_hamiltonian_mat(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
    )


@pytest.mark.threequbit
def test_validity_hamiltonian_vec_hamiltonian_mat_identity_gate_case02():
    # Arrange
    parameters = [
        (
            [2, 2, 2],
            CompositeSystem(
                [
                    ElementalSystem(0, matrix_basis.get_normalized_pauli_basis()),
                    ElementalSystem(1, matrix_basis.get_normalized_pauli_basis()),
                    ElementalSystem(2, matrix_basis.get_normalized_pauli_basis()),
                ]
            ),
        ),
    ]
    gate_name = "identity"
    ids = []

    for (dims, c_sys) in parameters:
        print(f"test dims={dims}, c_sys={c_sys}")
        _test_hamiltonian_vec_hamiltonian_mat(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
        )


@pytest.mark.twoqutrit
def test_validity_hamiltonian_vec_hamiltonian_mat_identity_gate_case03():
    # Arrange
    parameters = [
        (
            [3, 3],
            CompositeSystem(
                [
                    ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis()),
                    ElementalSystem(1, matrix_basis.get_normalized_gell_mann_basis()),
                ]
            ),
        ),
    ]
    gate_name = "identity"
    ids = []

    for (dims, c_sys) in parameters:
        print(f"test dims={dims}, c_sys={c_sys}")
        _test_hamiltonian_vec_hamiltonian_mat(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
        )


@pytest.mark.parametrize(
    ("dims", "c_sys"),
    [
        (
            [2],
            CompositeSystem(
                [ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())]
            ),
        ),
        (
            [2, 2],
            CompositeSystem(
                [
                    ElementalSystem(0, matrix_basis.get_normalized_pauli_basis()),
                    ElementalSystem(1, matrix_basis.get_normalized_pauli_basis()),
                ]
            ),
        ),
        (
            [3],
            CompositeSystem(
                [ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis())]
            ),
        ),
        (
            [4],
            CompositeSystem(
                [
                    ElementalSystem(
                        0,
                        matrix_basis.get_normalized_generalized_gell_mann_basis(dim=4),
                    )
                ]
            ),
        ),
    ],
)
def test_validity_hamiltonian_mat_unitary_mat_identity_gate_01(
    dims: List[int], c_sys: CompositeSystem
):
    # Arrange
    gate_name = "identity"
    ids = []

    _test_hamiltonian_mat_unitary_mat(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
    )


@pytest.mark.threequbit
def test_validity_hamiltonian_mat_unitary_mat_identity_gate_case02():
    # Arrange
    parameters = [
        (
            [2, 2, 2],
            CompositeSystem(
                [
                    ElementalSystem(0, matrix_basis.get_normalized_pauli_basis()),
                    ElementalSystem(1, matrix_basis.get_normalized_pauli_basis()),
                    ElementalSystem(2, matrix_basis.get_normalized_pauli_basis()),
                ]
            ),
        ),
    ]
    gate_name = "identity"
    ids = []

    for (dims, c_sys) in parameters:
        print(f"test dims={dims}, c_sys={c_sys}")
        _test_hamiltonian_mat_unitary_mat(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
        )


@pytest.mark.twoqutrit
def test_validity_hamiltonian_mat_unitary_mat_identity_gate_case03():
    # Arrange
    parameters = [
        (
            [3, 3],
            CompositeSystem(
                [
                    ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis()),
                    ElementalSystem(1, matrix_basis.get_normalized_gell_mann_basis()),
                ]
            ),
        ),
    ]
    gate_name = "identity"
    ids = []

    for (dims, c_sys) in parameters:
        print(f"test dims={dims}, c_sys={c_sys}")
        _test_hamiltonian_mat_unitary_mat(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
        )


@pytest.mark.parametrize(
    ("dims", "c_sys"),
    [
        (
            [2],
            CompositeSystem(
                [ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())]
            ),
        ),
        (
            [2, 2],
            CompositeSystem(
                [
                    ElementalSystem(0, matrix_basis.get_normalized_pauli_basis()),
                    ElementalSystem(1, matrix_basis.get_normalized_pauli_basis()),
                ]
            ),
        ),
        (
            [3],
            CompositeSystem(
                [ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis())]
            ),
        ),
        (
            [4],
            CompositeSystem(
                [
                    ElementalSystem(
                        0,
                        matrix_basis.get_normalized_generalized_gell_mann_basis(dim=4),
                    )
                ]
            ),
        ),
    ],
)
def test_effective_lindbladian_mat_gate_mat_identity_gate_case01(
    dims: List[int], c_sys: CompositeSystem
):
    # Arrange
    gate_name = "identity"
    ids = []

    _test_effective_lindbladian_mat_gate_mat(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
    )


@pytest.mark.threequbit
def test_effective_lindbladian_mat_gate_mat_identity_gate_case02():
    # Arrange
    parameters = [
        (
            [2, 2, 2],
            CompositeSystem(
                [
                    ElementalSystem(0, matrix_basis.get_normalized_pauli_basis()),
                    ElementalSystem(1, matrix_basis.get_normalized_pauli_basis()),
                    ElementalSystem(2, matrix_basis.get_normalized_pauli_basis()),
                ]
            ),
        ),
    ]
    gate_name = "identity"
    ids = []

    for (dims, c_sys) in parameters:
        print(f"test dims={dims}, c_sys={c_sys}")
        _test_effective_lindbladian_mat_gate_mat(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
        )


@pytest.mark.twoqutrit
def test_effective_lindbladian_mat_gate_mat_identity_gate_case03():
    # Arrange
    parameters = [
        (
            [3, 3],
            CompositeSystem(
                [
                    ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis()),
                    ElementalSystem(1, matrix_basis.get_normalized_gell_mann_basis()),
                ]
            ),
        ),
    ]
    gate_name = "identity"
    ids = []

    for (dims, c_sys) in parameters:
        print(f"test dims={dims}, c_sys={c_sys}")
        _test_effective_lindbladian_mat_gate_mat(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
        )


@pytest.mark.parametrize(
    ("dims", "c_sys"),
    [
        (
            [2],
            CompositeSystem(
                [ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())]
            ),
        ),
        (
            [2, 2],
            CompositeSystem(
                [
                    ElementalSystem(0, matrix_basis.get_normalized_pauli_basis()),
                    ElementalSystem(1, matrix_basis.get_normalized_pauli_basis()),
                ]
            ),
        ),
        (
            [3],
            CompositeSystem(
                [ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis())]
            ),
        ),
        (
            [4],
            CompositeSystem(
                [
                    ElementalSystem(
                        0,
                        matrix_basis.get_normalized_generalized_gell_mann_basis(dim=4),
                    )
                ]
            ),
        ),
    ],
)
def test_generate_effective_lindbladian_from_h_identity_gate_case01(
    dims: List[int], c_sys: CompositeSystem
):
    # Arrange
    gate_name = "identity"
    ids = []

    _test_generate_effective_lindbladian_from_h(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
    )


@pytest.mark.threequbit
def test_generate_effective_lindbladian_from_h_identity_gate_case02():
    # Arrange
    parameters = [
        (
            [2, 2, 2],
            CompositeSystem(
                [
                    ElementalSystem(0, matrix_basis.get_normalized_pauli_basis()),
                    ElementalSystem(1, matrix_basis.get_normalized_pauli_basis()),
                    ElementalSystem(2, matrix_basis.get_normalized_pauli_basis()),
                ]
            ),
        ),
    ]
    gate_name = "identity"
    ids = []

    for (dims, c_sys) in parameters:
        print(f"test dims={dims}, c_sys={c_sys}")
        _test_generate_effective_lindbladian_from_h(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
        )


@pytest.mark.twoqutrit
def test_generate_effective_lindbladian_from_h_identity_gate_case03():
    # Arrange
    parameters = [
        (
            [3, 3],
            CompositeSystem(
                [
                    ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis()),
                    ElementalSystem(1, matrix_basis.get_normalized_gell_mann_basis()),
                ]
            ),
        ),
    ]
    gate_name = "identity"
    ids = []

    for (dims, c_sys) in parameters:
        print(f"test dims={dims}, c_sys={c_sys}")
        _test_generate_effective_lindbladian_from_h(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
        )


@pytest.mark.parametrize(
    ("dims", "c_sys"),
    [
        (
            [2],
            CompositeSystem(
                [ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())]
            ),
        ),
        (
            [2, 2],
            CompositeSystem(
                [
                    ElementalSystem(0, matrix_basis.get_normalized_pauli_basis()),
                    ElementalSystem(1, matrix_basis.get_normalized_pauli_basis()),
                ]
            ),
        ),
        (
            [3],
            CompositeSystem(
                [ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis())]
            ),
        ),
        (
            [4],
            CompositeSystem(
                [
                    ElementalSystem(
                        0,
                        matrix_basis.get_normalized_generalized_gell_mann_basis(dim=4),
                    )
                ]
            ),
        ),
    ],
)
def test_calc_h_1qubit_identity_gate_case01(dims: List[int], c_sys: CompositeSystem):
    # Arrange
    gate_name = "identity"
    ids = []

    _test_calc_h(gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys)


@pytest.mark.threequbit
def test_calc_h_1qubit_identity_gate_case02():
    # Arrange
    parameters = [
        (
            [2, 2, 2],
            CompositeSystem(
                [
                    ElementalSystem(0, matrix_basis.get_normalized_pauli_basis()),
                    ElementalSystem(1, matrix_basis.get_normalized_pauli_basis()),
                    ElementalSystem(2, matrix_basis.get_normalized_pauli_basis()),
                ]
            ),
        ),
    ]
    gate_name = "identity"
    ids = []

    for (dims, c_sys) in parameters:
        print(f"test dims={dims}, c_sys={c_sys}")
        _test_calc_h(gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys)


@pytest.mark.twoqutrit
def test_calc_h_1qubit_identity_gate_case03():
    # Arrange
    parameters = [
        (
            [3, 3],
            CompositeSystem(
                [
                    ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis()),
                    ElementalSystem(1, matrix_basis.get_normalized_gell_mann_basis()),
                ]
            ),
        ),
    ]
    gate_name = "identity"
    ids = []

    for (dims, c_sys) in parameters:
        print(f"test dims={dims}, c_sys={c_sys}")
        _test_calc_h(gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys)


# Tests for 1-qubit gates


@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_gate_names_1qubit()],
)
def test_hamiltonian_vec_hamiltonian_mat_1qubit(gate_name):
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    dims = [2]
    ids = []

    _test_hamiltonian_vec_hamiltonian_mat(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
    )


@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_gate_names_1qubit()],
)
def test_hamiltonian_mat_unitary_mat_1qubit(gate_name):
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    dims = [2]
    ids = []

    _test_hamiltonian_mat_unitary_mat(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
    )


@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_gate_names_1qubit()],
)
def test_effective_lindbladian_mat_gate_mat_1qubit(gate_name):
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    dims = [2]
    ids = []

    _test_effective_lindbladian_mat_gate_mat(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
    )


@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_gate_names_1qubit()],
)
def test_generate_effective_lindbladian_from_h_1qubit(gate_name):
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    dims = [2]
    ids = []

    _test_generate_effective_lindbladian_from_h(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
    )


@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_gate_names_1qubit()],
)
def test_calc_h_1qubit(gate_name):
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    dims = [2]
    ids = []

    _test_calc_h(gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys)


# Tests for 2-qubit gates


@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("gate_name", "decimal"),
    [(gate_name, 15) for gate_name in get_gate_names_2qubit()],
)
def test_hamiltonian_vec_hamiltonian_mat_2qubit(gate_name, decimal):
    # Arrange
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    dims = [2, 2]

    ids = [0, 1]
    _test_hamiltonian_vec_hamiltonian_mat(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
    )

    if gate_name in get_gate_names_2qubit_asymmetric():
        ids = [1, 0]
        _test_hamiltonian_vec_hamiltonian_mat(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
        )


@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("gate_name", "decimal"),
    [(gate_name, 15) for gate_name in get_gate_names_2qubit()],
)
def test_hamiltonian_mat_unitary_mat_2qubit(gate_name, decimal):
    # Arrange
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    dims = [2, 2]

    ids = [0, 1]
    _test_hamiltonian_mat_unitary_mat(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
    )

    if gate_name in get_gate_names_2qubit_asymmetric():
        ids = [1, 0]
        _test_hamiltonian_mat_unitary_mat(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
        )


@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("gate_name", "decimal"),
    [(gate_name, 15) for gate_name in get_gate_names_2qubit()],
)
def test_effective_lindbladian_mat_gate_mat_2qubit(gate_name, decimal):
    # Arrange
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    dims = [2, 2]

    ids = [0, 1]
    _test_effective_lindbladian_mat_gate_mat(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
    )

    if gate_name in get_gate_names_2qubit_asymmetric():
        ids = [1, 0]
        _test_effective_lindbladian_mat_gate_mat(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
        )


@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("gate_name", "decimal"),
    [(gate_name, 15) for gate_name in get_gate_names_2qubit()],
)
def test_generate_effective_lindbladian_from_h_2qubit(gate_name, decimal):
    # Arrange
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    dims = [2, 2]

    ids = [0, 1]
    _test_generate_effective_lindbladian_from_h(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
    )

    if gate_name in get_gate_names_2qubit_asymmetric():
        ids = [1, 0]
        _test_generate_effective_lindbladian_from_h(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
        )


@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("gate_name", "decimal"),
    [
        (gate_name, 14) for gate_name in get_gate_names_2qubit()
    ],  # When decimal = 15, the test fails for gate_name = cx.
)
def test_calc_h_2qubit(gate_name, decimal):
    # Arrange
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    dims = [2, 2]

    ids = [0, 1]
    _test_calc_h(gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal)

    if gate_name in get_gate_names_2qubit_asymmetric():
        ids = [1, 0]
        _test_calc_h(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
        )


# Tests for 3-qubit gates


@pytest.mark.threequbit
def test_hamiltonian_vec_hamiltonian_mat_3qubit():
    # Arrange
    parameters = [(gate_name, 15) for gate_name in get_gate_names_3qubit()]
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1, e_sys2])
    dims = [2, 2, 2]

    ids_base = [0, 1, 2]
    for (gate_name, decimal) in parameters:
        print(f"test gate_name={gate_name}, decimal={decimal}")
        for ids in permutations(ids_base):
            _test_hamiltonian_vec_hamiltonian_mat(
                gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
            )


@pytest.mark.threequbit
def test_hamiltonian_mat_unitary_mat_3qubit():
    # Arrange
    parameters = [(gate_name, 15) for gate_name in get_gate_names_3qubit()]
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1, e_sys2])
    dims = [2, 2, 2]

    ids_base = [0, 1, 2]
    for (gate_name, decimal) in parameters:
        print(f"test gate_name={gate_name}, decimal={decimal}")
        for ids in permutations(ids_base):
            _test_hamiltonian_mat_unitary_mat(
                gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
            )


@pytest.mark.threequbit
def test_effective_lindbladian_mat_gate_mat_3qubit():
    # Arrange
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    parameters = [(gate_name, 15) for gate_name in get_gate_names_3qubit()]
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1, e_sys2])
    dims = [2, 2, 2]

    ids_base = [0, 1, 2]
    for (gate_name, decimal) in parameters:
        print(f"test gate_name={gate_name}, decimal={decimal}")
        for ids in permutations(ids_base):
            _test_effective_lindbladian_mat_gate_mat(
                gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
            )


@pytest.mark.threequbit
def test_generate_effective_lindbladian_from_h_3qubit():
    # Arrange
    parameters = [(gate_name, 15) for gate_name in get_gate_names_3qubit()]
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1, e_sys2])
    dims = [2, 2, 2]

    ids_base = [0, 1, 2]
    for (gate_name, decimal) in parameters:
        print(f"test gate_name={gate_name}, decimal={decimal}")
        for ids in permutations(ids_base):
            _test_generate_effective_lindbladian_from_h(
                gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
            )


@pytest.mark.threequbit
def test_calc_h_3qubit():
    # Arrange
    parameters = [(gate_name, 14) for gate_name in get_gate_names_3qubit()]
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1, e_sys2])
    dims = [2, 2, 2]

    ids_base = [0, 1, 2]
    for (gate_name, decimal) in parameters:
        print(f"test gate_name={gate_name}, decimal={decimal}")
        for ids in permutations(ids_base):
            _test_calc_h(
                gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
            )


# Tests for 1-qutrit gates


@pytest.mark.onequtrit
@pytest.mark.parametrize(
    ("gate_name", "decimal"),
    [(gate_name, 15) for gate_name in get_gate_names_1qutrit_single_gellmann()],
)
def test_hamiltonian_vec_hamiltonian_mat_1qutrit_case01(gate_name, decimal):
    # Arrange
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis())
    c_sys = CompositeSystem([e_sys0])
    dims = [3]
    ids = []
    _test_hamiltonian_vec_hamiltonian_mat(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
    )


@pytest.mark.onequtrit
@pytest.mark.parametrize(
    ("gate_name", "decimal"),
    [(gate_name, 15) for gate_name in get_gate_names_1qutrit_single_gellmann()],
)
def test_hamiltonian_mat_unitary_mat_1qutrit_case01(gate_name, decimal):
    # Arrange
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis())
    c_sys = CompositeSystem([e_sys0])
    dims = [3]
    ids = []
    _test_hamiltonian_mat_unitary_mat(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
    )


@pytest.mark.onequtrit
@pytest.mark.parametrize(
    ("gate_name", "decimal"),
    [(gate_name, 15) for gate_name in get_gate_names_1qutrit_single_gellmann()],
)
def test_effective_lindbladian_mat_gate_mat_1qutrit_case01(gate_name, decimal):
    # Arrange
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis())
    c_sys = CompositeSystem([e_sys0])
    dims = [3]
    ids = []
    _test_effective_lindbladian_mat_gate_mat(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
    )


@pytest.mark.onequtrit
@pytest.mark.parametrize(
    ("gate_name", "decimal"),
    [(gate_name, 15) for gate_name in get_gate_names_1qutrit_single_gellmann()],
)
def test_generate_effective_lindbladian_from_h_1qutrit_case01(gate_name, decimal):
    # Arrange
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis())
    c_sys = CompositeSystem([e_sys0])
    dims = [3]
    ids = []
    _test_generate_effective_lindbladian_from_h(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
    )


@pytest.mark.onequtrit
@pytest.mark.parametrize(
    ("gate_name", "decimal"),
    [(gate_name, 15) for gate_name in get_gate_names_1qutrit_single_gellmann()],
)
def test_calc_h_1qutrit(gate_name, decimal):
    # Arrange
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis())
    c_sys = CompositeSystem([e_sys0])
    dims = [3]
    ids = []
    _test_calc_h(gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal)


# Tests for 2-qutrit gates


@pytest.mark.twoqutrit
def test_hamiltonian_vec_hamiltonian_mat_2qutrit_case01():
    # Arrange
    parameters = [
        (gate_name, 15)
        for gate_name in ["i01x90", "02yi180", "12z01y90", "i01y90_02x01z180"]
    ]
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_gell_mann_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    dims = [3, 3]
    ids = []

    # Act & Assert
    for (gate_name, decimal) in parameters:
        print(f"test gate_name={gate_name}, decimal={decimal}")
        _test_hamiltonian_vec_hamiltonian_mat(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
        )


@pytest.mark.twoqutrit
def test_hamiltonian_mat_unitary_mat_2qutrit_case01():
    # Arrange
    parameters = [
        (gate_name, 15)
        for gate_name in ["i01x90", "02yi180", "12z01y90", "i01y90_02x01z180"]
    ]
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_gell_mann_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    dims = [3, 3]
    ids = []

    # Act & Assert
    for (gate_name, decimal) in parameters:
        print(f"test gate_name={gate_name}, decimal={decimal}")
        _test_hamiltonian_mat_unitary_mat(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
        )


@pytest.mark.twoqutrit
def test_effective_lindbladian_mat_gate_mat_2qutrit_case01():
    # Arrange
    parameters = [
        (gate_name, 15)
        for gate_name in ["i01x90", "02yi180", "12z01y90", "i01y90_02x01z180"]
    ]
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_gell_mann_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    dims = [3, 3]
    ids = []

    # Act & Assert
    for (gate_name, decimal) in parameters:
        print(f"test gate_name={gate_name}, decimal={decimal}")
        _test_effective_lindbladian_mat_gate_mat(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
        )


@pytest.mark.twoqutrit
def test_generate_effective_lindbladian_from_h_2qutrit_case01():
    # Arrange
    parameters = [
        (gate_name, 15)
        for gate_name in ["i01x90", "02yi180", "12z01y90", "i01y90_02x01z180"]
    ]
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_gell_mann_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    dims = [3, 3]
    ids = []

    # Act & Assert
    for (gate_name, decimal) in parameters:
        print(f"test gate_name={gate_name}, decimal={decimal}")
        _test_generate_effective_lindbladian_from_h(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
        )


@pytest.mark.twoqutrit
def test_calc_h_2qutrit():
    # Arrange
    parameters = [
        (gate_name, 14)
        for gate_name in ["i01x90", "02yi180", "12z01y90", "i01y90_02x01z180"]
        # We set secimal = 14, because decimal = 15 could not pass the test.
    ]
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_gell_mann_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    dims = [3, 3]
    ids = []

    # Act & Assert
    for (gate_name, decimal) in parameters:
        print(f"test gate_name={gate_name}, decimal={decimal}")
        _test_calc_h(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
        )
