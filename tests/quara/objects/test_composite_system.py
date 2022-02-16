from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import (
    get_comp_basis,
    get_pauli_basis,
    get_normalized_pauli_basis,
)
from quara.objects.operators import tensor_product
from quara.objects.composite_system_typical import generate_composite_system
from quara.utils import matrix_util


class TestCompositeSystem:
    def test_init_duplicate_elemental_system(self):
        e1 = ElementalSystem(1, get_pauli_basis())

        with pytest.raises(ValueError):
            _ = CompositeSystem([e1, e1])

    def test_init_duplicate_elemental_system_name(self):
        e1 = ElementalSystem(1, get_pauli_basis())
        e2 = ElementalSystem(1, get_pauli_basis())

        with pytest.raises(ValueError):
            _ = CompositeSystem([e1, e2])

    def test_init_sorted(self):
        # Arange
        e1 = ElementalSystem(1, get_pauli_basis())
        e2 = ElementalSystem(2, get_pauli_basis())
        e3 = ElementalSystem(3, get_pauli_basis())
        source = [e2, e3, e1]

        # Act
        actual = CompositeSystem(source)

        # Assert
        expected = [e1, e2, e3]
        assert actual[0] is expected[0]
        assert actual[1] is expected[1]
        assert actual[2] is expected[2]

        # Verify that source is not affected
        expected = [e2, e3, e1]
        assert source == expected

    def test_comp_basis(self):
        ### mode=default
        e1 = ElementalSystem(1, get_pauli_basis())
        e2 = ElementalSystem(2, get_comp_basis())

        # case: single ElementalSystem
        source = [e1]
        actual = CompositeSystem(source).comp_basis()
        expected = get_comp_basis()
        assert len(actual) == 4
        assert matrix_util.allclose(actual[0], expected[0])
        assert matrix_util.allclose(actual[1], expected[1])
        assert matrix_util.allclose(actual[2], expected[2])
        assert matrix_util.allclose(actual[3], expected[3])

        # case: multi ElementalSystem
        source = [e2, e1]
        actual = CompositeSystem(source).comp_basis()
        expected = tensor_product(get_comp_basis(), get_comp_basis())
        assert len(actual) == 16
        assert matrix_util.allclose(actual[0], expected[0])
        assert matrix_util.allclose(actual[1], expected[1])

        ### mode="row_major"
        e = ElementalSystem(1, get_comp_basis(mode="row_major"))
        source = [e]
        actual = CompositeSystem(source).comp_basis(mode="row_major")
        expected = get_comp_basis(mode="row_major")
        assert len(actual) == 4
        assert matrix_util.allclose(actual[0], expected[0])
        assert matrix_util.allclose(actual[1], expected[1])
        assert matrix_util.allclose(actual[2], expected[2])
        assert matrix_util.allclose(actual[3], expected[3])

        ### mode="column_major"
        e = ElementalSystem(1, get_comp_basis(mode="column_major"))
        source = [e]
        actual = CompositeSystem(source).comp_basis(mode="column_major")
        expected = get_comp_basis(mode="column_major")
        assert len(actual) == 4
        assert matrix_util.allclose(actual[0], expected[0])
        assert matrix_util.allclose(actual[1], expected[1])
        assert matrix_util.allclose(actual[2], expected[2])
        assert matrix_util.allclose(actual[3], expected[3])

        ### unsupported mode
        e = ElementalSystem(1, get_comp_basis(mode="column_major"))
        source = [e]
        with pytest.raises(ValueError):
            CompositeSystem(source).comp_basis(mode="unsupported")

    def test_basis(self):
        e1 = ElementalSystem(1, get_pauli_basis())
        e2 = ElementalSystem(2, get_comp_basis())

        # case: single ElementalSystem
        source = [e1]
        actual = CompositeSystem(source).basis()
        expected = get_pauli_basis()
        assert len(actual) == 4
        assert matrix_util.allclose(actual[0], expected[0])
        assert matrix_util.allclose(actual[1], expected[1])
        assert matrix_util.allclose(actual[2], expected[2])
        assert matrix_util.allclose(actual[3], expected[3])

        # case: multi ElementalSystem
        source = [e2, e1]
        actual = CompositeSystem(source).basis()
        expected = tensor_product(get_pauli_basis(), get_comp_basis())
        assert len(actual) == 16
        assert matrix_util.allclose(actual[0], expected[0])
        assert matrix_util.allclose(actual[1], expected[1])

    def test_access_dim(self):
        e1 = ElementalSystem(1, get_pauli_basis())
        e2 = ElementalSystem(2, get_comp_basis())

        # case: single ElementalSystem
        source = [e1]
        actual = CompositeSystem(source).dim
        assert actual == 2

        # case: multi ElementalSystem
        source = [e2, e1]
        actual = CompositeSystem(source).dim
        assert actual == 4

    def test_get_basis(self):
        e1 = ElementalSystem(1, get_pauli_basis())
        e2 = ElementalSystem(2, get_comp_basis())
        source = [e2, e1]
        c_sys = CompositeSystem(source)

        # access by int
        actual = c_sys.get_basis(0)
        expected = tensor_product(get_pauli_basis(), get_comp_basis())[0]
        assert matrix_util.allclose(actual, expected)

        # access by tuple
        actual = c_sys.get_basis((0, 0))
        expected = tensor_product(get_pauli_basis(), get_comp_basis())[0]
        assert matrix_util.allclose(actual, expected)

        actual = c_sys.get_basis((1, 2))
        expected = tensor_product(get_pauli_basis(), get_comp_basis())[6]  # 1*2**2 + 2
        assert matrix_util.allclose(actual, expected)

        actual = c_sys.get_basis((3, 1))
        expected = tensor_product(get_pauli_basis(), get_comp_basis())[13]  # 3*2**2 + 1
        assert matrix_util.allclose(actual, expected)

    def test_basis_basisconjugate(self):
        basis = get_normalized_pauli_basis()
        e1 = ElementalSystem(0, basis)
        source = [e1]
        c_sys = CompositeSystem(source)
        assert c_sys._basis_basisconjugate == None

        # access by int
        actual = c_sys.basis_basisconjugate(0)
        expected = matrix_util.kron(basis[0], np.conjugate(basis[0]))  # 0*2**2 + 0 = 0
        assert matrix_util.allclose(actual, expected)

        actual = c_sys.basis_basisconjugate(6)
        expected = matrix_util.kron(basis[1], np.conjugate(basis[2]))  # 1*2**2 + 2 = 6
        assert matrix_util.allclose(actual, expected)

        actual = c_sys.basis_basisconjugate(13)
        expected = matrix_util.kron(basis[3], np.conjugate(basis[1]))  # 3*2**2 + 1 = 13
        assert matrix_util.allclose(actual, expected)

        # access by tuple
        actual = c_sys.basis_basisconjugate((0, 0))
        expected = matrix_util.kron(basis[0], np.conjugate(basis[0]))  # 0*2**2 + 0 = 0
        assert matrix_util.allclose(actual, expected)

        actual = c_sys.basis_basisconjugate((1, 2))
        expected = matrix_util.kron(basis[1], np.conjugate(basis[2]))  # 1*2**2 + 2 = 6
        assert matrix_util.allclose(actual, expected)

        actual = c_sys.basis_basisconjugate((3, 1))
        expected = matrix_util.kron(basis[3], np.conjugate(basis[1]))  # 3*2**2 + 1 = 13
        assert matrix_util.allclose(actual, expected)

    def test_dict_from_hs_to_choi(self):
        # arrange
        basis = get_normalized_pauli_basis()
        e1 = ElementalSystem(0, basis)
        source = [e1]
        c_sys = CompositeSystem(source)
        assert c_sys._basis_basisconjugate == None

        # act
        actual = c_sys.dict_from_hs_to_choi

        # assert
        assert actual != None
        expected = {
            (0, 0): [
                (0, 0, (0.5 + 0j)),
                (0, 3, (0.5 + 0j)),
                (3, 0, (0.5 + 0j)),
                (3, 3, (0.5 + 0j)),
            ],
            (1, 1): [
                (0, 0, (0.5 + 0j)),
                (0, 3, (-0.5 - 0j)),
                (3, 0, (0.5 + 0j)),
                (3, 3, (-0.5 - 0j)),
            ],
            (2, 2): [
                (0, 0, (0.5 + 0j)),
                (0, 3, (0.5 + 0j)),
                (3, 0, (-0.5 + 0j)),
                (3, 3, (-0.5 + 0j)),
            ],
            (3, 3): [
                (0, 0, (0.5 + 0j)),
                (0, 3, (-0.5 - 0j)),
                (3, 0, (-0.5 + 0j)),
                (3, 3, (0.5 + 0j)),
            ],
            (0, 1): [
                (0, 1, (0.5 + 0j)),
                (0, 2, 0.5j),
                (3, 1, (0.5 + 0j)),
                (3, 2, 0.5j),
            ],
            (1, 0): [
                (0, 1, (0.5 + 0j)),
                (0, 2, -0.5j),
                (3, 1, (0.5 + 0j)),
                (3, 2, -0.5j),
            ],
            (2, 3): [
                (0, 1, (0.5 + 0j)),
                (0, 2, 0.5j),
                (3, 1, (-0.5 + 0j)),
                (3, 2, (-0 - 0.5j)),
            ],
            (3, 2): [
                (0, 1, (0.5 + 0j)),
                (0, 2, -0.5j),
                (3, 1, (-0.5 + 0j)),
                (3, 2, 0.5j),
            ],
            (0, 2): [
                (1, 0, (0.5 + 0j)),
                (1, 3, (0.5 + 0j)),
                (2, 0, -0.5j),
                (2, 3, -0.5j),
            ],
            (1, 3): [
                (1, 0, (0.5 + 0j)),
                (1, 3, (-0.5 - 0j)),
                (2, 0, -0.5j),
                (2, 3, (-0 + 0.5j)),
            ],
            (2, 0): [
                (1, 0, (0.5 + 0j)),
                (1, 3, (0.5 + 0j)),
                (2, 0, 0.5j),
                (2, 3, 0.5j),
            ],
            (3, 1): [
                (1, 0, (0.5 + 0j)),
                (1, 3, (-0.5 - 0j)),
                (2, 0, 0.5j),
                (2, 3, -0.5j),
            ],
            (0, 3): [
                (1, 1, (0.5 + 0j)),
                (1, 2, 0.5j),
                (2, 1, -0.5j),
                (2, 2, (0.5 + 0j)),
            ],
            (1, 2): [
                (1, 1, (0.5 + 0j)),
                (1, 2, -0.5j),
                (2, 1, -0.5j),
                (2, 2, (-0.5 - 0j)),
            ],
            (2, 1): [
                (1, 1, (0.5 + 0j)),
                (1, 2, 0.5j),
                (2, 1, 0.5j),
                (2, 2, (-0.5 + 0j)),
            ],
            (3, 0): [
                (1, 1, (0.5 + 0j)),
                (1, 2, -0.5j),
                (2, 1, 0.5j),
                (2, 2, (0.5 + 0j)),
            ],
        }
        assert len(actual) == len(expected)
        for a, e in zip(actual.items(), expected.items()):
            assert a[0] == e[0]
            assert len(a[1]) == len(e[1])
            for a_tuple, e_tuple in zip(a[1], e[1]):
                npt.assert_allclose(a_tuple, e_tuple)

    def test_dict_from_choi_to_hs(self):
        # arrange
        basis = get_normalized_pauli_basis()
        e1 = ElementalSystem(0, basis)
        source = [e1]
        c_sys = CompositeSystem(source)
        assert c_sys._basis_basisconjugate == None

        # act
        actual = c_sys.dict_from_choi_to_hs

        # assert
        assert actual != None
        expected = {
            (0, 0): [
                (0, 0, (0.5 + 0j)),
                (1, 1, (0.5 + 0j)),
                (2, 2, (0.5 + 0j)),
                (3, 3, (0.5 + 0j)),
            ],
            (0, 1): [
                (0, 1, (0.5 + 0j)),
                (1, 0, (0.5 + 0j)),
                (2, 3, (0.5 + 0j)),
                (3, 2, (0.5 + 0j)),
            ],
            (0, 2): [(0, 1, 0.5j), (1, 0, -0.5j), (2, 3, 0.5j), (3, 2, -0.5j)],
            (0, 3): [
                (0, 0, (0.5 + 0j)),
                (1, 1, (-0.5 - 0j)),
                (2, 2, (0.5 + 0j)),
                (3, 3, (-0.5 - 0j)),
            ],
            (1, 0): [
                (0, 2, (0.5 + 0j)),
                (1, 3, (0.5 + 0j)),
                (2, 0, (0.5 + 0j)),
                (3, 1, (0.5 + 0j)),
            ],
            (1, 1): [
                (0, 3, (0.5 + 0j)),
                (1, 2, (0.5 + 0j)),
                (2, 1, (0.5 + 0j)),
                (3, 0, (0.5 + 0j)),
            ],
            (1, 2): [(0, 3, 0.5j), (1, 2, -0.5j), (2, 1, 0.5j), (3, 0, -0.5j)],
            (1, 3): [
                (0, 2, (0.5 + 0j)),
                (1, 3, (-0.5 - 0j)),
                (2, 0, (0.5 + 0j)),
                (3, 1, (-0.5 - 0j)),
            ],
            (2, 0): [(0, 2, -0.5j), (1, 3, -0.5j), (2, 0, 0.5j), (3, 1, 0.5j)],
            (2, 1): [(0, 3, -0.5j), (1, 2, -0.5j), (2, 1, 0.5j), (3, 0, 0.5j)],
            (2, 2): [
                (0, 3, (0.5 + 0j)),
                (1, 2, (-0.5 - 0j)),
                (2, 1, (-0.5 + 0j)),
                (3, 0, (0.5 + 0j)),
            ],
            (2, 3): [(0, 2, -0.5j), (1, 3, (-0 + 0.5j)), (2, 0, 0.5j), (3, 1, -0.5j)],
            (3, 0): [
                (0, 0, (0.5 + 0j)),
                (1, 1, (0.5 + 0j)),
                (2, 2, (-0.5 + 0j)),
                (3, 3, (-0.5 + 0j)),
            ],
            (3, 1): [
                (0, 1, (0.5 + 0j)),
                (1, 0, (0.5 + 0j)),
                (2, 3, (-0.5 + 0j)),
                (3, 2, (-0.5 + 0j)),
            ],
            (3, 2): [(0, 1, 0.5j), (1, 0, -0.5j), (2, 3, (-0 - 0.5j)), (3, 2, 0.5j)],
            (3, 3): [
                (0, 0, (0.5 + 0j)),
                (1, 1, (-0.5 - 0j)),
                (2, 2, (-0.5 + 0j)),
                (3, 3, (0.5 + 0j)),
            ],
        }

        assert len(actual) == len(expected)
        for a, e in zip(actual.items(), expected.items()):
            assert a[0] == e[0]
            assert len(a[1]) == len(e[1])
            for a_tuple, e_tuple in zip(a[1], e[1]):
                npt.assert_allclose(a_tuple, e_tuple)

    def test_delete_dict_from_hs_to_choi(self):
        # arrange
        basis = get_normalized_pauli_basis()
        e1 = ElementalSystem(0, basis)
        source = [e1]
        c_sys = CompositeSystem(source)
        assert c_sys._dict_from_hs_to_choi == None
        assert c_sys.dict_from_hs_to_choi != None

        # act
        c_sys.delete_dict_from_hs_to_choi()

        # assert
        assert c_sys._dict_from_hs_to_choi == None
        assert c_sys.dict_from_hs_to_choi != None

    def test_delete_dict_from_choi_to_hs(self):
        # arrange
        basis = get_normalized_pauli_basis()
        e1 = ElementalSystem(0, basis)
        source = [e1]
        c_sys = CompositeSystem(source)
        assert c_sys._dict_from_choi_to_hs == None
        assert c_sys.dict_from_choi_to_hs != None

        # act
        c_sys.delete_dict_from_choi_to_hs()

        # assert
        assert c_sys._dict_from_choi_to_hs == None
        assert c_sys.dict_from_choi_to_hs != None

    def test_delete_basis_T_sparse(self):
        # arrange
        basis = get_normalized_pauli_basis()
        e1 = ElementalSystem(0, basis)
        source = [e1]
        c_sys = CompositeSystem(source)
        assert c_sys._basis_T_sparse == None
        assert c_sys.basis_T_sparse != None

        # act
        c_sys.delete_basis_T_sparse()

        # assert
        assert c_sys._basis_T_sparse == None
        assert c_sys.basis_T_sparse != None

    def test_delete_basisconjugate_sparse(self):
        # arrange
        basis = get_normalized_pauli_basis()
        e1 = ElementalSystem(0, basis)
        source = [e1]
        c_sys = CompositeSystem(source)
        assert c_sys._basisconjugate_sparse == None
        assert c_sys.basisconjugate_sparse != None

        # act
        c_sys.delete_basisconjugate_sparse()

        # assert
        assert c_sys._basisconjugate_sparse == None
        assert c_sys.basisconjugate_sparse != None

    def test_delete_basisconjugate_basis_sparse(self):
        # arrange
        basis = get_normalized_pauli_basis()
        e1 = ElementalSystem(0, basis)
        source = [e1]
        c_sys = CompositeSystem(source)
        assert c_sys._basisconjugate_basis_sparse == None
        assert c_sys.basisconjugate_basis_sparse != None

        # act
        c_sys.delete_basisconjugate_basis_sparse()

        # assert
        assert c_sys._basisconjugate_basis_sparse == None
        assert c_sys.basisconjugate_basis_sparse != None

    def test_delete_basis_basisconjugate_T_sparse(self):
        # arrange
        basis = get_normalized_pauli_basis()
        e1 = ElementalSystem(0, basis)
        source = [e1]
        c_sys = CompositeSystem(source)
        assert c_sys._basis_basisconjugate_T_sparse == None
        assert c_sys.basis_basisconjugate_T_sparse != None

        # act
        c_sys.delete_basis_basisconjugate_T_sparse()

        # assert
        assert c_sys._basis_basisconjugate_T_sparse == None
        assert c_sys.basis_basisconjugate_T_sparse != None

    def test_delete_basis_basisconjugate_T_sparse_from_1(self):
        # arrange
        basis = get_normalized_pauli_basis()
        e1 = ElementalSystem(0, basis)
        source = [e1]
        c_sys = CompositeSystem(source)
        assert c_sys._basis_basisconjugate_T_sparse_from_1 == None
        assert c_sys.basis_basisconjugate_T_sparse_from_1 != None

        # act
        c_sys.delete_basis_basisconjugate_T_sparse_from_1()

        # assert
        assert c_sys._basis_basisconjugate_T_sparse_from_1 == None
        assert c_sys.basis_basisconjugate_T_sparse_from_1 != None

    def test_delete_basishermitian_basis_T_from_1(self):
        # arrange
        basis = get_normalized_pauli_basis()
        e1 = ElementalSystem(0, basis)
        source = [e1]
        c_sys = CompositeSystem(source)
        assert c_sys._basishermitian_basis_T_from_1 == None
        assert c_sys.basishermitian_basis_T_from_1 != None

        # act
        c_sys.delete_basishermitian_basis_T_from_1()

        # assert
        assert c_sys._basishermitian_basis_T_from_1 == None
        assert c_sys.basishermitian_basis_T_from_1 != None

    def test_access_elemental_systems(self):
        e1 = ElementalSystem(1, get_pauli_basis())
        e2 = ElementalSystem(2, get_comp_basis())
        source = [e2, e1]
        actual = CompositeSystem(source).elemental_systems

        assert actual[0] is e1
        assert actual[1] is e2

    def test_is_orthonormal_hermitian_0thprop_identity(self):
        e1 = ElementalSystem(1, get_normalized_pauli_basis())
        e2 = ElementalSystem(2, get_normalized_pauli_basis())
        source = [e1, e2]
        assert CompositeSystem(source).is_orthonormal_hermitian_0thprop_identity == True

        e1 = ElementalSystem(1, get_comp_basis())
        e2 = ElementalSystem(2, get_normalized_pauli_basis())
        source = [e1, e2]
        assert (
            CompositeSystem(source).is_orthonormal_hermitian_0thprop_identity == False
        )

        e1 = ElementalSystem(1, get_normalized_pauli_basis())
        e2 = ElementalSystem(2, get_comp_basis())
        source = [e1, e2]
        assert (
            CompositeSystem(source).is_orthonormal_hermitian_0thprop_identity == False
        )

    def test_len(self):
        e1 = ElementalSystem(1, get_pauli_basis())
        e2 = ElementalSystem(2, get_comp_basis())
        source = [e2, e1]
        actual = len(CompositeSystem(source))

        assert actual == 2

    def test_iter(self):
        e1 = ElementalSystem(1, get_pauli_basis())
        e2 = ElementalSystem(2, get_comp_basis())
        source = [e2, e1]
        actual = iter(CompositeSystem(source))

        assert next(actual) is e1
        assert next(actual) is e2

    def test_getitem(self):
        e1 = ElementalSystem(1, get_pauli_basis())
        e2 = ElementalSystem(2, get_pauli_basis())

        c1 = CompositeSystem([e1, e2])
        assert c1[0] is e1
        assert c1[1] is not e1

        # Test that element_system list cannot be updated
        with pytest.raises(TypeError):
            c1[0] = e2

    def test_eq(self):
        e1 = ElementalSystem(1, get_pauli_basis())
        e2 = ElementalSystem(2, get_pauli_basis())
        e3 = ElementalSystem(3, get_pauli_basis())

        c1 = CompositeSystem([e1, e2])
        c2 = CompositeSystem([e1, e2])
        assert (c1 == c2) is True

        c1 = CompositeSystem([e1, e2])
        c2 = CompositeSystem([e1, e3])
        assert (c1 == c2) is False

        c1 = CompositeSystem([e1, e2])
        c2 = CompositeSystem([e1, e2, e3])
        assert (c1 == c2) is False

        assert (c1 == c1) is True

        assert (c1 == "string") is False
        assert (c1 == None) is False
        assert (c1 == 1) is False

    def test_str(self):
        # Arrange
        e1 = ElementalSystem(1, get_pauli_basis())
        c1 = CompositeSystem([e1])

        # Act
        # Only test that the call does not cause an error
        print(c1)


class TestCompositeSystemImmutable:
    def test_deney_update_elemental_systems_item(self):
        m_basis_1 = get_pauli_basis()
        m_basis_2 = get_pauli_basis()
        e1 = ElementalSystem(1, m_basis_1)
        e2 = ElementalSystem(2, m_basis_2)
        c_sys = CompositeSystem([e1, e2])

        m_basis_3 = get_pauli_basis()
        e3 = ElementalSystem(3, m_basis_3)

        with pytest.raises(TypeError):
            # TypeError: 'CompositeSystem' object does not support item assignment
            c_sys[0] = e3

        assert c_sys[0] is e1
        assert c_sys[1] is e2

        with pytest.raises(TypeError):
            # TypeError: 'tuple' object does not support item assignment
            c_sys.elemental_systems[0] = e3
        assert c_sys[0] is e1
        assert c_sys[1] is e2

        with pytest.raises(AttributeError):
            # AttributeError: can't set 'is_orthonormal_hermitian_0thprop_identity'
            c_sys.is_orthonormal_hermitian_0thprop_identity = True
        assert c_sys.is_orthonormal_hermitian_0thprop_identity == False


def test_basis_basisconjugate_1qubit():
    c_sys = generate_composite_system("qubit", 1)
    dir_path = Path(__file__).parent / "data"
    for i in range(16):
        path = dir_path / f"basis_basisconjugate/1qubit_{i}.npy"
        expected = np.load(path)
        actual = c_sys.basis_basisconjugate(basis_index=i)

        npt.assert_almost_equal(actual.toarray(), expected)


def test_basis_basisconjugate_T_sparse():
    # 2qubit
    # Arrange
    c_sys = generate_composite_system("qubit", 2)
    path = (
        Path(__file__).parent / "data/expected_basis_basisconjugate_T_sparse_2qubit.npy"
    )
    expected = np.load(path)
    # Act
    actual = c_sys._basis_basisconjugate_T_sparse
    # Assert
    assert actual is None

    # property
    # Act
    actual = c_sys.basis_basisconjugate_T_sparse
    # Assert
    npt.assert_allclose(actual.toarray(), expected, atol=10 ** (-15), rtol=0)


def test_basis_basisconjugate_T_sparse_from_1():
    # 2qubit
    # Arrange
    c_sys = generate_composite_system("qubit", 2)
    path = (
        Path(__file__).parent
        / "data/expected_basis_basisconjugate_T_sparse_from_1_2qubit.npy"
    )
    expected = np.load(path)
    # Act
    actual = c_sys._basis_basisconjugate_T_sparse_from_1
    # Assert
    assert actual is None

    # property
    # Act
    actual = c_sys.basis_basisconjugate_T_sparse_from_1
    # Assert
    npt.assert_allclose(actual.toarray(), expected, atol=10 ** (-15), rtol=0)


def test_basishermitian_basis_T_from_1():
    # 2qubit
    # Arrange
    c_sys = generate_composite_system("qubit", 2)
    path = (
        Path(__file__).parent / "data/expected_basishermitian_basis_T_from_1_2qubit.npy"
    )
    expected = np.load(path)
    # Act
    actual = c_sys._basishermitian_basis_T_from_1
    # Assert
    assert actual is None

    # Act
    actual = c_sys.basishermitian_basis_T_from_1
    # Assert
    npt.assert_allclose(actual.toarray(), expected, atol=10 ** (-15), rtol=0)


def test_basisconjugate_basis_sparse():
    # 2qubit
    # Arrange
    c_sys = generate_composite_system("qubit", 2)
    path = (
        Path(__file__).parent / "data/expected_basisconjugate_basis_sparse_2qubit.npy"
    )
    expected = np.load(path)
    # Act
    actual = c_sys._basisconjugate_basis_sparse
    # Assert
    assert actual is None

    # Act
    actual = c_sys.basisconjugate_basis_sparse
    # Assert
    npt.assert_allclose(actual.toarray(), expected, atol=10 ** (-15), rtol=0)
