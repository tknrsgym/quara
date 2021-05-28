import numpy as np
import pytest

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import (
    get_comp_basis,
    get_pauli_basis,
    get_normalized_pauli_basis,
)
from quara.objects.operators import tensor_product


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
        assert np.all(actual[0] == expected[0])
        assert np.all(actual[1] == expected[1])
        assert np.all(actual[2] == expected[2])
        assert np.all(actual[3] == expected[3])

        # case: multi ElementalSystem
        source = [e2, e1]
        actual = CompositeSystem(source).comp_basis()
        expected = tensor_product(get_comp_basis(), get_comp_basis())
        assert len(actual) == 16
        assert np.all(actual[0] == expected[0])
        assert np.all(actual[1] == expected[1])

        ### mode="row_major"
        e = ElementalSystem(1, get_comp_basis(mode="row_major"))
        source = [e]
        actual = CompositeSystem(source).comp_basis(mode="row_major")
        expected = get_comp_basis(mode="row_major")
        assert len(actual) == 4
        assert np.all(actual[0] == expected[0])
        assert np.all(actual[1] == expected[1])
        assert np.all(actual[2] == expected[2])
        assert np.all(actual[3] == expected[3])

        ### mode="column_major"
        e = ElementalSystem(1, get_comp_basis(mode="column_major"))
        source = [e]
        actual = CompositeSystem(source).comp_basis(mode="column_major")
        expected = get_comp_basis(mode="column_major")
        assert len(actual) == 4
        assert np.all(actual[0] == expected[0])
        assert np.all(actual[1] == expected[1])
        assert np.all(actual[2] == expected[2])
        assert np.all(actual[3] == expected[3])

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
        assert np.all(actual[0] == expected[0])
        assert np.all(actual[1] == expected[1])
        assert np.all(actual[2] == expected[2])
        assert np.all(actual[3] == expected[3])

        # case: multi ElementalSystem
        source = [e2, e1]
        actual = CompositeSystem(source).basis()
        expected = tensor_product(get_pauli_basis(), get_comp_basis())
        assert len(actual) == 16
        assert np.all(actual[0] == expected[0])
        assert np.all(actual[1] == expected[1])

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
        assert np.all(actual == expected)

        # access by tuple
        actual = c_sys.get_basis((0, 0))
        expected = tensor_product(get_pauli_basis(), get_comp_basis())[0]
        assert np.all(actual == expected)

        actual = c_sys.get_basis((1, 2))
        expected = tensor_product(get_pauli_basis(), get_comp_basis())[6]  # 1*2**2 + 2
        assert np.all(actual == expected)

        actual = c_sys.get_basis((3, 1))
        expected = tensor_product(get_pauli_basis(), get_comp_basis())[13]  # 3*2**2 + 1
        assert np.all(actual == expected)

    def test_basis_basisconjugate(self):
        basis = get_normalized_pauli_basis()
        e1 = ElementalSystem(0, basis)
        source = [e1]
        c_sys = CompositeSystem(source)

        # access by int
        actual = c_sys.basis_basisconjugate(0)
        expected = np.kron(basis[0], np.conjugate(basis[0]))  # 0*2**2 + 0 = 0
        assert np.all(actual == expected)

        actual = c_sys.basis_basisconjugate(6)
        expected = np.kron(basis[1], np.conjugate(basis[2]))  # 1*2**2 + 2 = 6
        assert np.all(actual == expected)

        actual = c_sys.basis_basisconjugate(13)
        expected = np.kron(basis[3], np.conjugate(basis[1]))  # 3*2**2 + 1 = 13
        assert np.all(actual == expected)

        # access by tuple
        actual = c_sys.basis_basisconjugate((0, 0))
        expected = np.kron(basis[0], np.conjugate(basis[0]))  # 0*2**2 + 0 = 0
        assert np.all(actual == expected)

        actual = c_sys.basis_basisconjugate((1, 2))
        expected = np.kron(basis[1], np.conjugate(basis[2]))  # 1*2**2 + 2 = 6
        assert np.all(actual == expected)

        actual = c_sys.basis_basisconjugate((3, 1))
        expected = np.kron(basis[3], np.conjugate(basis[1]))  # 3*2**2 + 1 = 13
        assert np.all(actual == expected)

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
