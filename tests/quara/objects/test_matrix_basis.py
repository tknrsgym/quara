import numpy as np

from quara.objects.matrix_basis import MatrixBasis

# computational basis
array00 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
array01 = np.array([[0, 1], [0, 0]], dtype=np.complex128)
array10 = np.array([[0, 0], [1, 0]], dtype=np.complex128)
array11 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
comp_basis = MatrixBasis([array00, array01, array10, array11])

# Pauli basis
identity = 1 / np.sqrt(2) * np.array([[1, 0], [0, 1]], dtype=np.complex128)
pauli_x = 1 / np.sqrt(2) * np.array([[0, 1], [1, 0]], dtype=np.complex128)
pauli_y = 1 / np.sqrt(2) * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
pauli_z = 1 / np.sqrt(2) * np.array([[1, 0], [0, -1]], dtype=np.complex128)
pauli_basis = MatrixBasis([identity, pauli_x, pauli_y, pauli_z])

print(pauli_basis.dim)
print(pauli_basis.is_squares())
print(pauli_basis.is_same_size())
print(pauli_basis.is_basis())
print(pauli_basis.is_orthogonal())
print(pauli_basis.is_normal())
print(pauli_basis.is_hermitian())
print(pauli_basis.is_scalar_mult_of_identity())
print(pauli_basis.is_trace_less())
print(pauli_basis[0])
print(pauli_basis.size())
print(len(pauli_basis))
print(pauli_basis)
