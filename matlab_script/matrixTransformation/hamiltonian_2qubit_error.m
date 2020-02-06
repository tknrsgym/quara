function output = hamiltonian_2qubit_error(input)
%HAMILTONIAN_2QUBIT_ERROR returns the 2-qubit error Hamiltonian
%   H = sum_{j=1}^{16} input(j) .* basis(j)
    assert(size(input, 1) == 1, "Size of input for hamiltonian_2qubit_error() is invalid!")
    assert(size(input, 2) == 16, "Size of input for hamiltonian_2qubit_error() is invalid!")
    basis = matrixBasis_2qubit_pauli_unnormalized(); 
    basisNo = size(basis, 2);
    H = zeros(4,4);
    for iBasis = 1:basisNo
        H = H + input(iBasis) .* basis(iBasis).mat;
    end
    output = H;
end

