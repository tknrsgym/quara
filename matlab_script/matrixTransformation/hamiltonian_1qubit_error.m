function output = hamiltonian_1qubit_error(input)
%HAMILTONIAN_1QUBIT_ERROR returns the 1-qubit error Hamiltonian
%   H = input(1) .* I + input(2) .* X + input(3) .* Y + input(3) .* input(4)
    assert(size(input, 1) == 1, "Size of input for hamiltonian_1qubit_error() is invalid!")
    assert(size(input, 2) == 4, "Size of input for hamiltonian_1qubit_error() is invalid!")
    basis = matrixBasis_1qubit_pauli_unnormalized(); 
    H = zeros(2,2);
    for i = 1:4
        H = H + input(i) .* basis(i).mat;
    end
    output = H;
end

