function output = hamiltonian_1qubit_gateSdagger()
%HAMILTONIAN_1QUBIT_GATESDAGGER returns the Hamiltonian of the complex conjugate of the Phase gate.
%   Hamiltonian = -(Z - I) .* pi / 4
    basis = matrixBasis_1qubit_pauli_unnormalized;
    output = (-basis(4).mat - basis(1).mat) .* pi .* 0.25;
end

