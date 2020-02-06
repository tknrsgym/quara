function output = hamiltonian_1qubit_gateT()
%HAMILTONIAN_1QUBIT_GATEI returns the Hamiltonian of the T gate.
%   Hamiltonian = (Z - I) .* pi / 8
    basis = matrixBasis_1qubit_pauli_unnormalized;
    output = (basis(4).mat - basis(1).mat) .* pi .* 0.125;
end

