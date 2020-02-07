function output = hamiltonian_1qubit_gateS()
%HAMILTONIAN_1QUBIT_GATEI returns the Hamiltonian of the Phase gate.
%   Hamiltonian = (Z - I) .* pi / 4
    basis = matrixBasis_1qubit_pauli_unnormalized;
    output = (basis(4).mat - basis(1).mat) .* pi .* 0.25;
end

