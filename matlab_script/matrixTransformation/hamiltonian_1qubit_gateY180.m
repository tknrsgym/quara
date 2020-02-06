function output = hamiltonian_1qubit_gateY180()
%HAMILTONIAN_1QUBIT_GATEY180 returns the Hamiltonian of the Y gate.
%   H = (Y-I) .* pi /2
    basis = matrixBasis_1qubit_pauli_unnormalized();
    output = 0.50 .* pi .* (basis(3).mat - basis(1).mat);
end

