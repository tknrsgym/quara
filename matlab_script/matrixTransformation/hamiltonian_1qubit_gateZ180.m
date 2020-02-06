function output = hamiltonian_1qubit_gateZ180()
%HAMILTONIAN_1QUBIT_GATEZ180 returns the Hamiltonian of the Z gate.
%   H = (Z-I) .* pi /2
    basis = matrixBasis_1qubit_pauli_unnormalized();
    output = 0.50 .* pi .* (basis(4).mat - basis(1).mat);
end

