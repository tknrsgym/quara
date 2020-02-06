function output = hamiltonian_1qubit_gateZ90()
%HAMILTONIAN_1QUBIT_GATEZ90 returns the Hamiltonian of the Z90 gate.
%   H = (Z-I) .* pi /4
    basis = matrixBasis_1qubit_pauli_unnormalized();
    output = 0.25 .* pi .* (basis(4).mat - basis(1).mat);
end

