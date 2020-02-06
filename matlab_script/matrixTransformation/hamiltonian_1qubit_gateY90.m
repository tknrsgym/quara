function output = hamiltonian_1qubit_gateY90()
%HAMILTONIAN_1QUBIT_GATEY90 returns the Hamiltonian of the Y45 gate.
%   H = (Y-I) .* pi /4
    basis = matrixBasis_1qubit_pauli_unnormalized();
    output = 0.25 .* pi .* (basis(3).mat - basis(1).mat);
end

