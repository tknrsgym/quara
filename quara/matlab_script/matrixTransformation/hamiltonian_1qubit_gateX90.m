function output = hamiltonian_1qubit_gateX90()
%HAMILTONIAN_1QUBIT_GATEX90 returns the Hamiltonian of the X45 gate.
%   H = (X - I) .* pi/4
    basis = matrixBasis_1qubit_pauli_unnormalized();
    output = 0.25 .* pi .* (basis(2).mat - basis(1).mat);
end

