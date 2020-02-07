function output = hamiltonian_1qubit_gateX180()
%HAMILTONIAN_1QUBIT_GATX180I returns the Hamiltonian of the X gate.
%   H = (X-I) .* pi /2
    basis = matrixBasis_1qubit_pauli_unnormalized();
    output = 0.50 .* pi .* (basis(2).mat - basis(1).mat);
end

