function output = hamiltonian_1qubit_gateH()
%HAMILTONIAN_1QUBIT_GATEI returns the Hamiltonian of the Hadamard gate.
%   Hamiltonian = (H - I) .* pi / 2
    matH = [1, 1; 1, -1] ./ sqrt(2);
    output = (matH - eye(2,2)) .* pi .* 0.50;
end

