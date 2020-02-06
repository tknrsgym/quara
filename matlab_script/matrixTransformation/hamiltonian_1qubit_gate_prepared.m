function output = hamiltonian_1qubit_gate_prepared(str, vec_error)
%HAMILTONIAN_1QUBIT_GATE_PREPARED returns the 1-qubit prepared Hamiltonian
%   H = H_target + H_error
    H_target = hamiltonian_1qubit_gate_target(str);
    H_error  = hamiltonian_1qubit_error(vec_error);
    H = H_target + H_error;
    output = H;
end

