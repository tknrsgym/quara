function output = hamiltonian_2qubit_gate_target(input)
%HAMILTONIAN_2QUBIT_GATE_TARGET returns the target Hamiltonian
%   input : string
    H = zeros(4,4);
    if input == "I"
        H = hamiltonian_2qubit_gateI();
    elseif input == "CX"    
        H = hamiltonian_2qubit_gateCX();
    elseif input == "CZ"    
        H = hamiltonian_2qubit_gateCZ();
    elseif input == "SWAP"    
        H = hamiltonian_2qubit_gateSWAP();
    elseif input == "ZX90"    
        H = hamiltonian_2qubit_gateZX90();
    elseif input == "ZZ90"    
        H = hamiltonian_2qubit_gateZZ90();
    else
        disp("[Warning] input string for hamiltonian_2qubit_gate_target() is invalid!")
    end
    output = H;
end