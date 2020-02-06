function output = hamiltonian_1qubit_gate_target(input)
%HAMILTONIAN_1QUBIT_GATE_TARGET returns the target Hamiltonian
%   input : string
    H = zeros(2,2);
    if input == "I"
        H = hamiltonian_1qubit_gateI();
    elseif input == "X90"    
        H = hamiltonian_1qubit_gateX90();
    elseif input == "Y90"    
        H = hamiltonian_1qubit_gateY90();
    elseif input == "Z90"    
        H = hamiltonian_1qubit_gateZ90();
    elseif input == "X180"    
        H = hamiltonian_1qubit_gateX180();
    elseif input == "Y180"    
        H = hamiltonian_1qubit_gateY180();
    elseif input == "Z180"    
        H = hamiltonian_1qubit_gateZ180(); 
    elseif input == "H"    
        H = hamiltonian_1qubit_gateH();               
    elseif input == "S"    
        H = hamiltonian_1qubit_gateS();
    elseif input == "Sdagger"    
        H = hamiltonian_1qubit_gateSdagger();
    elseif input == "T"    
        H = hamiltonian_1qubit_gateT(); 
    else
        disp("[Warning] input string for hamiltonian_1qubit_gate_target() is invalid!")
    end
    output = H;
end