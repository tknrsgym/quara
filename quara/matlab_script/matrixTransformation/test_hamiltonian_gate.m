format long;

% %% 1-qubit, Identity gate
% H_gate = hamiltonian_1qubit_gateI() 
% matU = unitaryMatrix_from_hamiltonian(H_gate)


%% 1-qubit, X180 gate
% H_gate = hamiltonian_1qubit_gateX180() 
% matU = unitaryMatrix_from_hamiltonian(H_gate)
%  

%% 1-qubit, Y180 gate
% H_gate = hamiltonian_1qubit_gateY180() 
% matU = unitaryMatrix_from_hamiltonian(H_gate)
 
%% 1-qubit, Z180 gate
% H_gate = hamiltonian_1qubit_gateZ180() 
% matU = unitaryMatrix_from_hamiltonian(H_gate)

%% 1-qubit, X90 gate
% H_gate = hamiltonian_1qubit_gateX90() 
% matU = unitaryMatrix_from_hamiltonian(H_gate) 
% matU * matU

%% 1-qubit, Y90 gate
% H_gate = hamiltonian_1qubit_gateY90() 
% matU = unitaryMatrix_from_hamiltonian(H_gate)
% matU * matU
 
%% 1-qubit, Z90 gate
% H_gate = hamiltonian_1qubit_gateZ90() 
% matU = unitaryMatrix_from_hamiltonian(H_gate)
% matU * matU

%% 1-qubit, H gate
% H_gate = hamiltonian_1qubit_gateH() 
% matU = unitaryMatrix_from_hamiltonian(H_gate)
% matU * matU
% 

%% 1-qubit, S gate
% H_gate = hamiltonian_1qubit_gateS() 
% matU = unitaryMatrix_from_hamiltonian(H_gate)
% matU * matU


%% 1-qubit, S^dagger gate
% H_gate = hamiltonian_1qubit_gateSdagger() 
% matU = unitaryMatrix_from_hamiltonian(H_gate)
% matU * matU


%% 1-qubit, T gate
% H_gate = hamiltonian_1qubit_gateT() 
% matU = unitaryMatrix_from_hamiltonian(H_gate)
% matU * matU * matU * matU

%% 1-qubit, target
% list_str = ["I", "X90", "Y90", "Z90", "X180", "Y180", "Z180", "H", "S", "Sdagger", "T"];
% size_list_str = size(list_str, 2);
% for iStr = 1:size_list_str 
%     str = list_str(iStr);
%     H_gate = hamiltonian_1qubit_gate_target(str)
%     matU = unitaryMatrix_from_hamiltonian(H_gate)
% end

%% 1-qubit, error
% vec_error = [0.1, 0.2, 0.3];
% H_gate = hamiltonian_1qubit_error(vec_error)

%% 1-qubit, prepared
% % Target gate
% list_str = ["I", "X90", "Y90", "Z90", "X180", "Y180", "Z180", "H", "S", "Sdagger", "T"];
% 
% % Hamiltonian Error
% vec_error = [0.1, 0.2, 0.3];
% 
% size_list_str = size(list_str, 2);
% for iStr = 1:size_list_str 
%     str = list_str(iStr);
%     H_gate = hamiltonian_1qubit_gate_prepared(str, vec_error)
%     matU = unitaryMatrix_from_hamiltonian(H_gate)
% end

%% 1-qubit, dissipator
% vecGamma = [1, 1, 2];
% list_c = jumpOperator_1qubit_model01(vecGamma);
% cNo = size(list_c, 2);
% for ic = 1:cNo
%    list_c(ic).mat 
% end

%% 2-qubit, CX gate
% H_gate = hamiltonian_2qubit_gateCX()
% matU = unitaryMatrix_from_hamiltonian(H_gate)


%% 2-qubit, CZ gate
% H_gate = hamiltonian_2qubit_gateCZ()
% matU = unitaryMatrix_from_hamiltonian(H_gate)
% [V, D, W] = eig(H_gate)


%% 2-qubit, SWAP gate
% H_gate = hamiltonian_2qubit_gateSWAP()
% matU = unitaryMatrix_from_hamiltonian(H_gate)
% [V, D, W] = eig(H_gate)


%% 2-qubit, ZX45 gate
% H_gate = hamiltonian_2qubit_gateZX90()
% matU = unitaryMatrix_from_hamiltonian(H_gate)
% [V, D, W] = eig(H_gate)

%% 2-qubit, ZZ45 gate
% H_gate = hamiltonian_2qubit_gateZZ90()
% matU = unitaryMatrix_from_hamiltonian(H_gate)
% [V, D, W] = eig(H_gate)

%% 2-qubit, dissipator

vecGamma1 = [1, 1, 2];
vecGamma2 = [1, 1, 2];
list_c = jumpOperator_2qubit_model01(vecGamma1, vecGamma2);
cNo = size(list_c, 2);
for ic = 1:cNo
   list_c(ic).mat 
end

