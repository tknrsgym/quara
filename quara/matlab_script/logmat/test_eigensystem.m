format long;
%% Eigenvalues of Hamiltonian

% 1-qubit
% str = "X90"
% matH = hamiltonian_1qubit_gate_target(str);
% matH2 = matrix_toTraceless(matH);
% list_eval = eig(matH2)

% 2-qubit
%str = "ZX90"
% matH = hamiltonian_2qubit_gate_target(str);
% %matH = matrix_toTraceless(matH);
% list_eval = eig(matH)

%% Eigenvalues of Lindbladian with Hamiltonian only

% % 1-qubit
% str = "X180";
% matH = hamiltonian_1qubit_gate_target(str);
% HS_L = HScb_from_hamiltonian(matH);
% list_eval = eig(HS_L)
% 
% % 2-qubit
% str = "ZZ90";
% matH = hamiltonian_2qubit_gate_target(str);
% HS_L = HScb_from_hamiltonian(matH);
% list_eval = eig(HS_L);
% list_eval = sort(list_eval,'ComparisonMethod','abs')

%% Eigenvalues of Lindbladian with Hamiltonian and Decoherence

% 1-qubit
% str = "X90";
% matH = hamiltonian_1qubit_gate_target(str);
% vecGamma = [0.01, 0.02, 0.03];
% list_c = jumpOperator_1qubit_model01(vecGamma);
% HS_L = HScb_Lindbladian_from_hamiltonian_jumpOperator(matH, list_c);
% list_eval = eig(HS_L)

% 2-qubit
str = "CX";
matH = hamiltonian_2qubit_gate_target(str);
vecGamma1 = [0.01, 0.02, 0.03];
vecGamma2 = [0.008, 0.014, 0.023];
list_c = jumpOperator_2qubit_model01(vecGamma1, vecGamma2);
HS_L = HScb_Lindbladian_from_hamiltonian_jumpOperator(matH, list_c);
list_eval = eig(HS_L);
list_eval = sort(list_eval,'ComparisonMethod','abs')

%% 


