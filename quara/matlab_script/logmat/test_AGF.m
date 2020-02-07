format long;

%% averageGateFidelity_from_HS_Gate.m
% 1-qubit
% dim = 2;
% 
% str = "H";
% matH_target = hamiltonian_1qubit_gate_target(str);
% HS_L_target = HScb_from_hamiltonian(matH_target);
% HS_G_target = expm(HS_L_target);
% 
% vecH_error  = [0.0, 0.02, 0.01, 0.03];
% vecH_error  = zeros(1,4);
% matH_error  = hamiltonian_1qubit_error(vecH_error);
% matH = matH_target + matH_error;
% matH = matrix_toTraceless(matH);
% 
% vecGamma = [0.01, 0.01, 0.02];
% list_c = jumpOperator_1qubit_model01(vecGamma);
% HS_L = HScb_Lindbladian_from_hamiltonian_jumpOperator(matH, list_c);
% HS_G = expm(HS_L);
% 
% AGF = averageGateFidelity_from_HS_Gate(HS_G_target, HS_G, dim)

% 2-qubit
dim = 4;

str = "SWAP";
matH_target = hamiltonian_2qubit_gate_target(str);
HS_L_target = HScb_from_hamiltonian(matH_target);
HS_G_target = expm(HS_L_target);

vecH_error  = [0.0, 0.02, 0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15];
%vecH_error  = zeros(1,16);
matH_error  = hamiltonian_2qubit_error(vecH_error);
matH = matH_target + matH_error;
matH = matrix_toTraceless(matH);

vecGamma1 = [0.01, 0.01, 0.02];
%vecGamma1 = [0.0, 0.0, 0.0];
vecGamma2 = [0.008, 0.009, 0.014];
%vecGamma2 = [0.0, 0.0, 0.0];
list_c = jumpOperator_2qubit_model01(vecGamma1, vecGamma2);
HS_L = HScb_Lindbladian_from_hamiltonian_jumpOperator(matH, list_c);
HS_G = expm(HS_L);

AGF1 = averageGateFidelity_from_HS_Gate(HS_G_target, HS_G, dim);
L0 = HS_L_target;
L1 = HS_L;
AGF2 = averageGateFidelity_from_HS_Lindbladian(L0, L1, dim);

diff = abs(AGF1 - AGF2)

%% 

