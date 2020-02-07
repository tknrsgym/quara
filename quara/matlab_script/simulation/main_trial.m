clear;
format long;

%% Setting

% = = = = = [ Input ] = = = = = 
% 1-qubit
dim = 2;
str = "X90";
vecH_error  = [0.0, 0.02, 0.01, 0.03];
vecGamma = [0.0, 0.0, 0.0];
k = 11;
% = = = = = = = = = = = = = = = 

basis_pauli_normalized = matrixBasis_1qubit_pauli_normalized();
basis_comp = matrixBasis_1qubit_comp();

matH_target = hamiltonian_1qubit_gate_target(str);
HS_L_target = HScb_from_hamiltonian(matH_target);

matH_error  = hamiltonian_1qubit_error(vecH_error);
matH = matH_target + matH_error;
matH = matrix_toTraceless(matH);

list_c = jumpOperator_1qubit_model01(vecGamma);
matJ = matJ_from_jumpOperator_TPcondition(list_c);
matK = matK_from_jumpOperator(list_c, basis_pauli_normalized);

HS_L = HScb_Lindbladian_from_hamiltonian_jumpOperator(matH, list_c);
HS_G = expm(HS_L);
HS_Lk = k .* HS_L;
HS_Gk = expm(HS_Lk);

%% Check whether conditions are satisfied or not




%% Computation 1: 
eps = 10^(-9);
HS_Lk_recovered = nonpv_log_matrix(HS_Gk, HS_L_target, k, eps);
HS_L_recovered = HS_Lk_recovered ./ k;
HS_G_recovered = expm(HS_L_recovered);

diff_L = norm(HS_L - HS_L_recovered, 'fro')% Output
diff_G = norm(HS_G - HS_G_recovered, 'fro');% Output

[matH_recovered, matJ_recovered, matK_recovered] = dynamicsGenerator_from_HS_comp_Lindbladian(HS_L_recovered, basis_pauli_normalized);

diff_H = norm(matH - matH_recovered, 'fro');% Output
diff_J = norm(matJ - matJ_recovered, 'fro');% Output
diff_K = norm(matK - matK_recovered, 'fro');% Output


%% Computation 2

% = = = = = [ Input ] = = = = =
eps_sedumi = 0;% = sedumi.eps, desired accuracy of optimization, used in sdpsettings(). 
int_verbose = 0;
%filename_state_prepared = '../ImportFiles/tester_1qubit_state_withError.csv';
filename_state_prepared = '../ImportFiles/tester_1qubit_state.csv';
filename_povm_prepared = '../ImportFiles/tester_1qubit_povm.csv';
filename_state_tester = '../ImportFiles/tester_1qubit_state.csv';
filename_povm_tester = '../ImportFiles/tester_1qubit_povm.csv';
filename_schedule = '../ImportFiles/schedule_1qubit.csv';
filename_weight = '../ImportFiles/weight_2valued_uniform.csv';
% = = = = = = = = = = = = = = =

num_state = 4;
num_povm = 3; 
num_outcome = 2;

list_state_prepared    = FileImport_state(filename_state_prepared, dim, num_state);
list_povm_prepared     = FileImport_povm(filename_povm_prepared, dim, num_povm, num_outcome);
list_state_tester = FileImport_state(filename_state_tester, dim, num_state);
list_povm_tester  = FileImport_povm(filename_povm_tester, dim, num_povm, num_outcome);
list_schedule = csvread(filename_schedule);
list_weight   = FileImport_weight(filename_weight, num_outcome);

size_Choi = dim * dim;
label = 1;
matI2 = eye(dim);

Choi_Gk_prepared = Choi_from_HS(HS_Gk, basis_comp);
list_probDist = ListProbDist_QPT_v2( Choi_Gk_prepared, list_state_prepared, list_povm_prepared, list_schedule ); 
list_empiDist = list_probDist;

matD = MatD( list_state_tester, list_povm_tester, list_schedule, list_weight );
vecE = VecE( list_state_tester, list_povm_tester, list_schedule, list_weight, list_empiDist );
h    = ConstantH( list_weight, list_empiDist );

option            = sdpsettings('solver', 'sedumi');
option            = sdpsettings(option, 'sedumi.eps', eps_sedumi);
option            = sdpsettings(option, 'verbose', int_verbose);
% varChoi           = sdpvar(size_Choi, size_Choi, 'hermitian', 'complex');
% constraints       = [PartialTrace(varChoi, dim, dim, label) == matI2, varChoi >=0];
% objectiveFunction = WeightedSquaredDistance(varChoi, matD, vecE, h);

varHS           = sdpvar(size_Choi -1, size_Choi, 'full');
constraints       = [Choi_from_HS_parameterMatrix(varHS, basis_pauli_normalized) >=0];
objectiveFunction = WeightedSquaredDistance(HS_hermite_from_parameterMatrix(varHS), matD, vecE, h);

tic
optimize(constraints, objectiveFunction, option);
toc

obj_opt  = value(objectiveFunction);
% Choi_Gk_est = value(varChoi);
% diff_Choi_Gk = norm(Choi_Gk_est - Choi_Gk_prepared)

HS_Gk_est = value(varHS);
diff_HS_Gk = norm(HS_Gk_est - HS_Gk)

% HS_Gk_est = HS_from_Choi(Choi_Gk_est, basis_comp);
% eps = 10^(-9);
% HS_Lk_est = nonpv_log_matrix(HS_Gk_est, HS_L_target, k, eps);
% HS_L_est = HS_Lk_est ./ k;
% HS_G_est = expm(HS_L_est);
% 
% diff_L = norm(HS_L - HS_L_est, 'fro')% Output
% diff_G = norm(HS_G - HS_G_est, 'fro');% Output
% 
% [matH_est, matJ_est, matK_est] = dynamicsGenerator_from_HS_comp_Lindbladian(HS_L_est, basis_pauli_normalized);
% 
% diff_H = norm(matH - matH_est, 'fro');% Output
% diff_J = norm(matJ - matJ_est, 'fro');% Output
% diff_K = norm(matK - matK_est, 'fro');% Output

% 
% %% Computation 3: Monte Carlo Simulation
% list_Nrep = [100, 1000, 10000, 100000];
% Nave = 3;
% seed_x = 999;
% gene_x = 'twister';
% 
% num_Nrep = numel(list_Nrep);
% 
% 
% for i_ave = 1:Nave
%     seed_x = seed_x + i_ave;% + i_k;
%     set_list_empiDist = set_list_empiDist_from_list_probDist_list_Nrep(list_probDist, list_Nrep, seed_x, gene_x);
%     num_id    = size(set_list_empiDist, 2);
%     num_value = size(set_list_empiDist, 3);
% 
%     for i_Nrep = 1:num_Nrep
%         Nrep = list_Nrep(i_Nrep);
% 
%         list_empiDist = zeros(num_id, num_value);
%         for id = 1:num_id
%             for i_value = 1:num_value
%                 list_empiDist(id, i_value) = set_list_empiDist(i_Nrep, id, i_value);
%             end
%         end
% 
%         list_weight = list_weight_from_list_empiDist_case1(list_empiDist, Nrep);
%         %display(list_weight);
% 
% 
%         % = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
%         % 4.3 Calculation of D, vec(E), and h.
%         % = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
%         matD = MatD( list_state_tester, list_povm_tester, list_schedule, list_weight );
%         vecE = VecE( list_state_tester, list_povm_tester, list_schedule, list_weight, list_empiDist );
%         h    = ConstantH( list_weight, list_empiDist );
% 
%         % = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
%         % 4.4 Optimization
%         % = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
% 
%         % 4.4.1 Standard QPT
% 
%         option            = sdpsettings('solver', 'sedumi');
%         option            = sdpsettings(option, 'sedumi.eps', eps_sedumi);
%         option            = sdpsettings(option, 'verbose', int_verbose);
%         varChoi           = sdpvar(size_Choi, size_Choi, 'hermitian', 'complex');
%         constraints       = [PartialTrace(varChoi, dim, dim, label) == matI2, varChoi >=0];
%         objectiveFunction = WeightedSquaredDistance(varChoi, matD, vecE, h);
% 
%         tic
%         optimize(constraints, objectiveFunction, option);
%         toc
% 
%         obj_opt  = value(objectiveFunction);
%         Choi_Gk_est = value(varChoi);
%         %option.sedumi
%         
%         HS_Gk_est = HS_from_Choi(Choi_Gk_est, basis_comp);
%         eps = 10^(-9);
%         HS_Lk_est = nonpv_log_matrix(HS_Gk_est, HS_L_target, k, eps);
%         HS_L_est = HS_Lk_est ./ k;
%         HS_G_est = expm(HS_L_est);
% 
%         diff_L = norm(HS_L - HS_L_est, 'fro')% Output
%         diff_G = norm(HS_G - HS_G_est, 'fro');% Output
% 
%         [matH_est, matJ_est, matK_est] = dynamicsGenerator_from_HS_comp_Lindbladian(HS_L_est, basis_pauli_normalized);
% 
%         diff_H = norm(matH - matH_est, 'fro');% Output
%         diff_J = norm(matJ - matJ_est, 'fro');% Output
%         diff_K = norm(matK - matK_est, 'fro');% Output
%     end% i_Nrep
% end% i_Nave
% 
% 
% 
% %% 
% 
% 
% 
% 
