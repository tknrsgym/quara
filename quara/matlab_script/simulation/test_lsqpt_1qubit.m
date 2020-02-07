clear;
format long;
char_precision = '%.15e';

%% 
dim         = 2;
num_state   = 4;
num_povm    = 3;
num_outcome = 2;
k           = 3;

% = = = = = [ Input ] = = = = =
eps_sedumi = 0;% = sedumi.eps, desired accuracy of optimization, used in sdpsettings(). 
int_verbose = 0;
filename_state_tester = '../ImportFiles/tester_1qubit_state.csv';
filename_povm_tester  = '../ImportFiles/tester_1qubit_povm.csv';
filename_schedule     = '../ImportFiles/schedule_1qubit.csv';
filename_weight       = '../ImportFiles/weight_2valued_uniform.csv';
filename_listEmpiDist = '../ImportFiles/listEmpiDist_2valued_k3.csv';
filename_matL0        = '../ImportFiles/matL0_1qubit_X90.csv';
% = = = = = = = = = = = = = = =

list_schedule = csvread(filename_schedule);
list_weight   = FileImport_weight(filename_weight, num_outcome);

doDataPreparation = true;
if doDataPreparation == true
    filename_state_prepared = '../ImportFiles/tester_1qubit_state.csv';
    filename_povm_prepared  = '../ImportFiles/tester_1qubit_povm.csv';
    list_state_prepared     = FileImport_state(filename_state_prepared, dim, num_state);
    list_povm_prepared      = FileImport_povm(filename_povm_prepared, dim, num_povm, num_outcome);
    
    % 1-qubit
    dim = 2;
    str = "X90";
    vecH_error  = [0.0, 0.02, 0.01, 0.03];
    vecGamma = [0.01, 0.012, 0.014];
    k = 3;
    % = = = = = = = = = = = = = = = 

    basis_pauli_normalized = matrixBasis_1qubit_pauli_normalized();
    basis_comp = matrixBasis_1qubit_comp();

    matH_target = hamiltonian_1qubit_gate_target(str);
    HS_L_target = HScb_from_hamiltonian(matH_target);
    FilePreparation_matrix(HS_L_target, filename_matL0, char_precision);

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
    Choi = Choi_from_HS(HS_Gk, basis_comp);
    
    FilePreparation_1qubit_listProbDist( filename_listEmpiDist, Choi, list_state_prepared, list_povm_prepared, list_schedule, char_precision );% 
end

list_empiDist     = csvread(filename_listEmpiDist);
list_state_tester = FileImport_state(filename_state_tester, dim, num_state);
list_povm_tester  = FileImport_povm(filename_povm_tester, dim, num_povm, num_outcome);
matL0             = csvread(filename_matL0);
eps_logmat = 10^(-10);

[Choi_est, obj_value] = lsqpt_ChoiBased(dim, list_state_tester, list_povm_tester, list_schedule, list_weight, list_empiDist, eps_sedumi, int_verbose, k, matL0, eps_logmat);

Choi_est
obj_value


