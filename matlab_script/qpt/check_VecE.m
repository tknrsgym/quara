% = = = = = = = = = = = = = = = = = = = = = = = = = = =
% 0. System Information
% = = = = = = = = = = = = = = = = = = = = = = = = = = =

dim = 2;% 1-qubit system.
%dim = 4;% 2-qubit system.

size_Choi  = dim * dim;% size of Choi matrix, d^2 x d^2.
label = 1;
matI2 = eye(dim);


% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
% 1. Prepare Files of Tester States, Tester POVMs, IDs, 
%                     Weights, and Empirical Distributions. 
% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
filename_state = './ImportFiles/tester_1qubit_state.csv';
num_state = 4;

filename_povm = './ImportFiles/tester_1qubit_povm.csv';
num_povm    = 3;
num_outcome = 2;

filename_schedule = './ImportFiles/schedule.csv';

filename_weight = './ImportFiles/weight_2valued_uniform.csv';

filename_listEmpiDist = './ImportFiles/listEmpiDist_2valued.csv';



% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
% 2. Import Files of Tester States, Tester POVMs, IDs, Weights.
% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
list_state    = FileImport_state(filename_state, dim, num_state);
list_povm     = FileImport_povm(filename_povm, dim, num_povm, num_outcome);
list_schedule = csvread(filename_schedule);
list_weight   = FileImport_weight(filename_weight, num_outcome);
list_empiDist = csvread(filename_listEmpiDist);


% Check of VecE()
vecE = VecE( list_state, list_povm, list_schedule, list_weight, list_empiDist )