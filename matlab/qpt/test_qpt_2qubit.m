% Start of test_qpt_1qubit.m
format long
char_precision = '%.15e';

% = = = = = = = = = = = = = = = = = = = = = = = = = = =
% 0. System Information
% = = = = = = = = = = = = = = = = = = = = = = = = = = =
dim         = 4;% 2-qubit system.
eps_sedumi  = 0;
int_verbose = 1;% 0: no stdout, 1: 

% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
% 2. Import Files of Tester States, Tester POVMs, IDs, Weights.
% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
num_state   = 16;
num_povm    = 9;
num_outcome = 4;

path_quara = '../../quara/';
path_importFolder = strcat(path_quara, 'tests/data/');

filename_state        = strcat(path_importFolder, 'tester_2qubit_state.csv');
filename_povm         = strcat(path_importFolder, 'tester_2qubit_povm.csv');
filename_weight       = strcat(path_importFolder, 'weight_4valued_uniform.csv');
filename_schedule     = strcat(path_importFolder, 'schedule_2qubit.csv');
filename_listEmpiDist = strcat(path_importFolder, 'listEmpiDist_4valued.csv');  

list_state    = FileImport_state(filename_state, dim, num_state);
list_povm     = FileImport_povm(filename_povm, dim, num_povm, num_outcome);
list_schedule = csvread(filename_schedule);
list_weight   = FileImport_weight(filename_weight, num_outcome);
list_empiDist = csvread(filename_listEmpiDist);

% = = = = = = = = = = = = = = =
% Calculation of QPT estimate
% = = = = = = = = = = = = = = =
[Choi, obj_value, option] = qpt(dim, list_state, list_povm, list_schedule, list_weight, list_empiDist, eps_sedumi, int_verbose);

Choi
obj_value
option

% End of test_qpt_1qubit.m