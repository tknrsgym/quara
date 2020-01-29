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
filename_state = './ImportFiles/tester_1qubit_state.txt';
num_state = 4;

filename_povm = './ImportFiles/tester_1qubit_povm.txt';
num_povm    = 3;
num_outcome = 2;

filename_schedule = './ImportFiles/schedule.txt';



% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
% 2. Import Files of Tester States, Tester POVMs, IDs, 
%                 Weights, and Empirical Distributions.
% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
list_state    = FileImport_state(filename_state, dim, num_state);
list_povm     = FileImport_povm(filename_povm, dim, num_povm, num_outcome);
list_schedule = csvread(filename_schedule);

num_schedule = size(list_schedule); 

% Check of ListProbDist_QPT()
Choi = [1, 0, 0, 1; 0, 0, 0, 0; 0, 0, 0, 0; 1, 0, 0, 1];
list_probDist = ListProbDist_QPT( Choi, list_state, list_povm, list_schedule );

for id = 1:num_schedule
     id
     list_probDist(id).vec
end