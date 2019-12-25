% Start of main_qpt.m
format long
char_precision = '%.15e';

% = = = = = = = = = = = = = = = = = = = = = = = = = = =
% 0. System Information
% = = = = = = = = = = = = = = = = = = = = = = = = = = =

dim = 2;% 1-qubit system.
%dim = 4;% 2-qubit system.

size_Choi  = dim * dim;% size of Choi matrix, d^2 x d^2.
label = 1;
matI2 = eye(dim);

eps = 1e-8;% = sedumi.eps, desired accuracy of optimization, used in sdpsettings(). 


% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
% 1. Prepare Files of Tester States, Tester POVMs, IDs, 
%                     Weights, and Empirical Distributions. 
% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
filename_state = './ImportFiles/tester_1qubit_state.csv';
num_state = FilePreparation_1qubit_state(filename_state, char_precision);

filename_povm = './ImportFiles/tester_1qubit_povm.csv';
[num_povm, num_outcome] = FilePreparation_1qubit_povm(filename_povm, char_precision);

filename_schedule = './ImportFiles/schedule.csv';
num_schedule = FilePreparation_1qubit_schedule(filename_schedule, num_state, num_povm);

filename_weight = './ImportFiles/weight_2valued_uniform.csv';
FilePreparation_1qubit_weight_2valued_uniform(filename_weight, filename_schedule, char_precision);


% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
% 2. Import Files of Tester States, Tester POVMs, IDs, Weights.
% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
list_state    = FileImport_state(filename_state, dim, num_state);
list_povm     = FileImport_povm(filename_povm, dim, num_povm, num_outcome);
list_schedule = csvread(filename_schedule);
list_weight   = FileImport_weight(filename_weight, num_outcome);



% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
% 3. Import File of Empirical Distributions.
% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
filename_listEmpiDist = './ImportFiles/listEmpiDist_2valued.csv';

% Preparation of data file, for debugging and numerical simulation
if true
    if false
        p1 = 0.01;
        p2 = 0.02;
        p3 = 0.03;
        Choi = Choi_1qubit_stochasticPauli(p1, p2, p3);
    end
    if true
        alpha = 0.1;
        beta  = -0.15;
        theta = 0.3;
        Choi = Choi_1qubit_unitary(alpha, beta, theta); 
    end
    FilePreparation_1qubit_listProbDist( filename_listEmpiDist, Choi, list_state, list_povm, list_schedule, char_precision );% 
    %FilePreparation_1qubit_listEmpiDist_sampling(Choi, list_state, list_povm, %list_schedule, num_sampling, seed, char_precision);% Not implemented yet! To be implemented!
end
    
list_empiDist = csvread(filename_listEmpiDist);


% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
% 4. Calculation of D, vec(E), and h.
% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

% 4.1 D
matD = MatD( list_state, list_povm, list_schedule, list_weight );


% 4.2 vec(E)
vecE = VecE( list_state, list_povm, list_schedule, list_weight, list_empiDist );

% 4.3 h
h = ConstantH( list_weight, list_empiDist );


% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
% 5. Optimization
% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

option            = sdpsettings('solver', 'sedumi');
option            = sdpsettings(option, 'sedumi.eps', eps);
varChoi           = sdpvar(size_Choi, size_Choi, 'hermitian', 'complex');
constraints       = [PartialTrace(varChoi, dim, dim, label) == matI2, varChoi >=0];
objectiveFunction = WeightedSquaredDistance(varChoi, matD, vecE, h);

tic
optimize(constraints, objectiveFunction, option);
toc

%obj_opt  = value(objectiveFunction)
Choi_opt = value(varChoi)
%option.sedumi





% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
% 6. Result Analysis
% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

% 6.1 Physicality Check
%eig(Choi_opt)

% 6.2 Dynamics Generator Analysis

% 6.3 Goodness of Fit

% 6.4 Estimation Error

%Choi
norm(Choi - Choi_opt)

% 6.5 Error Bar by Bootstrap

% 6.6 Make a report


% End of main_qpt.m