function [Choi, obj_value, list_option] = qpt(dim, list_state, list_povm, list_schedule, list_weight, list_empiDist, eps_sedumi, int_verbose)
%QPT ‚±‚ÌŠÖ”‚ÌŠT—v‚ð‚±‚±‚É‹Lq
%  - dim: dimension of the system
%
%  - list_state: list of density matrices. 
%
%  - list_povm: list of POVMs. 
%
%  - list_schedule: list of pair of labels for an input state and for measurement
%
%  - list_weight: list of weight matrices
%
%  - list_empiDist: list of empirical distributions
%
%  - eps_sedumi: desired accuracy used for SeDuMi. 
%                If eps_sedumi = 0, then SeDuMi runs as long as it can make progress.
%
%  - int_verbose: By setting verbose to 0, the solvers will run with minimal display. 
%                 By increasing the value, the display level is controlled 
%                 (typically 1 gives modest display level while 2 gives an awful amount of information printed). 
%  
%  Reference
%   1. About sdpsettings of YALMIP
%    https://yalmip.github.io/command/sdpsettings/
%   2. About use of SeDuMi through YALMIP
%    https://yalmip.github.io/solver/sedumi/
%

    % = = = = = = = = = = = = = = = = = = = = = = = = = = =
    % System Information
    % = = = = = = = = = = = = = = = = = = = = = = = = = = =
    size_Choi  = dim * dim;% size of Choi matrix, d^2 x d^2.
    label = 1;
    matI2 = eye(dim);

    % = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    % Calculation of D, vec(E), and h.
    % = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    matD = MatD( list_state, list_povm, list_schedule, list_weight );
    vecE = VecE( list_state, list_povm, list_schedule, list_weight, list_empiDist );
    h    = ConstantH( list_weight, list_empiDist );

    % = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    % Optimization using YALMIP and SeDuMi
    % = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    option            = sdpsettings('solver', 'sedumi');
    option            = sdpsettings(option, 'sedumi.eps', eps_sedumi);
    option            = sdpsettings(option, 'verbose', int_verbose);
    varChoi           = sdpvar(size_Choi, size_Choi, 'hermitian', 'complex');
    constraints       = [PartialTrace(varChoi, dim, dim, label) == matI2, varChoi >=0];
    objectiveFunction = WeightedSquaredDistance(varChoi, matD, vecE, h);

    optimize(constraints, objectiveFunction, option);

    % = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    % Optimization results
    % = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =    
    obj_value = value(objectiveFunction);
    Choi      = value(varChoi);
    list_option    = option.sedumi;

end

