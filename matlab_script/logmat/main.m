%format long
char_precision = '%.15e';

%% Amplification strength
k = 6;


%% Set of Target Generator
vecH_target = [0.0; 0.25 * pi ; 0.0; 0.0];
HSgb_L_target_comp = HSgb_L_from_vecH_1qubit(vecH_target);
HSgb_L_target = real(HSgb_L_target_comp);
%HSgb_G_target = expm(HSgb_L_target);

[V_L_target, D_L_target] = eig(HSgb_L_target);
[Ds_L_target, ind_L_target] = sort(diag(D_L_target), 'descend')
Vs_L_target = V_L_target(:,ind_L_target)

[vec_n, vec_Delta] = vec_n_from_eval_k(Ds_L_target, k)

evecs_L_target = Vs_L_target(:,2);
HSgb_L_target * evecs_L_target;

%% Set of Prepared Generator
vecH_prepared = [0.0; 0.50 * pi ; 0.0; 0.0];
vecJ_prepared = [0.0; 0.0; 0.0; 0.0];
matK_prepared = [0.0, 0.0, 0.0;
                 0.0, 0.0, 0.0;
                 0.0, 0.0, 0.0];

matL_prepared = zeros(4);
%matL_prepared = matL_from_HJK(matL_prepared, vecH_prepared, vecJ_prepared, matK_prepared);            


%% Pauli-Liouville representation of Gates
%matGk_target = expm(k .* matL_target);
%matGk_prepared = expm(k .* matL_prepared);





%% Calculation of non-principal matrix logarithm
%flag = 'jordan';% flag = 'schur';
%matL_estimate = logMat_from_target_k(matGk_prepared, matL_target, k, flag);
%matG_estimate = expm(matL_estimate);


%% Evaluation of Computation Accuracy

%matL_estimate - matL_prepared;
%[vecH_estimate, vecJ_estimate, matK_estimate] = generator_from_matL(matL_estimate);

%matG_estimate - matG_prepared

