%% 190620 (Thu.) test of non-principal matrix logarithm function
% 

format long
theta = 0.50 * pi;
k = 3;
vecH_target = [0.0; 0.50 * theta ; 0.0; 0.0];

%% Eigensystem analysis of Target

% Validity Check of k
% isValid_k = isValid_k_for_vecH_1qubit(k, vecH);

[eigsys_kL_target, recov_kL_target] = eigsys_recov_HSgb_kL_from_vecH_k_1qubit(vecH_target, k);



%% Eigensystem analysis of (Pseudo) Prepared

% Hamiltonian part
vecE = [0.0; 0.010; 0.0; 0.0];
vecH_prepared = vecH_target + vecE;

% Dissipator part
T1 = 10;% us
T2 = 10;% us
alpha = 0.20;
t = 20 * power(10, -3);% us

HScb_L_prepared = HScb_L_model_rotation_BE99_1qubit(vecH_prepared, T1/t, T2/t, alpha);
HSgb_L_prepared = HSgb_from_HScb_1qubit(HScb_L_prepared);
HSgb_kL_prepared = k.*HSgb_L_prepared;% used for performance evaluation later
HSgb_Gk_prepared = expm(HSgb_kL_prepared);
PlnGk_prepared = logm(HSgb_Gk_prepared);
eigsys_PlnGk_prepared = eigsys_matA(PlnGk_prepared);


%% Index Correspondence
gamma = 0.5 * k;
idx_eval_target_from_prepared = index_correspondence(eigsys_kL_target, eigsys_PlnGk_prepared, gamma);


%% Phase recovering

HSgb_kL_recovered = matA_with_recoveredPhase_from_eigsys(eigsys_PlnGk_prepared, recov_kL_target, idx_eval_target_from_prepared);

error_recover = norm(HSgb_kL_recovered ./k - HSgb_L_prepared)


%% Debug log
% 190620 Thu. 
%  - error_recover becomes very large when k = 17, 25, 101 is chosen.
% theta = 0.50 * pi;
% k = 17;% 25;% 101;
% vecH_target = [0.0; 0.50 * theta ; 0.0; 0.0];
% 
% % Hamiltonian part
% vecE = [0.0; 0.010; 0.0; 0.0];
% vecH_prepared = vecH_target + vecE;
% % Dissipator part
% T1 = 10;% us
% T2 = 10;% us
% alpha = 0.20;
% t = 20 * power(10, -3);% us
%
% 190621 Fri.

