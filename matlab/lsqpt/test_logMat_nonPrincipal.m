% System Information
d = 2;

% Target Gate
theta = 0.50 * pi;
vecH_target = [0.0; 0.50 * theta ; 0.0; 0.0];

% Prepared Gate
% Hamiltonian part
vecE = [0.0; 0.010; 0.0; 0.0];
vecH_prepared = vecH_target + vecE;

% Dissipator part
T1 = 10;% us
T2 = 10;% us
alpha = 0.20;
t = 20 * power(10, -3);% us

% Amplification information
k = 27;
eps = 0.10;

HScb_L_prepared = HScb_L_model_rotation_BE99_1qubit(vecH_prepared, T1/t, T2/t, alpha);
HSgb_L_prepared = HSgb_from_HScb_1qubit(HScb_L_prepared);
HSgb_kL_prepared = k.*HSgb_L_prepared;% used for performance evaluation later
HSgb_Gk_prepared = expm(HSgb_kL_prepared);


nPlogMat_HSgb_Gk = logMat_nonPrincipal_1qubit(vecH_target, HSgb_Gk_prepared, k, eps);
error_recover = norm(nPlogMat_HSgb_Gk ./k - HSgb_L_prepared)