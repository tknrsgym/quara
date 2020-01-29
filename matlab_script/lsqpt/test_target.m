clear
format long
% Target Gate
theta = 0.50 * pi;
vecH_target = [0.0; 0.50 * theta ; 0.0; 0.0];

HSgb_L_target = HSgb_L_from_vecH_1qubit(vecH_target);
HSgb_G_target = expm(HSgb_L_target)

% Prepared Gate
% Hamiltonian part
vecE = [0.0; 0.010; 0.0; 0.0];
vecH_prepared = vecH_target + vecE;

% Dissipator part
T1 = 100;% us
T2 = 100;% us
alpha = 0.20;
t = 20 * power(10, -3);% us

HScb_L_prepared  = HScb_L_model_rotation_BE99_1qubit(vecH_prepared, T1/t, T2/t, alpha);
HSgb_L_prepared  = HSgb_from_HScb_1qubit(HScb_L_prepared);
HSgb_G_prepared  = expm(HSgb_L_prepared)