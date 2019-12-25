format long
t = 40 * power(10.0, -3);% us



%% Target gate
%
vecH = [0.0; 0.250 * pi / t; 0.0; 0.0];
matH = matH_from_vecH_1qubit(vecH);

HScb_L_target = HScb_H_from_matH(matH)
HSgb_L_target = HSgb_from_HScb_1qubit(HScb_L_target)

HScb_G_target = expm(t .* HScb_L_target)
HSgb_G_target = expm(t .* HSgb_L_target)

%Sym_HSgb_G_target = sym(HSgb_G_target);
%[Sym_V_G_target, Sym_J_G_target] = jordan(Sym_HSgb_G_target);
%Double_V_G_target = double(Sym_V_G_target)
%Double_J_G_target = double(Sym_J_G_target)
%
%cond = J_G_target == V_G_target \ HSgb_G_target * V_G_target;
%isAlways(cond)
%B = Double_V_G_target * Double_J_G_target * inv(Double_V_G_target)

[V_G_target, D_G_target] = eig(HSgb_G_target)
V_G_target * D_G_target * V_G_target'

[V_L_target, D_L_target] = eig(HSgb_L_target)

%% Prepared Gate
%
vecH_prepared = [0.0; 0.251 * pi / t; 0.0; 0.0];
%vecH_prepared = [0.0; 0.0; 0.0; 0.0];

T1 = 50;% us
T2 = 40;% us
alpha = 0.010;

[Gamma_plus, Gamma_minus, Gamma_zero] = decayRates_from_coherenceTimes_model_BE99_1qubit(T1, T2, alpha);

L_cb_prepared = HScb_L_model_rotation_BE99_1qubit(vecH_prepared, T1, T2, alpha);
L_gb_prepared = HSgb_from_HScb_1qubit(L_cb_prepared)

G_cb_prepared = expm(t .* L_cb_prepared);
G_gb_prepared = expm(t .* L_gb_prepared)

%vec_J_prepared = vecJ_from_model_BE99_1qubit(T1, T2, alpha);
%matJ_prepared  = matJ_from_model_BE99_1qubit(T1, T2, alpha);
%matK_prepared  = matK_from_model_BE99_1qubit(T1, T2, alpha);

% Check of TPness on Dynamics Generator
%matJ_TP = matJ_from_matK_ifTP_1qubit(matK_prepared);
%matJ_TP - matJ_prepared;

% Jordan decomposition of G^prepared
Sym_G_gb_prepared = sym(G_gb_prepared);
[Sym_V_G_prepared, Sym_J_G_prepared] = jordan(Sym_G_gb_prepared);
Double_V_G_prepared = double(Sym_V_G_prepared)
Double_J_G_prepared = double(Sym_J_G_prepared)
Double_V_G_prepared * Double_J_G_prepared * inv(Double_V_G_prepared) - G_gb_prepared

% Jordan decomposition of L^prepared
Sym_L_gb_prepared = sym(L_gb_prepared);
[Sym_V_L_prepared, Sym_J_L_prepared] = jordan(Sym_L_gb_prepared);
Double_V_L_prepared = double(Sym_V_L_prepared)
Double_J_L_prepared = double(Sym_J_L_prepared)
Double_V_L_prepared * Double_J_L_prepared * inv(Double_V_L_prepared) - L_gb_prepared


% Eigenvalue decomposition?
[V_L_prepared, D_L_prepared] = eig(L_gb_prepared)
V_L_prepared * D_L_prepared * inv(V_L_prepared) - L_gb_prepared

[V_G_prepared, D_G_prepared] = eig(G_gb_prepared)
V_G_prepared * D_G_prepared * inv(V_G_prepared) - G_gb_prepared

% 
%% 

