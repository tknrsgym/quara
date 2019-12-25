theta = 0.50 * pi;
k = 9;

%% Eigensystem analysis of Target

veckH = k .* [0.0; 0.50 * theta ; 0.0; 0.0];

% Validity Check of k
% isValid_k = isValid_k_for_vecH_1qubit(k, vecH);

%[eigsys_kL, recov_kL] = eigsys_recov_kL_from_vecH_k_1qubit(vecH, k);

HSgb_kL_comp = HSgb_L_from_vecH_1qubit(veckH);
HSgb_kL = real(HSgb_kL_comp);
[V_kL, D_kL] = eig(HSgb_kL);

[Ds_kL, ind_kL] = sort(diag(D_kL), 'descend');
Vs_kL = V_kL(:,ind_kL);
for i = 1:4
    esys_kL(i).eval = Ds_kL(i);
    esys_kL(i).revec = Vs_kL(:,i);
end


%% Calculation of Period and Residual

for i = 1:4
    lambda_imag = imag(esys_kL(i).eval);
    [phase(i).periodID, phase(i).residual] = mod_2pi_type2(lambda_imag);
    lambda_out = 2.0 * pi * phase(i).periodID + phase(i).residual;
end




%% Eigensystem analysis of (Pseudo) Prepared

HSgb_Gk = expm(HSgb_kL);% Input
PlnGk = logm(HSgb_Gk);
% eigsys_PlnGk = eigsys_matA(PlnGk);


[V_PlnGk, D_PlnGk] = eig(PlnGk);

[Ds_PlnGk, ind_PlnGk] = sort(diag(D_PlnGk), 'descend');
Vs_PlnGk = V_PlnGk(:,ind_PlnGk);
for i = 1:4
    esys_PlnGk(i).eval = Ds_PlnGk(i);
    esys_PlnGk(i).revec = Vs_PlnGk(:,i);
end




%% Index Correspondence

% idx_eval_target_from_prepared = idx_eval_eigsys1_from_eigsys2(eigsys1, eigsys2);

sz = 4;

for i = 1:sz
    idx_eval_target_from_prepared(i) = -1;
end% i

overlap_array(1:4,1:4) = -1.0;
for i = 1:sz
    vec1 = esys_kL(i).revec;
    for j = 1:sz
        vec2 = esys_PlnGk(j).revec; 
        ip = dot(vec1, vec2);
        overlap = abs(ip);
        overlap_array(i,j) = overlap;
        threshold = 0.70 * norm(vec2);
        if (overlap > threshold)
            idx_eval_target_from_prepared(j) = i;
            break;
        end
    end% j
end% i

%overlap_array;
%idx_eval_target_from_prepared;

%% Phase recovering

% HSgb_kL_recovered = matA_with_recoveredPhase_from_eigsys(eigsys, recov, index_correspondence);

diag_new = zeros(4);
for i = 1:4
    idx = idx_eval_target_from_prepared(i); 
    if (idx > 0)
        modification = 2.0 .* pi .* 1.0i .* phase(idx).periodID;
    end
    diag_new(i,i) = esys_PlnGk(i).eval + modification;  
end
HSgb_kL_recovered = Vs_PlnGk * diag_new * inv(Vs_PlnGk);% Output

norm(HSgb_kL_recovered - HSgb_kL)




