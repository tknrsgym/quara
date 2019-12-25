function output = logMat_nonPrincipal_1qubit(vecH_target, HSgb_k, k, eps)
%LOGMAT_NONPRINCIPLE_1QUBIT Ç±ÇÃä÷êîÇÃäTóvÇÇ±Ç±Ç…ãLèq
%   

    %% Eigensystem analysis of Target
    [eigsys_kL_target, recov_kL_target] = eigsys_recov_HSgb_kL_from_vecH_k_1qubit(vecH_target, k);

    %% Eigensystem analysis of Input
    PlogMat = logm(HSgb_k);
    eigsys_PlogMat = eigsys_matA(PlogMat);

    %% Index Correspondence
    gamma = 0.5 * k;
    idx_eval_target_from_input = index_correspondence(eigsys_kL_target, eigsys_PlogMat, gamma);


    %% Phase recovering
    HSgb_kL_recovered = matA_with_recoveredPhase_from_eigsys(eigsys_PlogMat, recov_kL_target, idx_eval_target_from_input);

    %error_recover = norm(HSgb_kL_recovered ./k - HSgb_L_prepared)

    output = HSgb_kL_recovered;
end

