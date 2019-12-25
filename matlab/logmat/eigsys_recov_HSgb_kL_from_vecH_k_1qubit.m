function [output1,output2] = eigsys_recov_HSgb_kL_from_vecH_k_1qubit(vecH, k)
%EIGSYS_RECOV_KL_FROM_VECH_K_1QUBIT この関数の概要をここに記述
%   詳細説明をここに記述

    % Eigensystem of kL
    veckH = k .* vecH;
    HSgb_kL_comp = HSgb_L_from_vecH_1qubit(veckH);
    HSgb_kL = real(HSgb_kL_comp);
    eigsys_kL = eigsys_matA(HSgb_kL);

    % Recovering information
    sz = size(HSgb_kL);
    for i = 1:sz
        lambda_imag = imag(eigsys_kL(i).eval);
        [recov_kL(i).periodID, recov_kL(i).residual] = mod_2pi_type2(lambda_imag);
    end

    output1 = eigsys_kL;
    output2 = recov_kL;
end

