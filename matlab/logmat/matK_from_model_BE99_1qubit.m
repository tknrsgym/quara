function output = matK_from_model_BE99_1qubit(T1, T2, alpha)
%MATK_FROM_MODEL_BE99_1QUBIT_1 この関数の概要をここに記述
%   詳細説明をここに記述
    [Gamma_plus, Gamma_minus, Gamma_zero] = decayRates_from_coherenceTimes_model_BE99_1qubit(T1, T2, alpha);

    matK = zeros(3);
    matK(1,1) = Gamma_plus + Gamma_minus;
    matK(1,2) = i * (Gamma_plus - Gamma_minus);
    matK(2,1) = -i * (Gamma_plus - Gamma_minus);
    matK(2,2) = Gamma_plus + Gamma_minus;
    matK(3,3) = 4.0*Gamma_zero;
    
    matK = 0.25 * matK;
    
    output = matK;
end

