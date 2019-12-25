function output = vecJ_from_model_BE99_1qubit(T1, T2, alpha)
%VECJ_FROM_MODEL_BE99_1QUBIT この関数の概要をここに記述
%   詳細説明をここに記述
    [Gamma_plus, Gamma_minus, Gamma_zero] = decayRates_from_coherenceTimes_model_BE99_1qubit(T1, T2, alpha);
    
    vecJ(1) = -0.25 * (Gamma_plus + Gamma_minus + 2.0*Gamma_zero);
    vecJ(2) = 0.0;
    vecJ(3) = 0.0;
    vecJ(4) = -0.25 * (Gamma_plus - Gamma_minus);
    
    output = vecJ;
end

