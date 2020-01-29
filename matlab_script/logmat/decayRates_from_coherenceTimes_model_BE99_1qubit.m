function [output1, output2, output3] = decayRates_from_coherenceTimes_model_BE99_1qubit(T1, T2, alpha)
%DECAYRATES_FROM_COHERENCETIMES_MODEL_BE99_1QUBIT 
% Ref. Briegel & Englert, Phs. Rev. A 47, 3311 (1999).  
% Note of Sugiyama: 190605 on iPad Pro.
%   - T1: energy relaxation time
%   - T2: phase relaxation time
%   - alpha: Thermal population   
    Gamma_plus  = 0.50 * (1.0 + alpha) / T1;
    Gamma_minus = 0.50 * (1.0 - alpha) / T1;
    Gamma_zero  = 0.50/T2 - 0.25/T1;
    
    output1 = Gamma_plus;
    output2 = Gamma_minus;
    output3 = Gamma_zero;
end

