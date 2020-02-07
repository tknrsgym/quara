function output = averageGateFidelity_from_HS_Lindbladian(L0, L1, dim)
%AVERAGEGATEFIDELITY_FROM_HS_Lindbladian returns the value of average date
%fidelity
%   - L0: HS representation of an ideal Lindbladian including Hamiltonian-part only
%   - L1: HS representation of a possible noisy Lindbladian
    Gate0 = expm(L0);
    Gate1 = expm(L1);
    AGF = averageGateFidelity_from_HS_Gate(Gate0, Gate1, dim);
    output = AGF;
end

