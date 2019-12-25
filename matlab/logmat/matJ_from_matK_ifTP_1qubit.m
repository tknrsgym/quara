function output = matJ_from_matK_ifTP_1qubit(matK)
%MATJ_FROM_MATK_IFTP_1QUBIT ‚±‚ÌŠÖ”‚ÌŠT—v‚ğ‚±‚±‚É‹Lq
%   If the dynamics is trace-preserving, the generator must satisfy
%      matJ = \frac{1}{2}\sum_{i,j}K_{ij} lambda_{j}^{\dagger} lambda_{i},
%      where i and j are for X, Y, Z (not including for I).
    matPauli = matPauli_1qubit();
    matJ = zeros(2);
    for i = 1:3
        for j = 1:3
            mat = (matPauli(j+1).mat)' * matPauli(i+1).mat;  
            matJ = matJ + matK(i,j) .* mat;
        end
    end
    matJ = -0.50 * matJ;
    output = matJ;
end

