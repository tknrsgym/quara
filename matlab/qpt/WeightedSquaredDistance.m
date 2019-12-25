function res = WeightedSquaredDistance(matC, D, vecE, h)
%WeightedSquaredDistance is a function for calculating a weighted squared
%2-norm distance
% - matC: Choi matrix
% - D: D matrix
% - vecE: 
% - h
    vecC = matC(:);
    
    term1 = vecC' * D * vecC;
    term2 = vecE' * vecC;
    term3 = h;

    res = term1 - 2.0 * term2 + h;
end
