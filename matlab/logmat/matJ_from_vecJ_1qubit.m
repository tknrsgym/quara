function output = matJ_from_vecJ_1qubit(vecJ)
%MATJ_FROM_VECJ_1QUBIT returns matJ
%   
    % Pauli Matrices
    matPauli = matPauli_1qubit();
    
    % matH
    matJ = zeros(2);
    for i = 1:4 
        matJ = matJ + vecJ(i) .* matPauli(i).mat;
    end
   
    output = matJ;
end

