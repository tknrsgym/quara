function output = matH_from_vecH_1qubit(vecH)
%MATH_FROM_VECH_1QUBIT Calculate Hamiltonian matrix from Hamiltonian Bloch
%vector.
% 
    % Pauli Matrices
    matPauli = matPauli_1qubit();
    
    % matH
    matH = zeros(2);
    for i = 1:4 
        matH = matH + vecH(i) .* matPauli(i).mat;
    end
   
    output = matH; 
end

