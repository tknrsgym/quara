function [output1, output2, output3] = dynamicsGenerator_from_HS_comp_Lindbladian(HS_L, basis)
%DYNAMICSGENERATOR_FROM_HS_COMP_LINDBLADIAN returns the triad of the
%dynamics generators from the Hilbert-Schmidt representation of a
%Linbladian with respect to the computational basis.
%   - basis is an orthoginal and normalized matrix basis with basis(1).mat
%   = I ./ sqrt(dim). 
%     
    matH = hamiltonian_from_HS_comp_Lindbladian(HS_L, basis);
    matJ = matJ_from_HS_comp_Lindbladian(HS_L, basis);
    matK = matK_from_HS_comp_Lindbladian(HS_L, basis);
    
    output1 = matH;
    output2 = matJ;
    output3 = matK;
end

