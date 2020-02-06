function output = matJ_from_HS_comp_Lindbladian(HS_L, basis)
%MATJ_FROM_HS_COMP_LINDBLADIAN returns the J-matrix from the
%Hilbert-Schmidt representation of a Lindbladian.
%   - basis is an orthoginal and normallized matrix with basis(1).mat = I /
%   sqrt(dim).
    vecJ = matJCoefficients_from_HS_comp_Lindbladian(HS_L, basis);
    matJ = matrix_from_coefficients_basis(vecJ, basis);
    output = matJ;
end

