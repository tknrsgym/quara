function output = hamiltonian_from_HS_comp_Lindbladian(HS_L, basis)
%HAMILTONIAN_FROM_HS_COMP_LINDBLADIAN returns Hamiltonian from the
%Hilbert-Schmidt representation of a Lindbladian.
%   - basis is an orthoginal and normallized matrix with basis(1).mat = I /
%   sqrt(dim).
    vecH = hamiltonianCoefficients_from_HS_comp_Lindbladian(HS_L, basis);
    matH = matrix_from_coefficients_basis(vecH, basis);
    output = matH;
end

