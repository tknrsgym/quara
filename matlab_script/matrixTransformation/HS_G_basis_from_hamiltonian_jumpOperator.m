function output = HS_G_basis_from_hamiltonian_jumpOperator(H, list_c, basis_comp, basis)
%HS_G_BASIS_FROM_HAMILTONIAN_JUMPOPERATOR 
%   - H: Hamiltonian
%   - list_c: a list of jump operator
%   - basis_comp : the computational basis
%   - basis: matrix basis for the output Hilbert-Schmidt representation
    HScb_L = HScb_Lindbladian_from_hamiltonian_jumpOperator(H, list_c);
    HScb_G = expm(HScb_L);
    matU = matU_HS_basisTransform_to_B1_from_B2(basis, basis_comp);
    HS   = matU * HScb_G * ctranspose(matU);

    output = HS;
end

