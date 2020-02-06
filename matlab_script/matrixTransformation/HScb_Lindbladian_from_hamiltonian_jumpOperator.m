function output = HScb_Lindbladian_from_hamiltonian_jumpOperator(H, list_c)
%HSCB_LINDBLADIAN_FROM_HAMILTONIAN_JUMPOPERATOR returns the Hilbert-Schmidt
%representation of the Lindbladian w.r.t. the computational basis.
%   H: Hamiltonian
%   list_c: list of jump operators
	HS_H = HScb_from_hamiltonian(H);
    HS_D = HScb_from_jumpOperator(list_c);
    HS = HS_H + HS_D;
    output = HS;
end

