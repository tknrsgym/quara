function output = matPauliMatPauliT_1qubit()
%MATPAULIMATPAULIT_1QUBIT ���̊֐��̊T�v�������ɋL�q
%   �ڍא����������ɋL�q
    matPauli = matPauli_1qubit();
    
    for i_row = 0:3
        i_Pauli = i_row + 1;
        for j_col = 0:3
            j_Pauli = j_col + 1;
            matPauliMatPauliT(i_Pauli, j_Pauli).mat = kron(matPauli(i_Pauli).mat, conj(matPauli(j_Pauli).mat)); 
        end
    end
    output = matPauliMatPauliT;
end

