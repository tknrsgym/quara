function output = HScb_K_from_matK_1qubit(matK)
%HSCB_K_FROM_MATK_1QUBIT この関数の概要をここに記述
%   詳細説明をここに記述
    % Pauli Matrices
    matPauli = matPauli_1qubit();
    
    % HScb_K
    HScb_K = zeros(4);
    for i_row = 1:3
        i_Pauli = i_row + 1;
        for j_col = 1:3
            j_Pauli = j_col + 1;
            mat = kron(matPauli(i_Pauli).mat, conj(matPauli(j_Pauli).mat)); 
            HScb_K = HScb_K + matK(i_row,j_col) .* mat;
        end
    end
    output = HScb_K;
end

