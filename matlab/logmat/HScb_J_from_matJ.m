function output = HScb_J_from_matJ(matJ)
%HSCB_J_FROM_MATJ ���̊֐��̊T�v�������ɋL�q
%   J \otimes I - I \otimes transpose(J) }
%   �ڍא����������ɋL�q
    [row, col] = size(matJ);
    matI = eye(row);
    
    HScb_J = zeros(row * col);
    HScb_J = HScb_J + kron(matJ, matI);
    HScb_J = HScb_J + kron(matI, transpose(matJ));

    output = HScb_J;
end