function output = HScb_J_from_matJ(matJ)
%HSCB_J_FROM_MATJ この関数の概要をここに記述
%   J \otimes I - I \otimes transpose(J) }
%   詳細説明をここに記述
    [row, col] = size(matJ);
    matI = eye(row);
    
    HScb_J = zeros(row * col);
    HScb_J = HScb_J + kron(matJ, matI);
    HScb_J = HScb_J + kron(matI, transpose(matJ));

    output = HScb_J;
end