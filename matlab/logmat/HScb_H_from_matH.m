function output = HScb_H_from_matH(matH)
%HSCB_H_FROM_MATH この関数の概要をここに記述
%   -i { H \otimes I - I \otimes transpose(H) }
%   詳細説明をここに記述
    [row, col] = size(matH);
    matI = eye(row);
    
    HScb_H = zeros(row * col);
    HScb_H = HScb_H + kron(matH, matI);
    HScb_H = HScb_H - kron(matI, transpose(matH));
    HScb_H = -i .* HScb_H;

    output = HScb_H;
end

