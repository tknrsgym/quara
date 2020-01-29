function output = HSgb_from_HScb_1qubit(HS_cb)
%HSGB_FROM_HSCB_1QUBIT この関数の概要をここに記述
%   詳細説明をここに記述
    matV = [1.0, 0.0, 0.0, 1.0;
            0.0, 1.0, 1.0, 0.0;
            0.0, -i, i, 0.0;
            1.0, 0.0, 0.0, -1.0];
    matV = (1.0 / sqrt(2.0)) .* matV;
    HSgb = matV * HS_cb * ctranspose(matV);
    
    output = HSgb;
end

