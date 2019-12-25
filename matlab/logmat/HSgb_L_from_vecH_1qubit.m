function output = HSgb_L_from_vecH_1qubit(vecH)
%HSGB_L_FROM_VECH_1QUBIT この関数の概要をここに記述
%   詳細説明をここに記述
    matH = matH_from_vecH_1qubit(vecH);
    HScb = HScb_H_from_matH(matH);
    HSgb = HSgb_from_HScb_1qubit(HScb);
    output = HSgb;
end

