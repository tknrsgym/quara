function output = HScb_L_from_vecHvecJmatK_1qubit(vecH, vecJ, matK)
%HSCB_L_FROM_VECHVECJMATK_1QUBIT この関数の概要をここに記述
%   詳細説明をここに記述
    
    % H
    matH = matH_from_vecH_1qubit(vecH);
    HScb_H = HScb_H_from_matH(matH);
    
    % J
    matJ = matJ_from_vecJ_1qubit(vecJ);
    HScb_J = HScb_J_from_matJ(matJ);
    
    % K
    HScb_K = HScb_K_from_matK_1qubit(matK);
    
    % L
    HScb_L = zeros(4);
    HScb_L = HScb_L + HScb_H;
    HScb_L = HScb_L + HScb_J;
    HScb_L = HScb_L + HScb_K;

    output = HScb_L;
end

