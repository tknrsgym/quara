function output = HScb_L_model_rotation_BE99_1qubit(vecH, T1, T2, alpha)
%HSCB_L_MODEL_ROTATION_BE99_1QUBIT この関数の概要をここに記述
%   詳細説明をここに記述
    
    % vecJ, matK
    vecJ = vecJ_from_model_BE99_1qubit(T1, T2, alpha);
    matK = matK_from_model_BE99_1qubit(T1, T2, alpha);
    
    HScb_L = HScb_L_from_vecHvecJmatK_1qubit(vecH, vecJ, matK);
    output = HScb_L;
end

