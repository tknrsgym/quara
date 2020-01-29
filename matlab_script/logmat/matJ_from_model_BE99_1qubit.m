function output = matJ_from_model_BE99_1qubit(T1, T2, alpha)
%MATJ_FROM_MODEL_BE99_1QUBIT この関数の概要をここに記述
%   詳細説明をここに記述
    vecJ = vecJ_from_model_BE99_1qubit(T1, T2, alpha);   
    matJ = matJ_from_vecJ_1qubit(vecJ);
    
    output = matJ;
end

