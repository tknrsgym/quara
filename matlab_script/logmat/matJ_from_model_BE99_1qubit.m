function output = matJ_from_model_BE99_1qubit(T1, T2, alpha)
%MATJ_FROM_MODEL_BE99_1QUBIT ���̊֐��̊T�v�������ɋL�q
%   �ڍא����������ɋL�q
    vecJ = vecJ_from_model_BE99_1qubit(T1, T2, alpha);   
    matJ = matJ_from_vecJ_1qubit(vecJ);
    
    output = matJ;
end

