function [output1, output2] = vec_n_from_eval_k(eval, k)
%VEC_N_FROM_EVAL_K この関数の概要をここに記述
%   詳細説明をここに記述
    sz = size(eval,1);
    for i = 1:sz
        lambda = k * imag(eval(i));
        [n, Delta] = mod_2pi_type2(lambda);
        vec_n(i) = n;
        vec_Delta(i) = Delta;
    end
    output1 = vec_n;
    output2 = vec_Delta;
end

