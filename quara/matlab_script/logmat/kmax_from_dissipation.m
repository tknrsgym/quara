function [output1, output2] = kmax_from_dissipation(matL, eps)
%KMAX_FROM_DISSIPATION returns the inteder closest to 1 / max{ abs( real( lambda_i(matL) ) )}
%   - matL: a Hilbert-Schmidt representation of Lindbladian
%   - eps > 0
    [size1, size2] = size(matL);
    assert(size1 == size2);
    assert(eps > 0);
    
    list_eval = eig(matL);
    num_eval = size(list_eval,1);
    val_min = eps;
    count_min = 0;
    val_max = eps;
    for iEval = 1:num_eval
        eval = list_eval(iEval);
        val = abs(real(eval));
        
        if val > eps & count_min == 0
            val_min = val;
            count_min = 1;
        end
        
        if val > eps & count_min == 1
            if val < val_min
                val_min = val;
            end
        end
        
        if val > val_max
            val_max = val;
        end
    end
    kmax_min = round(1./val_max);
    kmax_max = round(1./val_min);
    
    output1 = kmax_min;
    output2 = kmax_max;
end

