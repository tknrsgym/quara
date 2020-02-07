function output = isTrue_condition6_1(k, matL0, matL1)
%ISTRUE_CONDITION6_1 returns true if condition 6-1 is satisfied.
%   Condition 6-1. k * \| matL1 - matL0 \|_2 < pi
%   - k > 0, integer
    assert(k > 0);
    [size0_1, size0_2] = size(matL0);
    assert(size0_1 == size0_2);
    size0 = size0_1;
    
    [size1_1, size1_2] = size(matL1);
    assert(size1_1 == size1_2);
    size1 = size1_1;
    
    assert(size0 == size1);
    
    res = false;
    matE = matL1 - matL0;
    norm_matE = norm(matE, 2);% Spectral norm
    if k .* norm_matE < pi
        res = true;
    end
    
    output = res;
end

