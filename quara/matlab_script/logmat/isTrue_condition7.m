function output = isTrue_condition7(k, mat1, eps)
%ISTRUE_CONDITION7 returns true if a matrix and an integer satisfy
%condition 7.
%   Condition 7. k <= kmax_min
    res = false;
    [kmax_min, kmax_max] = kmax_from_dissipation(mat1, eps);
    if k <= kmax_min
        res = true;
    end

    output = res;
end

