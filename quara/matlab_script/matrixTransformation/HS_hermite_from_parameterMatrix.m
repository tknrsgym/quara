function output = HS_hermite_from_parameterMatrix(matA)
%HS_HERMITE_FROM_PARAMETERMATRIX returns square matrix
%   - matA: (dim^2 -1) x dim^2 real matrix
%   - HS: output, dim^2 x dim^2 real matrix
    [size1, size2] = size(matA);
    assert(size1 + 1 == size2);
    
    vec = zeros(1, size2);
    vec(1) = 1;
    HS = [vec; matA];
    
    output = HS;
end

