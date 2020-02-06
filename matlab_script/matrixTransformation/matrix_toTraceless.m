function output = matrix_toTraceless(matA)
%MATRIX_TRACELESS returns traceless matrix from possibly
%non-traceless matrix.
%   - matA: a square matrix with size dim
%   - output = matA - trace(matA) .* eye(dim, dim) ./ dim
    [dim1, dim2] = size(matA);
    assert(dim1 == dim2);
    dim = dim1;
    
    matI = eye(dim, dim);
    tr = trace(matA);
    mat1 = tr .* matI ./ dim;
    mat2 = matA - mat1;
    
    output = mat2;
end

