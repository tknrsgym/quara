function output = is1stElementProportionalToIdentity_matrixBasis(basis, eps)
%IS1STELEMENTPROPORTIONALTOIDENTITY_MATRIXBASIS returns true if the 1st element of the
%matrix basis is proportional to the identity matrix.
%   The matrix size is assumed to be square.
    assert(isSquare_matrixBasis(basis));
    res = true;
    
    mat1 = basis(1).mat;
    [size1, size2] = size(mat1);
    dim = size1;
    c = trace(mat1) ./ dim;
    mat2 = c .* eye(dim, dim);
    diff = norm(mat1 - mat2, 'fro');
    if diff > eps
        res = false;
    end
    
    output = res;
end

