function output = isHermitian_matrix(matA, eps)
%ISHERMITIAN_MATRIX returns true if all the matrix is Hermitian.
%   
    [dim1, dim2] = size(matA);
    assert(dim1 == dim2);
    res = true;
    
    mat1 = matA;
    mat2 = ctranspose(matA);
    diff = norm(mat1 - mat2, 'fro');
    if diff > eps
        res = false;
    end
       
    output = res;
end

