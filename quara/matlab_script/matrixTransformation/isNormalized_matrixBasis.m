function output = isNormalized_matrixBasis(basis, eps)
%ISNORMALIZED_MATRIXASIS returns true if all of the matrix basis elements
%are normalized.
%   
    res = true;

    basisNo = size(basis, 2);
    for i1 = 1:basisNo
        mat1 = basis(i1).mat;
        mat2 = ctranspose(mat1) * mat1;
        pr = trace(mat2);
        if abs(pr - 1) > eps
            res = false;
        end
    end
    
    output = res;       
end

