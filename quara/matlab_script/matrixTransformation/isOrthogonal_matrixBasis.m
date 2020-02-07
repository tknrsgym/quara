function output = isOrthogonal_matrixBasis(basis, eps)
%ISORTHOGONAL_MATRIXBASIS returns true if the input matrix basis is
%orthogonal.
%   
    res = true;
    
    basisNo = size(basis, 2);
    for i1 = 1:basisNo-1
        mat1 = basis(i1).mat;
        for i2 = i1+1:basisNo
            mat2 = basis(i2).mat;
            mat3 = ctranspose(mat1) * mat2;
            pr = trace(mat3);
            if abs(pr) > eps
                res = false;
            end
        end
    end
    
    output = res;
end

