function output = isHermitian_matrixBasis(basis, eps)
%ISHERMITIAN_MATRIXBASIS returns true if all the matrix basis elements are
%Hermitian.
%   
    res = true;
    
     basisNo = size(basis, 2);
    for i1 = 1:basisNo
       mat1 = basis(i1).mat;
       mat2 = ctranspose(mat1);
       diff = norm(mat1 - mat2, 'fro');
       if diff > eps
           res = false;
       end
    end
    
    output = res;
end

