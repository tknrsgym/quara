function output = isSquare_matrixBasis(basis)
%ISSQUARE_MATRIXBASIS returns true if all elements of the matrix basis are
%square matrices with the same size.
%   
    res = true;
    basisNo = size(basis, 2);
    for j = 1:basisNo
       [dim1, dim2] = size(basis(j).mat);
       if dim1 ~= dim2
           res = false;
       end
    end
    
    output = res;
end

