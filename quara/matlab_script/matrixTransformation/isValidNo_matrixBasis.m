function output = isValidNo_matrixBasis(basis)
%ISVALIDNO_MATRIXBASIS returns true if the number of basis elements is
%valid.
%  
    res = true;
    
    basisNo = size(basis, 2);
    for i1 = 1:basisNo
        dim1 = size(basis(i1).mat, 1);
        dim2 = size(basis(i1).mat, 2);
        if basisNo ~= dim1 * dim2
            res = false;
        end
    end
    
    output = res;
end

