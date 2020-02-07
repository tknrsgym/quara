function output = isEqualElementSize_matrixBasis(basis)
%ISEQUALELEMENTSIZE_MATRIXBASIS returns true if sizes of all elements in
%the matrix basis are equal.
%   
    res = true;
    basisNo = size(basis, 2);
    [dim1, dim2] = size(basis(1).mat);
    for j = 2:basisNo
       [size1, size2] = size(basis(j).mat);
       if dim1 ~= size1 | dim2 ~= size2
           res = false
    end
    output = res;
end

