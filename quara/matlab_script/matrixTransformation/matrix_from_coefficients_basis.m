function output = matrix_from_coefficients_basis(vecA, basis)
%MATRIX_FROM_COEFFICIENTS_BASIS returns Hamiltonian 
%   matA = \sum_{j} vecA(j) .* basis(j).mat, j = 1,...,dim^2 
    assert(isEqualElementSize_matrixBasis(basis));
    assert(isValidNo_matrixBasis(basis));
    
    basisNo = size(basis, 2);
    [dim1, dim2] = size(basis(1).mat);
    
    size_vec = size(vecA, 2);
    assert(size_vec == basisNo);
    
    matA = zeros(dim1, dim2);
    for iA = 1:basisNo
        matA = matA + vecA(iA) .* basis(iA).mat;
    end
    
    output = matA;
end

