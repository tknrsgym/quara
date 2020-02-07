function output = matrixCoefficients_from_mat_basis(matA, basis)
%MATRIXCOEFFICIENTS_FROM_MAT_BASIS returns a list of matrix coefficients
%   matA = \sum_{j=1}^{dim .* dim} a(j) .* basis(j).mat
%   a(j) = Tr[ ctranspose(basis(j).mat) * matA ] / Tr[ ctranspose(basis(j).mat) * basis(j).mat ]
%   The matrix basis needs to be orthogonal.
    eps = 10^(-10);
    assert(isEqualElementSize_matrixBasis(basis));
    assert(isValidNo_matrixBasis(basis));
    assert(isOrthogonal_matrixBasis(basis, eps));
    
    basisNo = size(basis, 2);
    [dim1, dim2] = size(basis(1).mat);
    
    [size1, size2] = size(matA);
    assert(dim1 == size1);
    assert(dim2 == size2);
    
    size_vec = basisNo;
    vecA = zeros(1, basisNo);
    for iA = 1:size_vec
        mat1 = basis(iA).mat;
        mat1_dag = ctranspose(mat1); 
        val1 = trace(mat1_dag * matA);
        val2 = trace(mat1_dag * mat1);
        vecA(iA) = val1 ./ val2;
    end
    
    output = vecA;
end

