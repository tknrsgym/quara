function output = matK_from_jumpOperator(list_c, basis)
%MATK_FROM_JUMPOPERATOR returns K-matrix of the dissipation part of
%Lindbladian
%   \sum_{j} c(j) \rho c(j)^\dagger = \sum_{a, b =1}^{d^2 - 1} K(a,b) B(a + 1) \rho B(b + 1)^\dagger
%   K: (d^2 -1) x (d^2 -1) complex matrix, to be positive semidefinite automatically.
%   {B(a)}_{a} the orthogonal and normalized matrix basis with B(0) \propto matI.
%   c(j): traceless
%   Note : K(a, b) corresponds to B(a + 1) and B(b+1)^\dagger.
    eps = 10^(-10);
    assert(isOrthogonal_matrixBasis(basis, eps));
    assert(isNormalized_matrixBasis(basis, eps));
    assert(isSquare_matrixBasis(basis));
    assert(isEqualElementSize_matrixBasis(basis));
    assert(isValidNo_matrixBasis(basis));
    assert(is1stElementProportionalToIdentity_matrixBasis(basis, eps));
    dim_basis = size(basis(1).mat, 1);
    
    assert(isEqualElementSize_matrixBasis(list_c));
    assert(isSquare_matrixBasis(list_c));
    assert(isEqualElementSize_matrixBasis(list_c));
    assert(isTraceless_matrixList(list_c, eps));
    cNo = size(list_c, 2);
    dim_c = size(list_c(1).mat, 1);
    
    assert(dim_basis == dim_c);
    dim = dim_basis;
    sizeK = dim .* dim -1;
    
    matC = zeros(sizeK + 1, cNo);
    for ic = 1:cNo
        mat_c = list_c(ic).mat;
        coef = matrixCoefficients_from_mat_basis(mat_c, basis);
        matC(:,ic) = coef;
    end
    matK_dummy = matC * ctranspose(matC);
    matK = matK_dummy(2:sizeK+1, 2:sizeK+1);
    
    output = matK;
end

