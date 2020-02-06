function output = Choi_from_HS_parameterMatrix(matA, basis)
%CHOI_FROM_HS_PARAMETERMATRIX returns Choi matrix
%   
    eps = 10^(-10);
    assert(isOrthogonal_matrixBasis(basis, eps));
    assert(isNormalized_matrixBasis(basis, eps));
    assert(isSquare_matrixBasis(basis));
    assert(isEqualElementSize_matrixBasis(basis));
    assert(isValidNo_matrixBasis(basis));
    assert(is1stElementProportionalToIdentity_matrixBasis(basis, eps));
    
    HS = HS_hermite_from_parameterMatrix(matA);
    Choi = Choi_from_HS(HS, basis);
    
    output = Choi;
end

