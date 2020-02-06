function output = matK_from_HS_comp_Lindbladian(HS_L, basis)
%MATK_FROM_HS_COMP_LINDBLADIAN returns matK from the Hilbert-Schmidt
%representation oa a Lindbladian in the computational basis.
%   - HS_L: the Hilbert-Schmidt representation of a Lindbladian w.r.t. the
%           computational basis.
%   - d : the dimension of the quantum system
%   - matK : a K-matrix of a Lindbladian. The output of this function.
%            matK(j1, j2) = trace( (B(j1) \otimes conj(B(j2))) HS_L ) 
%   - basis: an orthogonal and normalized matrix basis with 
%            basis(1).mat = eye(d,d) / sqrt(d)
    eps = 10^(-10);
    assert(isOrthogonal_matrixBasis(basis, eps));
    assert(isNormalized_matrixBasis(basis, eps));
    assert(isSquare_matrixBasis(basis));
    assert(isEqualElementSize_matrixBasis(basis));
    assert(isValidNo_matrixBasis(basis));
    assert(is1stElementProportionalToIdentity_matrixBasis(basis, eps));
    basisNo = size(basis, 2);
    dim = size(basis(1).mat, 1);
    
    [size1, size2] = size(HS_L);
    assert(size1 == size2);
    assert(size1 == dim .* dim);
    
    sizeK = size1 - 1;
    matK = zeros(sizeK, sizeK);
    for j1 = 1:sizeK
        mat1 = basis(j1 +1).mat;
        for j2 = 1:sizeK
            mat2 = conj(basis(j2 +1).mat);
            mat3 = kron(mat1, mat2);
            matK(j1, j2) = trace(mat3 * HS_L);    
        end
    end
    
    output = matK;
end

