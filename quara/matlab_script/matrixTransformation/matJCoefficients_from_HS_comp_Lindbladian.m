function output = matJCoefficients_from_HS_comp_Lindbladian(HS_L, basis)
%MATJCOEFFICIENTS_FROM_HS_COMP_LINDBLADIAN returns the J-matrix
%coefficients from the Hilbert-Schmidt representation of a Lindbladian
%w.r.t. the computational basis.
%   - HS_L: the Hilbert-Schmidt representation of a Lindbladian w.r.t. the
%           computational basis.
%   - d : the dimension of the quantum system
%   - matJ = sum_{j=1}^{d^2} vecJ(j) .* basis(j).mat, a J-matrix
%   - vecJ : the output of this function.
%            vecJ(j) = trace( (B(j) \otimes I + I \otimes conj(B(j))) HS_L ) 
%                      .* / { 2d ( 1 + delta_{1, j}) }
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
    
    vecJ = zeros(1, size1);
    matI = eye(dim, dim);
    for iJ = 1:size1
        factor = 1;
        if iJ == 1
            factor = 2;
        end
        mat1 = basis(iJ).mat;
        mat2 = kron(mat1, matI) + kron(matI, conj(mat1));
        vecJ(iJ) = trace(mat2 * HS_L) ./ (2 .* dim .* factor);
    end
    
    output = vecJ;
end

