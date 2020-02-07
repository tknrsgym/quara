function output = hamiltonianCoefficients_from_HS_comp_Lindbladian(HS_L, basis)
%HAMILTONIANCOEFFICIENTS_FROM_HS_COMP_LINDBLADIAN returns the hamiltonian
%coefficients from the Hilbert-Schmidt representation of a Lindbladian
%w.r.t. the computational basis.
%   - HS_L: the Hilbert-Schmidt representation of a Lindbladian w.r.t. the
%           computational basis.
%   - d : the dimension of the quantum system
%   - matH = sum_{j=1}^{d^2} vecH(j) .* basis(j).mat, a Hamiltonian
%   - vecH : the output of this function.
%            vecH(j) = trace( (B(j) \otimes I - I \otimes conj(B(j))) HS_L ) 
%                      .* 1i / 2d
%            vecH(1) = 0 is automatically satisfied.
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
    
    vecH = zeros(1, size1);
    matI = eye(dim, dim);
    for iH = 1:size1
       mat1 = basis(iH).mat;
       mat2 = kron(mat1, matI) - kron(matI, conj(mat1));
       vecH(iH) = trace(mat2 * HS_L) .* 1i ./ (2 .* dim);
    end
    
    output = vecH;
end

