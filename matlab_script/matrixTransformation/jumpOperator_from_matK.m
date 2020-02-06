function output = jumpOperator_from_matK(matK, basis)
%JUMPOPERATOR_FROM_MATK returns a list of jump operators corresponds to the
%given matK.
%   - matK : (dim^2 -1) x (dim^2 -1) Hermitian matrix, to be positive
%   semidefinite.
%   - basis: an orthogonal and normalized matrix basis with basis(1).mat = eye(dim, dim) ./ sqrt(dim).
%   - K = U D U^\dagger
%   - list_c(ic) = sqrt(D(ic,ic)) .* \sum_{j = 1}^{dim^2 -1} U(j, ic) .* basis(j).mat
    eps = 10^(-10);
    assert(isOrthogonal_matrixBasis(basis, eps));
    assert(isNormalized_matrixBasis(basis, eps));
    assert(isSquare_matrixBasis(basis));
    assert(isEqualElementSize_matrixBasis(basis));
    assert(isValidNo_matrixBasis(basis));
    assert(is1stElementProportionalToIdentity_matrixBasis(basis, eps));
    basisNo = size(basis, 2);
    dim = size(basis(1).mat, 1);
    
    assert(isHermitian_matrix(matK, eps));
    sizeK = size(matK, 1);
    assert(dim .* dim == sizeK + 1);
    
    [matU, matD] = eig(matK);
    
    eps_eval = 10^(-12);
    for ic = 1:sizeK
        eval = matD(ic, ic);
        if abs(eval) < eps_eval
            matD(ic, ic) = 0;
        end
        disp(matD(ic, ic))
        assert(matD(ic, ic) >= 0.0);
        mat1 = zeros(dim, dim);
        for iBasis = 1:basisNo-1
            mat2 = basis(iBasis + 1).mat;
            mat1 = mat1 + matU(iBasis, ic) .* mat2;
        end
        list_c(ic).mat = sqrt(matD(ic, ic)) .* mat1;
    end

    % Phase normalization
    eps_factor = 10^(-10);
    for ic = 1:sizeK
        isFound = false; 
        factor = 1.0;
        for i1 = 1:dim
            for i2 = 1:dim
               element = list_c(ic).mat(i1, i2);
               if abs(element) > eps_factor
                  factor = element ./ abs(element); 
                  isFound = true;
                  break;
               end
            end
            if isFound == true
                break
            end
        end
        list_c(ic).mat = list_c(ic).mat ./ factor;
    end
    
    output = list_c;
end

