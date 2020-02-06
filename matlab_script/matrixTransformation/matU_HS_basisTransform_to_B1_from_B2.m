function output = matU_HS_basisTransform_to_B1_from_B2(basis1, basis2)
%MATU_HS_BASISTRANSFORM_TO_B1_FROM_B2 
%   U_{i1, i2} = Tr [ ctranspose(B1_i1) * B2_i2 ]
    eps = 10^(-10);   
    % Check of basis1
    assert(isValidNo_matrixBasis(basis1), "The number of elements is invalid for basis1 in matU_HS_basisTransform_to_B1_from_B2()!");
    assert(isNormalized_matrixBasis(basis1, eps), "The basis1 is not normalized in matU_HS_basisTransform_to_B1_from_B2()!");
    assert(isOrthogonal_matrixBasis(basis1, eps), "The basis1 is not orthogonal in matU_HS_basisTransform_to_B1_from_B2()!");
    % Check of basis2
    assert(isValidNo_matrixBasis(basis2), "The number of elements is invalid for basis2 in matU_HS_basisTransform_to_B1_from_B2()!");
    assert(isNormalized_matrixBasis(basis2, eps), "The basis2 is not normalized in matU_HS_basisTransform_to_B1_from_B2()!");
    assert(isOrthogonal_matrixBasis(basis2, eps), "The basis2 is not orthogonal in matU_HS_basisTransform_to_B1_from_B2()!");
    % Check of basis1 and basis2
    basis1No = size(basis1, 2);
    basis2No = size(basis2, 2);
    assert(basis1No == basis2No, "The numbers of basis elements are different in matU_HS_basisTransform_to_B1_from_B2()!");
    
    % 
    matU = zeros(basis1No, basis2No);
    for i1 = 1:basis1No
        mat1 = basis1(i1).mat;
        for i2 = 1:basis2No
            mat2 = basis2(i2).mat;
            
            mat3 = ctranspose(mat1) * mat2;
            matU(i1, i2) = trace(mat3);
        end
    end
    
    output = matU;
end

