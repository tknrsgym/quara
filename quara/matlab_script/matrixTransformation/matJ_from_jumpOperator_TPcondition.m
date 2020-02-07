function output = matJ_from_jumpOperator_TPcondition(list_c)
%MATJ_FROM_JUMPOPERATOR_TPCONDITION returns matJ in Lindbladian from jump
%operators.
%   matJ = -0.5 * sum_j ctranspose(list_c(j).mat) * list_c(j).mat
    eps = 10^(-10);
    assert(isEqualElementSize_matrixBasis(list_c));
    assert(isSquare_matrixBasis(list_c));
    assert(isEqualElementSize_matrixBasis(list_c));
    assert(isTraceless_matrixList(list_c, eps));
    cNo = size(list_c, 2);
    dim = size(list_c(1).mat, 1);
    
    matJ = zeros(dim, dim);
    for ic = 1:cNo
        mat1 = list_c(ic).mat;
        mat2 = ctranspose(mat1);
        matJ = matJ + mat2 * mat1; 
    end
    matJ = -0.50 .* matJ;
    
    output = matJ;
end

