function output = isTrue_condition4_2(mat1, mat2, eps)
%ISTRUE_CONDITION4_2 returns true if two diagonalizable matrices satisfy condition 4-2.
%   Condition 4-2: \| E \|_F < c .* min_{(i,j), i ~= j} |lambda_i(mat1) -
%   lambda_j(mat2)|, c=(3sqrt(2) -2)/14
%    - mat1 needs to be Hermitian or anti-Hermitian.
%    - mat2 = mat1 + matE.
%    - eps >= 0   
    [size1_1, size1_2] = size(mat1);
    assert(size1_1 == size1_2);
    [size2_1, size2_2] = size(mat2);
    assert(size2_1 == size2_2);
    assert(size1_1 == size2_1);
    size_mat = size1_1;
    
    assert(isDiagonalizable_matrix(mat1));
    assert(isDiagonalizable_matrix(mat2));
    
    check1 = isHermitian_matrix(mat1, eps);
    check2 = isHermitian_matrix(1i .* mat1, eps);
    assert(check1 | check2);
    
    assert(eps >= 0);
    
    res = true;
    matE = mat2 - mat1;
    
    % Spectral decomposition of mat1
    [matU1, matD1] = diagonalize_matrix(mat1);

    list_eval1 = diag(matD1);
    [list_eval1_unique, listIdSet1] = uniqueValueList_idList_from_nonuniqueValueList(list_eval1, eps);
    num_eval1_unique = size(list_eval1_unique, 2); 
    
    % Spectral separation
    separation = abs(max(list_eval1) - min(list_eval1));
    for i1 = 1:num_eval1_unique -1
        eig1_i1 = list_eval1(i1);
        for j1 = i1+1:num_eval1_unique
            eig1_j1 = list_eval1(j1);
            diff = abs(eig1_i1 - eig1_j1);
            if diff < separation
                separation = diff;
            end
        end
    end
    
    % 
    norm_matE_F = norm(matE, 'fro');% Frobenius norm
    factor = (3.0 .* sqrt(2) - 2.0) ./ 14;
    if norm_matE_F >= factor .* separation
        res = false;
    end
    
    output = res;
end

