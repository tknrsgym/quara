function output = isTrue_condition6(mat1, mat2, k, eps)
%ISTRUE_CONDITION6 returns true if two diagonalizable matrices satisfy condition 6.
%   Condition 6: If (i,j) satisfies Tr[ Pi(mat1) Pj(mat2)] > 0.5 .* min{ rank(Pi(mat1)),
%   rank(Pj(mat2))}, k * |Imag(lambda_i(mat1)) Imag(lambda_j(mat2))| < pi holds. 
%    - mat1: a diagonalizable matrix.
%    - mat2: a diagonalizable matrix.
%    - { ( Pi(matrix), lambda_i(matrix) ) }: a set of pairs of eigenvalue
%    and orthogonal projections to the (right) eigen-subspace.
%    - k: a positive integer.
%    - eps >= 0: a tolerance parameter for identifying a tiny numerical
%    descripancy.
    [size1_1, size1_2] = size(mat1);
    assert(size1_1 == size1_2);
    [size2_1, size2_2] = size(mat2);
    assert(size2_1 == size2_2);
    assert(size1_1 == size2_1);
    size_mat = size1_1;
    
    assert(isDiagonalizable_matrix(mat1));
    assert(isDiagonalizable_matrix(mat2));    
    assert(eps >= 0);
    
    res = true;

    % Spectral decomposition of mat1
    [matV1, matD1] = diagonalize_matrix(mat1);

    list_eval1 = diag(matD1);
    [list_eval1_unique, listIdSet1] = uniqueValueList_idList_from_nonuniqueValueList(list_eval1, eps);
    num_eval1_unique = size(list_eval1_unique, 2); 
    proj1 = listProjection_from_matV_listIdSet(matV1, listIdSet1);
    
    % Spectral decomposition of mat2
    [matV2, matD2] = diagonalize_matrix(mat2);

    list_eval2 = diag(matD2);
    [list_eval2_unique, listIdSet2] = uniqueValueList_idList_from_nonuniqueValueList(list_eval2, eps);
    num_eval2_unique = size(list_eval2_unique, 2); 
    proj2 = listProjection_from_matV_listIdSet(matV2, listIdSet2);
    
    % 
    for i1 = 1:num_eval1_unique
       eval1 = list_eval1_unique(i1);
       matP1 = proj1(i1).mat;
       rank_P1 = rank(matP1);
       for j2 = 1:num_eval2_unique
           eval2 = list_eval2_unique(j2);
           matP2 = proj2(j2).mat;
           rank_P2 = rank(matP2);
           
           LHS = abs(trace(matP1 * matP2));
           RHS = 0.50 .* min([rank_P1, rank_P2]);
           if LHS > RHS
              y1 = imag(eval1);
              y2 = imag(eval2);
              diff = abs(y1 - y2);
              if k .* diff >= pi
                  res = false;
              end
           end
       end
    end

    output = res;
end

