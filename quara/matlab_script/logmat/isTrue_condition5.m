function output = isTrue_condition5(mat1, k, eps)
%ISTRUE_CONDITION5 returns true if a diagonalizable matrix and an integer k satisfy
%condition 5.
%   Condition 5. lambda_i(mat1) ~= lambda_j(mat1) => exp(k *
%   lambda_i(mat1)) ~= exp(k : lambda_j(mat1)).
%   - mat1: a diagonalizable matrix
%   - eps > 0: a tolerance parameter for identifying discrepancy.
    [size1_1, size1_2] = size(mat1);
    assert(size1_1 == size1_2);
    
    assert(isDiagonalizable_matrix(mat1));    
    assert(eps >= 0);
    
    res = true;
    
    % Spectral decomposition of mat1
    [matU1, matD1] = diagonalize_matrix(mat1);

    list_eval1 = diag(matD1);
    [list_eval1_unique, listIdSet1] = uniqueValueList_idList_from_nonuniqueValueList(list_eval1, eps);
    num_eval1_unique = size(list_eval1_unique, 2);     
    for i1 = 1:num_eval1_unique-1
       eval_i1 = list_eval1_unique(i1);
       val_i1  = exp(k .* eval_i1);
       for j1 = i1 + 1:num_eval1_unique
          eval_j1 = list_eval1_unique(j1);
          val_j1  = exp(k .* eval_j1);
          diff_k = abs(val_i1 - val_j1);
          if diff_k < eps
              res = false
          end
       end
    end
    
    output = res;
end

