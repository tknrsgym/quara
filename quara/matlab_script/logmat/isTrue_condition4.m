function output = isTrue_condition4(mat1, mat2, eps)
%ISTRUE_CONDITION4 returns true if two diagonalizable matrices satisfy condition 4.
%   Condition 4: gamma * eta / delta^2 < 1/8.
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
    proj1 = listProjection_from_matV_listIdSet(matU1, listIdSet1);
    
    % Calculate gamma, eta, delta for each invariant subspace of mat1
    for i1 = 1:num_eval1_unique
        matP = proj1(i1).mat;
        matP_comp = eye(size_mat, size_mat) - matP;
        
        % gamma
        mat_gamma = matP_comp * matE * matP;
        gamma = norm(mat_gamma, 'fro');
        
        % eta
        mat_eta = matP * matE * matP_comp;
        eta = norm(mat_eta, 'fro');
        
        % delta
        eig1 = list_eval1(i1);
        delta = abs(max(list_eval1) - min(list_eval1));
        for j1 = 1:num_eval1_unique
            if i1 == j1
                continue
            end
            
            eig1_j1 = list_eval1(j1);
            diff = abs(eig1 - eig1_j1);
            if diff < delta
                delta = diff;
            end
        end
        
        mat3 = matP * matE * matP;
        val3 = norm(mat3, 'fro');
        
        mat4 = matP_comp * matE * matP_comp;
        val4 = norm(mat4, 'fro');
        
        delta = delta - val3 - val4;
        
        % Is condition 4 satisfied?        
        if (gamma .* eta) ./ (delta .* delta) >= 0.125% = 1/8
           res = false; 
        end
    end
    
    output = res;
end

