function output = nonpv_log_matrix(matGk, matL0, k, eps)
%NONPV_LOG_MATRIX この関数の概要をここに記述
%   詳細説明をここに記述
    [size1_matGk, size2_matGk] = size(matGk);
    assert(size1_matGk == size2_matGk);
    size_matGk = size1_matGk;
    
    [size1_matL0, size2_matL0] = size(matL0);
    assert(size1_matL0 == size2_matL0);
    size_matL0 = size1_matL0;    
    assert(size_matGk == size_matL0);
    
    assert(k > 0);
    assert(eps > 0);
    
    % Reference matrix logarithm, matL0
    matL0k = k .* matL0;
    [matV0, matD0] = diagonalize_matrix(matL0k);

    list_eval0 = diag(matD0);
    [list_eval0_unique, listIdSet0] = uniqueValueList_idList_from_nonuniqueValueList(list_eval0, eps);
    num_eval0_unique = size(list_eval0_unique, 2);
    proj0 = listProjection_from_matV_listIdSet(matV0, listIdSet0);
    
    % Principal matrix logarithm, matL1
    matL1 = logm(matGk);
    [matV1, matD1] = diagonalize_matrix(matL1);
    list_eval1 = diag(matD1);
    [list_eval1_unique, listIdSet1] = uniqueValueList_idList_from_nonuniqueValueList(list_eval1, eps);
    proj1 = listProjection_from_matV_listIdSet(matV1, listIdSet1);

    % Correspondence of right invariant subspace
    num_eval1_unique = size(list_eval1_unique, 2);
    id_eval0_unique_correspond = -1 .* ones(num_eval1_unique, 1);
    for i1 = 1:num_eval1_unique
       mat1 = proj1(i1).mat;
       rank1 = round(trace(mat1));
       for i0 = 1:num_eval0_unique
           mat0 = proj0(i0).mat;
           rank0 = round(trace(mat0));
           if (rank1 > rank0)
                break;
           end
           value = trace(mat1 * mat0);
           if value > 0.50 .* rank1
               id_eval0_unique_correspond(i1) = i0;
               break;
           end
       end
       assert(id_eval0_unique_correspond(i1) > 0);
    end

    % Reconstruction of Eigenvalues
    eps1 = 10^(-10);
    eps2 = 10^(-10);
    for i1 = 1:num_eval1_unique
        eval1 = list_eval1_unique(i1);
        x1 = real(eval1);
        y1 = imag(eval1);
        eps_y1 = 10^(-11);
        if abs(y1 - pi) < eps_y1
            y1 = pi;
        elseif abs(y1 + pi) < eps_y1
            y1 = -pi; 
        end

        i0 = id_eval0_unique_correspond(i1);
        eval0 = list_eval0_unique(i0);
        y0 = imag(eval0);

        [y2, case_y0, case_y1] = argument_nonpv(y1, y0, eps1, eps2);
        assert(case_y1 ~= 3, 'The condition, |y1-y0| < pi, is not satisfied!');
        z2 = x1 +1i .* y2;
        eval2(i1) = z2;
    end

    % Non-principal matrix logarithm
    [size1, size2] = size(matD1);
    matD2 = zeros(size1, size2);
    num_eval2_unique = size(eval2, 2);
    for i2 = 1:num_eval2_unique
        num_id = size(listIdSet1(i2).vec, 2);
        for i_id = 1:num_id
            id = listIdSet1(i2).vec(i_id);
            matD2(id, id) = eval2(i2);
        end
    end
    matL2 = matV1 * matD2 * inv(matV1);

    output = matL2;
end

