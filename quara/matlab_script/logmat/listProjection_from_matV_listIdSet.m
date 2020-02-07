function output = listProjection_from_matV_listIdSet(matV, listIdSet)
%LISTPROJECTION_FROM_MATV_LISTIDSET returns a list of orthogonal projections
%from matV and listIdSet.
%   - matV: an invertible matrix
%   - listID: list of ID sets. Each id set includes ids for a common
%   projector
%   - matV(:, id) is a right eigenvector. Each eigen vector is normalized,
%   but not necessarilly orthogonal.
    [size1, size2] = size(matV);
    assert(size1 == size2);
    size_V = size1;
    rank_V = rank(matV);
    assert(rank_V == size_V);
    
    num_eval_unique = size(listIdSet, 2);
    for i_eval_unique = 1:num_eval_unique
        num_eval_i = size(listIdSet(i_eval_unique).vec, 2);
        mat1 = [];
        for j_eval = 1:num_eval_i
            id_j = listIdSet(i_eval_unique).vec(j_eval); 
            vec1 = matV(:, id_j);
            mat1 = [mat1, vec1];
        end

        mat2 = orth(mat1);
        mat3 = zeros(size_V, size_V);
        for i2 = 1:rank(mat2)
            vec2 = mat2(:,i2);
            mat4 = vec2 * ctranspose(vec2);
            mat3 = mat3 + mat4;
        end
        proj(i_eval_unique).mat = mat3;
    end

    output = proj;
end

