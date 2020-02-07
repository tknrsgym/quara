function [output1,output2] = diagonalize_matrix(matA)
%DIAGONALIZE_MATRIX returns an invertible matrix matV and a diagonal matrix
%matD satisfying matA = matV * matD * inv(matV).
%   - The order of eigenbalues in matD is sorted as 'ComparisonMethod' &
%   'abs'.
%    The meanings of the option is described in https://mathworks.com/help/matlab/ref/sort.html
    [size1, size2] = size(matA);
    assert(size1 == size2);
    size_matA = size1;
    
    [matV, matD] = eig(matA);
    rank_matV = rank(matV);
    assert(rank_matV == size_matA);
    
    list_eval = diag(matD);
    [list_eval, index] = sort(list_eval, 'ComparisonMethod', 'abs');
    
    matD = matD(index, index);
    matV = matV(:, index);

%     [matD, index] = sortrows(matD ,'ComparisonMethod','real');
%     matD = matD(:, index);
%     matV = matV(:, index);
    
    output1 = matV;
    output2 = matD;
end

