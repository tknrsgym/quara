function output = isDiagonalizable_matrix(matA)
%ISDIAGONALIZABLE_MATRIX returnes true if the input matrix is diagonalizable. 
%   A square matrix matA is called diagonalizable if there exit invertivle
%   matrix matV and diagonal matrix matD such that matA = matV * matD *
%   inv(matV).
    [size1, size2] = size(matA);
    assert(size1 == size2);
    
    res = false;    
    
    [matV, matD] = eig(matA);
    rank_V = rank(matV);
    if rank_V == size1
        res = true;
    end

    output = res;
end

