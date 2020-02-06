function output = HScb_from_jumpOperator(list_c)
%HSCB_FROM_JUMPOPERATOR returns the Hilbert-Schmidt representation of the 
%dissipation part in the Lindbladian
%   - list_c : a list of jump operators
%   HS = \sum_j c_j \otimes conj(c_j) 
%        - 0.5 .* ctranspose(c_j) * c_j \otimes I 
%        - 0.5 .* I \otimes transpose(c_j) * conj(c_j)
    cNo = size(list_c, 2);
    [dim1, dim2] = size(list_c(1).mat);
    assert(dim1 == dim2);
    dim = dim1;
    for ic = 1:cNo
        [dim1, dim2] = size(list_c(ic).mat);
        assert(dim1 == dim2);    
        assert(dim1 == dim);
    end
    matI = eye(dim, dim);
    
    HS = zeros(dim1.*dim2, dim1.*dim2);
    for ic = 1:cNo
        c = list_c(ic).mat;
        c_dagger = ctranspose(c);
        c_conj   = conj(c);
        c_t      = transpose(c);
        
        mat1 = kron(c, c_conj);
        mat2 = kron(c_dagger * c, matI);
        mat3 = kron(matI, c_t * c_conj);
        mat4 = mat1 - 0.50 .* mat2 - 0.50 .* mat3;
        HS = HS + mat4;
    end
    
    output = HS;
end

