function output = HS_from_Choi(Choi, basis)
%HS_FROM_CHOI returns the Hilbert-Schmidt representation w.r.t. the given matrix basis
%corresponding to the given Choi matrix.
%   HS(i1, i2) = Tr [ { ctranspose(basis(i1).mat) \otimes transpose(basis(i2).mat) } * Choi ]
    [size1, size2] = size(Choi);
    assert(size1 == size2);
    
    basisNo = size(basis, 2);
    assert(basisNo == size1);
    
    HS = zeros(basisNo, basisNo);
    for i1 = 1:basisNo
        mat1 = ctranspose(basis(i1).mat);
        for i2 = 1:basisNo
            mat2 = transpose(basis(i2).mat);
            mat3 = kron(mat1, mat2);
            HS(i1, i2) = trace(mat3 * Choi);
        end
    end

    output = HS;
end

