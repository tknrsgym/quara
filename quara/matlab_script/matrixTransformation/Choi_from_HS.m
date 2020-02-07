function output = Choi_from_HS(HS, basis)
%CHOI_FROM_HS returns Choi matrix for the given Hilbert-Schmidt
%representation with respect to the basis.
%   Choi = sum_{i1, i2} HS(i1, i2) .* basis(i1).mat \otimes conj(basis(i2).mat) 
    [size1, size2] = size(HS);
    assert(size1 == size2);
    
    basisNo = size(basis, 2);
    assert(basisNo == size1);
    
    Choi = zeros(size1, size2);
    for i1 = 1:size1
        mat1 = basis(i1).mat;
        for i2 = 1:size2
            mat2 = conj(basis(i2).mat);
            mat3 = kron(mat1, mat2);
            Choi = Choi + HS(i1, i2) .* mat3;
        end
    end
    
    output = Choi;
end

