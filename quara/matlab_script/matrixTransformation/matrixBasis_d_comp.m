function output = matrixBasis_d_comp(dim)
%MATRIXBASIS_D_COMP returns a list structure of the computational basis on
%d times d matrix space.
%   output(iBasis) = | iRow > < jCol |, iBasis = 1 ~ dim * dim
%   iRow = 1 ~ dim
%   jCol = 1 ~ dim
%   iBasis = (iRow - 1) * dim + jCol
    for iRow = 1:dim
        for jCol = 1:dim
            iBasis = (iRow - 1) * dim + jCol;
            basis(iBasis).mat = zeros(dim, dim);
            basis(iBasis).mat(iRow, jCol) = 1;
        end
    end
    
    
%     for iBasis = 1:dim*dim
%         basis(iBasis).mat = zeros(dim, dim);
%         quotient = idivide(iBasis - 1, int32(dim), 'floor');
%         iRow = quotient + 1;
%         jCol = iBasis - (iBasis - 1) .* quotient;
%         basis(iBasis).mat(iRow, jCol) = 1;
%         [iBasis, quotient, iRow, jCol]
%     end
    output = basis;
end

