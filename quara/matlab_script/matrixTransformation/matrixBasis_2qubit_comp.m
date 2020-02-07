function output = matrixBasis_1qubit_comp()
%MATRIXBASIS_1QUBIT_COMP returns a list structure of computational matrix basis for dim = 4. 
    dim = 4;
    basis = matrixBasis_d_comp(dim);
    output = basis;
end

