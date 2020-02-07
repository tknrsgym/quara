function output = matrixBasis_1qubit_comp()
%MATRIXBASIS_1QUBIT_COMP returns a list structure of matrix basis for dim = 2.
%   output(1).mat = |0><0|
%   output(2).mat = |0><1|
%   output(3).mat = |1><0|
%   output(4).mat = |1><1|
    dim = 2;
    basis = matrixBasis_d_comp(dim);
    output = basis;
end

