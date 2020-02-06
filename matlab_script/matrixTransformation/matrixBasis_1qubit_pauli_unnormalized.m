function output = matrixBasis_1qubit_pauli_unnormalized()
%MATRIXBASIS_1QUBIT_PAULI_UNNORMALIZED returns a list structure of the
%unnormalized Pauli matrices
%   output(1).mat = I = sigma_0
%   output(2).mat = X = sigma_1
%   output(3).mat = Y = sigma_2
%   output(4).mat = Z = sigma_3
    basis(1).mat = eye(2,2);
    basis(2).mat = [0, 1; 1, 0];
    basis(3).mat = [0, -1i; 1i, 0];
    basis(4).mat = [1, 0; 0, -1];
    output = basis;
end

