function output = matrixBasis_1qubit_pauli_normalized()
%MATRIXBASIS_1QUBIT_PAULI_UNNORMALIZED returns a list structure of the
%normalized Pauli matrices
%   output(1).mat = I / sqrt(2) = sigma_0 / sqrt(2)
%   output(2).mat = X / sqrt(2) = sigma_1 / sqrt(2)
%   output(3).mat = Y / sqrt(2) = sigma_2 / sqrt(2)
%   output(4).mat = Z / sqrt(2) = sigma_3 / sqrt(2)
    basis = matrixBasis_1qubit_pauli_unnormalized();
    basisNo = size(basis, 2);
    for iBasis = 1:basisNo
       basis(iBasis).mat = basis(iBasis).mat ./ sqrt(2); 
    end
    output = basis;
end

