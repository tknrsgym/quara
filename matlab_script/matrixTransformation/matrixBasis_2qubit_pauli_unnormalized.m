function output = matrixBasis_2qubit_pauli_unnormalized()
%MATRIXBASIS_2QUBIT_PAULI_UNNORMALIZED returns the unnormalized Pauli
%matrix basis on 2-qubit system.
%   output(i1, i2).mat = Pauli(i1) \otimes Pauli(i2)
%   iBasis = 4 * (i1 -1) + i2
    basis1 = matrixBasis_1qubit_pauli_unnormalized;
    basisNo = size(basis1, 2);
    for i1 = 1:basisNo
       for i2 = 1:basisNo
           iBasis = basisNo * (i1 - 1) + i2;
           basis(iBasis).mat = kron(basis1(i1).mat, basis1(i2).mat); 
       end
    end
    output = basis;
end

