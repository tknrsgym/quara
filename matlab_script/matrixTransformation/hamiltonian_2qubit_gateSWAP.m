function output = hamiltonian_2qubit_gateSWAP()
%HAMILTONIAN_2QUBIT_GATESWAP returns the Hamiltonian corresponding to the SWAP gate on 2-qubit system.
%   CX = |0><0| \otimes |0><0| + |0><1| \otimes |1><0| 
%        + |1><0| \otimes |0><1| + |1><1| \otimes |1><1|
%      = expm(-i H)
%    H = 0.25 .* pi .* ( II - XX -YY -ZZ)  
    size1 = 4;
    size2 = 4;
    basis = matrixBasis_2qubit_pauli_unnormalized;
    
    mat = zeros(4,4);
    % II
    i1 = 1;
    i2 = 1;
    index = index_1d_from_2d(size1, size2, i1, i2);
    mat = mat + basis(index).mat;
    % XX
    i1 = 2;
    i2 = 2;
    index = index_1d_from_2d(size1, size2, i1, i2);
    mat = mat - basis(index).mat;    
    % YY
    i1 = 3;
    i2 = 3;
    index = index_1d_from_2d(size1, size2, i1, i2);
    mat = mat - basis(index).mat;
    % ZZ
    i1 = 4;
    i2 = 4;
    index = index_1d_from_2d(size1, size2, i1, i2);
    mat = mat - basis(index).mat;

    output = 0.25 .* pi .* mat;
end

