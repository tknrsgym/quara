function output = hamiltonian_2qubit_gateCZ()
%HAMILTONIAN_2QUBIT_GATECZ returns the Hamiltonian corresponding to the CZ gate on 2-qubit system.
%   CX = |0><0| \otimes I + |1><1| \otimes Z
%      = expm(-i H)
%    H = 0.25 .* pi .* (II - ZI - IZ + ZZ)  
    size1 = 4;
    size2 = 4;
    basis = matrixBasis_2qubit_pauli_unnormalized;
    
    mat = zeros(4,4);
    % II
    i1 = 1;
    i2 = 1;
    index = index_1d_from_2d(size1, size2, i1, i2);
    mat = mat + basis(index).mat;
    
    % ZI
    i1 = 4;
    i2 = 1;
    index = index_1d_from_2d(size1, size2, i1, i2);
    mat = mat - basis(index).mat;    
    
    % IZ
    i1 = 1;
    i2 = 4;
    index = index_1d_from_2d(size1, size2, i1, i2);
    mat = mat - basis(index).mat;    

    % ZZ
    i1 = 4;
    i2 = 4;
    index = index_1d_from_2d(size1, size2, i1, i2);
    mat = mat + basis(index).mat;

    output = 0.25 .* pi .* mat;
end

