function output = hamiltonian_2qubit_gateCX()
%HAMILTONIAN_2QUBIT_GATECX returns the Hamiltonian corresponding to the CX gate on 2-qubit system.
%   CX = |0><0| \otimes I + |1><1| \otimes X
%      = expm(-i H)
%    H = 0.25 .* pi .* (II - ZI - IX + ZX)  
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
    % IX
    i1 = 1;
    i2 = 2;
    index = index_1d_from_2d(size1, size2, i1, i2);
    mat = mat - basis(index).mat;    
    % ZX
    i1 = 4;
    i2 = 2;
    index = index_1d_from_2d(size1, size2, i1, i2);
    mat = mat + basis(index).mat;    
    
    output = 0.25 .* pi .* mat;
end

