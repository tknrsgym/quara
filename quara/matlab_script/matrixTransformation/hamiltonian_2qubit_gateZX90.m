function output = hamiltonian_2qubit_gateZX90()
%HAMILTONIAN_2QUBIT_GATEZX90 returns the Hamiltonian corresponding to the ZX90 gate on 2-qubit system.
%    H = 0.25 .* pi .* ZX  
    basis = matrixBasis_2qubit_pauli_unnormalized;
    
    size1 = 4;
    size2 = 4;
    i1 = 4;
    i2 = 2;
    index = index_1d_from_2d(size1, size2, i1, i2);
    mat = basis(index).mat;
    output = 0.25 .* pi .* mat;
end

