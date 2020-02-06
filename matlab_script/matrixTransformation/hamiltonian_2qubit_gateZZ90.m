function output = hamiltonian_2qubit_gateZZ90()
%HAMILTONIAN_2QUBIT_GATEZZ90 returns the Hamiltonian corresponding to the ZZ90 gate on 2-qubit system.
%    H = 0.25 .* pi .* ZZ  
    size1 = 4;
    size2 = 4;
    i1 = 4;
    i2 = 4;
    index = index_1d_from_2d(size1, size2, i1, i2);
    basis = matrixBasis_2qubit_pauli_unnormalized;
    mat = basis(index).mat;
    output = 0.25 .* pi .* mat;
end

