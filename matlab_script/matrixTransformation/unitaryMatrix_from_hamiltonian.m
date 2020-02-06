function output = unitaryMatrix_from_hamiltonian(matH)
%UNITARYMATRIX_FROM_HAMILTONIAN returns the unitary matrix corresponding to the Hamiltonian. 
%   U = expm(-i matH)
    output = expm(-1i .* matH);
end

