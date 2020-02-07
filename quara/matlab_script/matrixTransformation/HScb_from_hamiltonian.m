function output = HScb_from_hamiltonian(matH)
%HSCB_FROM_HAMILTONIAN returns the Hilbert-Schmidt representation of the 
%Hamiltonain part in the Lindbladian
%   HScb(matH) = - i ( H \otimes I - I \otimes conj(H))
    [sizeRow, sizeCol] = size(matH);
    assert(sizeRow == sizeCol, "Matrix size of the input to HScb_from_hamiltonian() is invalid!");
    eps = 10^(-10);
    assert(isHermitian_matrix(matH, eps));
    matI = eye(sizeRow, sizeCol);
    mat1 = kron(matH, matI);
    mat2 = kron(matI, conj(matH));
    HS = -1i .* mat1 + 1i .* mat2;
    output = HS;
end

