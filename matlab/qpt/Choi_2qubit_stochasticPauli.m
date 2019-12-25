function mat = Choi_2qubit_stochasticPauli( vec_p)
%CHOI_1QUBIT_STOCHASTICPAULI Summary of this function goes here
%   Detailed explanation goes here    
    pauli_1q(1).mat = [1.0, 0.0; 0.0, 1.0];% I
    pauli_1q(2).mat = [0.0, 1.0; 1.0, 0.0];% X
    pauli_1q(3).mat = [0.0, -1.0i; 1.0i, 0.0];% Y
    pauli_1q(4).mat = [1.0, 0.0; 0.0, -1.0];% Z
    
    for i1 = 1:4
        for i2 = 1:4
            i = 4*(i1 -1) + i2;
            pauli_2q(i).mat = kron(pauli_1q(i1).mat, pauli_1q(i2).mat);
        end
    end
    
        
    mat = zeros(16);
    for i = 1:16
        matA = transpose(pauli_2q(i).mat);
        vecA = matA(:);
        mat = mat + vec_p(i) * vecA * vecA';
    end
    
    
    
end

