function output = matPauli_1qubit()
%MATPAULI_1QUBIT returns a set of Pauli matrices on 1-qubit. 
%   - (1).mat : I
%   - (2).mat : X
%   - (3).mat : Y
%   - (4).mat : Z
    pauli_1q(1).mat = [1.0, 0.0; 0.0, 1.0];% I
    pauli_1q(2).mat = [0.0, 1.0; 1.0, 0.0];% X
    pauli_1q(3).mat = [0.0, -1.0i; 1.0i, 0.0];% Y
    pauli_1q(4).mat = [1.0, 0.0; 0.0, -1.0];% Z
    
    output = pauli_1q;
end

