format long;

%% Computational basis
% Checked for dim = 2, 3, 4.
% dim = 4;
% basis = matrixBasis_d_comp(dim);
% basisNo = size(basis, 2);
% for iBasis = 1:basisNo
%     basis(iBasis).mat
% end


%% 1-qubit, Computational basis
% basis = matrixBasis_1qubit_comp();
% basisNo = size(basis, 2);
% for iBasis = 1:basisNo
%     basis(iBasis).mat
% end

%% 1-qubit, Pauli basis

% % Unnormalized
% basis = matrixBasis_1qubit_pauli_unnormalized();
% basisNo = size(basis, 2);
% for iBasis = 1:basisNo
%     basis(iBasis).mat
% end
% 
% % Normalized
% basis = matrixBasis_1qubit_pauli_normalized();
% basisNo = size(basis, 2);
% for iBasis = 1:basisNo
%     basis(iBasis).mat
% end

%% 2-qubit, Computational basis
% basis = matrixBasis_2qubit_comp();
% basisNo = size(basis, 2);
% for iBasis = 1:basisNo
%     basis(iBasis).mat
% end

%% 2-qubit, unnormalized Pauli basis
% Unnormalized
% basis = matrixBasis_2qubit_pauli_unnormalized();
% basisNo = size(basis, 2);
% for iBasis = 1:basisNo
%     basis(iBasis).mat
% end

% Normalized
basis = matrixBasis_2qubit_pauli_normalized();
basisNo = size(basis, 2);
for i1 = 1:basisNo
    for i2 = 1:basisNo
        basis(i1, i2).mat
    end
end