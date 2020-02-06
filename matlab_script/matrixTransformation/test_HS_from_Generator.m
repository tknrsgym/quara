format long;

%% HScb_from_Generator.m
% list_str = ["Z90"];%, "Y90", "Z90"];
% size_list_str = size(list_str, 2);
% for iStr = 1:size_list_str 
%     str = list_str(iStr);
%     H_gate = hamiltonian_1qubit_gate_target(str)
%     HScb_H = HScb_from_hamiltonian(H_gate)
%     expm(HScb_H)
% end


%% isOrthogonal_matrixBasis.m
% eps = 10^(-10);
% basis = matrixBasis_1qubit_comp();
% check_orthogonality = isOrthogonal_matrixBasis(basis, eps)
% 
% basis = matrixBasis_1qubit_pauli_normalized();
% check_orthogonality = isOrthogonal_matrixBasis(basis, eps)
% 
% basis = matrixBasis_1qubit_pauli_unnormalized();
% check_orthogonality = isOrthogonal_matrixBasis(basis, eps)

% 
% basis2(1).mat = [1,0;0,0];
% basis2(2).mat = [1,1;1,0];
% check_orthogonality = isOrthogonal_matrixBasis(basis2, eps)



%% isNormalized_matrixBasis.m
% eps = 10^(-10);
% basis = matrixBasis_1qubit_comp();
% check_normalized = isNormalized_matrixBasis(basis, eps)
% 
% basis = matrixBasis_1qubit_pauli_normalized();
% check_normalized = isNormalized_matrixBasis(basis, eps)
% 
% basis = matrixBasis_1qubit_pauli_unnormalized();
% check_normalized = isNormalized_matrixBasis(basis, eps)

% basis2(1).mat = [1,0;0,0];
% basis2(2).mat = [1,1;1,0];
% check_normalized = isNormalized_matrixBasis(basis2, eps)

%% isHermitian_matrixBasis.m
%eps = 10^(-10);
% basis = matrixBasis_1qubit_comp();
% check_hermiticity = isHermitian_matrixBasis(basis, eps)
% 
% basis = matrixBasis_1qubit_pauli_normalized();
% check_hermiticity = isHermitian_matrixBasis(basis, eps)
% 
% basis = matrixBasis_1qubit_pauli_unnormalized();
% check_hermiticity = isHermitian_matrixBasis(basis, eps)

% basis2(1).mat = [1,0;0,0];
% basis2(2).mat = [1,1;1,0];
% check_hermiticity = isHermitian_matrixBasis(basis2, eps)

%% isValidNo_matrixBasis.m
% basis = matrixBasis_1qubit_comp();
% check = isValidNo_matrixBasis(basis)
% 
% basis = matrixBasis_1qubit_pauli_normalized();
% check = isValidNo_matrixBasis(basis)
% 
% basis = matrixBasis_1qubit_pauli_unnormalized();
% check = isValidNo_matrixBasis(basis)

% basis2(1).mat = [1,0;0,0];
% basis2(2).mat = [1,1;1,0];
% check = isValidNo_matrixBasis(basis2)

%% matrixBasis_2qubit_pauli_unnormalized.m
% basis = matrixBasis_2qubit_pauli_unnormalized();
% eps = 10^(-10);
% check_orthogonality = isOrthogonal_matrixBasis(basis, eps)
% check_normalized = isNormalized_matrixBasis(basis, eps)
% check_hermiticity = isHermitian_matrixBasis(basis, eps)
% check_No = isValidNo_matrixBasis(basis)

% basis = matrixBasis_2qubit_pauli_normalized();
% eps = 10^(-10);
% check_orthogonality = isOrthogonal_matrixBasis(basis, eps)
% check_normalized = isNormalized_matrixBasis(basis, eps)
% check_hermiticity = isHermitian_matrixBasis(basis, eps)
% check_No = isValidNo_matrixBasis(basis)

%% index_1d_from_2d.m, index_2d_from_1d.m
% size1 = 4;
% size2 = 4;
% for i1 = 1:size1
%     for i2 = 1:size2
%         index = index_1d_from_2d(size1, size2, i1, i2);
%         [i1, i2, index]
%     end
% end

% for index = 1:size1*size2
%    [i1, i2] = index_2d_from_1d(size1, size2, index);
%    [index, i1, i2]
% end

%% matU_HS_basisTransform_to_B1_from_B2.m
% basis1 = matrixBasis_1qubit_pauli_normalized();
% basis2 = matrixBasis_1qubit_comp();
% matU = matU_HS_basisTransform_to_B1_from_B2(basis1, basis2);
% 
% list_str = ["H"];%, "Y90", "Z90"];
% size_list_str = size(list_str, 2);
% for iStr = 1:size_list_str 
%     str = list_str(iStr);
%     H_gate = hamiltonian_1qubit_gate_target(str);
%     HScb_H = HScb_from_hamiltonian(H_gate);
%     HScb_G = expm(HScb_H);
%     HSpb_G = matU * HScb_G * ctranspose(matU);
% end
% 
% Choi1 = Choi_from_HS(HSpb_G, basis1);
% Choi2 = Choi_from_HS(HScb_G, basis2);
% 
% norm(Choi1 - Choi2, 'fro');
% 
% HSpb = HS_from_Choi(Choi1, basis1);
% HScb = HS_from_Choi(Choi2, basis2);
% 
% norm(HSpb - HSpb_G)
% norm(HScb - HScb_G)

%% HScb_from_jumpOpeartor
%vecGamma = [1, 0, 0];% OK
%vecGamma = [0, 1, 0];% OK
%vecGamma = [0, 0, 2];% OK
% vecGamma = [1, 2, 2];
% list_c = jumpOperator_1qubit_model01(vecGamma);
% HScb_D = HScb_from_jumpOperator(list_c);
% HScb_G = expm(HScb_D);
% HSpb_G = matU * HScb_G * ctranspose(matU)

%% HScb_Lindbladian_from_hamiltonian_jumpOperator
% 1-qubit
% str = "T";
% H_gate = hamiltonian_1qubit_gate_target(str)
% vecGamma = [0.01, 0.02, 0.02];
% list_c = jumpOperator_1qubit_model01(vecGamma);
% 
% HScb_L = HScb_Lindbladian_from_hamiltonian_jumpOperator(H_gate, list_c)
% HScb_G = expm(HScb_L)
% 
% basis1 = matrixBasis_1qubit_pauli_normalized();
% basis2 = matrixBasis_1qubit_comp();
% matU = matU_HS_basisTransform_to_B1_from_B2(basis1, basis2);
% HSpb_G = matU * HScb_G * ctranspose(matU)
% 
% [V, D, W] = eig(HSpb_G)

% 2-qubit
str = "CZ";
H_gate = hamiltonian_2qubit_gate_target(str);
%vecGamma1 = [0.01, 0.02, 0.02];
%vecGamma2 = [0.02, 0.03, 0.01];
vecGamma1 = [0, 0, 0];
vecGamma2 = [0, 0, 0];
list_c = jumpOperator_2qubit_model01(vecGamma1, vecGamma2);

basis_pauli = matrixBasis_2qubit_pauli_normalized();
basis_comp = matrixBasis_2qubit_comp();

HS_G = HS_G_basis_from_hamiltonian_jumpOperator(H_gate, list_c, basis_comp, basis_pauli);
Choi = Choi_from_HS(HS_G, basis_pauli);

