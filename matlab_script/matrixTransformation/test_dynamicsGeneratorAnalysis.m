format long;

%% isSquare_matrixBasis.m
% basis = matrixBasis_1qubit_comp();
% isSquare_matrixBasis(basis)
% 
% basis2(1).mat = [1, 2, 3; 4, 5, 6];
% basis2(2).mat = [7, 8, 9; 10, 11, 12];
% isSquare_matrixBasis(basis2)

%% isEqualElementSize_matrixBasis.m
% basis = matrixBasis_1qubit_comp();
% isEqualElementSize_matrixBasis(basis)
% 
% basis2(1).mat = [1, 2, 3; 4, 5, 6];
% basis2(2).mat = [7, 8, 9; 10, 11, 12];
% isEqualElementSize_matrixBasis(basis2)
% 
% basis3(1).mat = [1, 2, 3; 4, 5, 6];
% basis3(2).mat = [7, 8; 10, 11];
% isEqualElementSize_matrixBasis(basis3)

%% is1stElementProportionalToIdentity_matrixBasis.m
eps = 10^(-10);

% basis = matrixBasis_1qubit_comp();
% is1stElementProportionalToIdentity_matrixBasis(basis, eps)

% basis = matrixBasis_2qubit_comp();
% is1stElementProportionalToIdentity_matrixBasis(basis, eps)

% basis = matrixBasis_1qubit_pauli_normalized();
% is1stElementProportionalToIdentity_matrixBasis(basis, eps)

% basis = matrixBasis_1qubit_pauli_unnormalized();
% is1stElementProportionalToIdentity_matrixBasis(basis, eps)

% basis = matrixBasis_2qubit_pauli_normalized();
% is1stElementProportionalToIdentity_matrixBasis(basis, eps)

% basis = matrixBasis_2qubit_pauli_unnormalized();
% is1stElementProportionalToIdentity_matrixBasis(basis, eps)

% basis = matrixBasis_2qubit_comp();
% is1stElementProportionalToIdentity_matrixBasis(basis, eps)

%% isTraceless_matrixList.m
% eps = 10^(-10);

% list = matrixBasis_1qubit_comp();
% isTraceless_matrixList(list, eps)

% list = matrixBasis_2qubit_comp();
% isTraceless_matrixList(list, eps)

% list = matrixBasis_1qubit_pauli_normalized();
% isTraceless_matrixList(list, eps)

% vecGamma = [0.01, 0.02, 0.02];
% list_c = jumpOperator_1qubit_model01(vecGamma);
% isTraceless_matrixList(list_c, eps)

% vecGamma1 = [0.01, 0.02, 0.02];
% vecGamma2 = [0.1, 0.2, 0.2];
% list_c = jumpOperator_2qubit_model01(vecGamma1, vecGamma2);
% isTraceless_matrixList(list_c, eps)

%% isHermitian_matrix.m
% eps = 10^(-10);
% matA = [1, i, 0; -i, 2, 3; 0, 3, -4];
% isHermitian_matrix(matA, eps)


%% hamiltonian_from_vectorAndBasis.m

% 1-qubit 
% basis = matrixBasis_1qubit_comp();
% vecH = [1, 2, 3, 4];
% matH = matrix_from_coefficients_basis(vecH, basis)
% 
% basis = matrixBasis_1qubit_pauli_normalized();
% vecH = [1, 2, 3, 4];
% matH = matrix_from_coefficients_basis(vecH, basis)
% matrixCoefficients_from_mat_basis(matH, basis)

% 2-qubit
% basis = matrixBasis_2qubit_comp();
% vecH = [1, -2, 3, 4, 5, 6, -7, 8 + 8i, 9, 10, 11, 12 - i, 13, 14, 15, -16];
% matH = matrix_from_coefficients_basis(vecH, basis);
% matrixCoefficients_from_mat_basis(matH, basis)

% basis = matrixBasis_2qubit_pauli_normalized();
% vecH = [1, -2, 3, 4, 5, 6, -7, 8 + 8i, 9, 10, 11, 12 - i, 13, 14, 15, -16];
% matH = matrix_from_coefficients_basis(vecH, basis);
% matrixCoefficients_from_mat_basis(matH, basis)



%% matK_from_jumpOperator.m, jumpOperator_from_matK.m
% 1-qubit
% vecGamma = [2, 4, 2];
% list_c = jumpOperator_1qubit_model01(vecGamma);
% basis_pauli = matrixBasis_1qubit_pauli_normalized();
% matK = matK_from_jumpOperator(list_c, basis_pauli);
% list_c2 = jumpOperator_from_matK(matK, basis_pauli);
% c2No = size(list_c2, 2);
% for ic = 1:c2No
%     disp(list_c2(ic).mat); 
% end
% HS_1 = HScb_from_jumpOperator(list_c);
% HS_2 = HScb_from_jumpOperator(list_c2);
% norm(HS_1 - HS_2, 'fro')

% 2-qubit
% vecGamma1 = [2, 4, 2];
% vecGamma2 = [3, 4, 9];
% list_c = jumpOperator_2qubit_model01(vecGamma1, vecGamma2);
% basis_pauli = matrixBasis_2qubit_pauli_normalized();
% matK = matK_from_jumpOperator(list_c, basis_pauli);
% list_c2 = jumpOperator_from_matK(matK, basis_pauli);
% 
% HS_1 = HScb_from_jumpOperator(list_c);
% HS_2 = HScb_from_jumpOperator(list_c2);
% norm(HS_1 - HS_2, 'fro')

%% hamiltonianCoefficients_from_HS_comp_Lindblad.m
% 1-qubit
% vecCoef = [1, 2, -3, 4];
% basis1 = matrixBasis_1qubit_pauli_normalized();
% matH = matrix_from_coefficients_basis(vecCoef, basis1);
% HScb_H = HScb_from_hamiltonian(matH);
% basis2 = matrixBasis_1qubit_pauli_normalized();
% vecH = hamiltonianCoefficients_from_HS_comp_Lindbladian(HScb_H, basis2)

% 2-qubit
% vecCoef = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, 12, 13, 14, 15, 16];
% basis1 = matrixBasis_2qubit_pauli_normalized();
% matH = matrix_from_coefficients_basis(vecCoef, basis1);
% HScb_H = HScb_from_hamiltonian(matH);
% basis2 = matrixBasis_2qubit_pauli_normalized();
% vecH = hamiltonianCoefficients_from_HS_comp_Lindbladian(HScb_H, basis2)

%% matJCoefficients_from_HS_comp_Lindbladian.m
% 1-qubit
% vecGamma = [2, 4, 2];
% list_c = jumpOperator_1qubit_model01(vecGamma);
% basis_pauli = matrixBasis_1qubit_pauli_normalized();
% HScb_D = HScb_from_jumpOperator(list_c)
% vecJ = matJCoefficients_from_HS_comp_Lindbladian(HScb_D, basis_pauli)
% matJ1 = matrix_from_coefficients_basis(vecJ, basis_pauli);
% matJ2 = matJ_from_jumpOperator_TPcondition(list_c);
% norm(matJ1 - matJ2, 'fro')

% 2-qubit
% vecGamma1 = [2, 1, 1];
% vecGamma2 = [1, 3, 5];
% list_c = jumpOperator_2qubit_model01(vecGamma1, vecGamma2);
% basis_pauli = matrixBasis_2qubit_pauli_normalized();
% HScb_D = HScb_from_jumpOperator(list_c)
% vecJ = matJCoefficients_from_HS_comp_Lindbladian(HScb_D, basis_pauli)
% matJ1 = matrix_from_coefficients_basis(vecJ, basis_pauli);
% matJ2 = matJ_from_jumpOperator_TPcondition(list_c);
% 
% norm(matJ1 - matJ2, 'fro')

%% matK_from_HS_comp_Lindbladian.m
% 1-qubit
% vecGamma = [2, 4, 2];
% list_c = jumpOperator_1qubit_model01(vecGamma);
% basis_pauli = matrixBasis_1qubit_pauli_normalized();
% HScb_D = HScb_from_jumpOperator(list_c);
% matK1 = matK_from_HS_comp_Lindbladian(HScb_D, basis_pauli);
% matK2 = matK_from_jumpOperator(list_c, basis_pauli);
% norm(matK1 - matK2, 'fro')

% 2-qubit
% vecGamma1 = [2, 4, 2];
% vecGamma2 = [1, 2, 5];
% list_c = jumpOperator_2qubit_model01(vecGamma1, vecGamma2);
% basis_pauli = matrixBasis_2qubit_pauli_normalized();
% HScb_D = HScb_from_jumpOperator(list_c);
% matK1 = matK_from_HS_comp_Lindbladian(HScb_D, basis_pauli);
% matK2 = matK_from_jumpOperator(list_c, basis_pauli);
% norm(matK1 - matK2, 'fro')


%% matrix_traceless

% mat1 = [1, 2; 3, 4];
% mat2 = matrix_toTraceless(mat1);

% mat1 = [1, 2, 3; 4, 5, 6; 7, 8, 9];
% mat2 = matrix_toTraceless(mat1)


%% dynamicsGenerator_from_HS_comp_Lindbladian.m

% 1-qubit
% vecCoef = [1, 2, -3, 4];
% basis = matrixBasis_1qubit_pauli_normalized();
% matH = matrix_from_coefficients_basis(vecCoef, basis);
% 
% vecGamma = [2, 4, 2];
% list_c = jumpOperator_1qubit_model01(vecGamma);
% 
% HS_L = HScb_Lindbladian_from_hamiltonian_jumpOperator(matH, list_c);
% 
% [matH2, matJ2, matK2] = dynamicsGenerator_from_HS_comp_Lindbladian(HS_L, basis);
% 
% matH1 = matrix_toTraceless(matH);
% matJ = matJ_from_jumpOperator_TPcondition(list_c);
% matK = matK_from_jumpOperator(list_c, basis);
% 
% diff_H = norm(matH1 - matH2, 'fro')
% diff_J = norm(matJ - matJ2, 'fro')
% diff_K = norm(matK - matK2, 'fro')

% 2-qubit
% vecCoef = [1, 2, -3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
% basis = matrixBasis_2qubit_pauli_normalized();
% matH = matrix_from_coefficients_basis(vecCoef, basis);
% 
% vecGamma1 = [2, 4, 2];
% vecGamma2 = [1, 3, 5];
% list_c = jumpOperator_2qubit_model01(vecGamma1, vecGamma2);
% 
% HS_L = HScb_Lindbladian_from_hamiltonian_jumpOperator(matH, list_c);
% 
% [matH2, matJ2, matK2] = dynamicsGenerator_from_HS_comp_Lindbladian(HS_L, basis);
% 
% matH1 = matrix_toTraceless(matH);
% matJ = matJ_from_jumpOperator_TPcondition(list_c);
% matK = matK_from_jumpOperator(list_c, basis);
% 
% diff_H = norm(matH1 - matH2, 'fro')
% diff_J = norm(matJ - matJ2, 'fro')
% diff_K = norm(matK - matK2, 'fro')