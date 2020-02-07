clear;
format long;

%% 

matA = [1,2,3,4; 5,6,7,8; 9,10,11,12]
matB = HS_hermite_from_parameterMatrix(matA)

%% 

matA = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1];
basis = matrixBasis_1qubit_pauli_normalized();
Choi_from_HS_parameterMatrix(matA, basis)