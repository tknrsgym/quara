clear;
format long;
char_precision = '%.15e';

matA = [0, -1i;1i,0]

filename_matA = '../ImportFiles/matA_1qubit_X90.tcsv';
FilePreparation_matrix(matA, filename_matA, char_precision);

matB = csvread(filename_matA)