format long
char_precision = '%.15e';

p=0.10;
filename_state = './ImportFiles/tester_1qubit_state_withError.csv';
num_state = FilePreparation_1qubit_state_withError(filename_state, p, char_precision);

