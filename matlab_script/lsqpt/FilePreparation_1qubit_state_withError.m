function num_state = FilePreparation_1qubit_state_withError(filename, p, char_precision)
%FILEPREPARATION_STATE_1QUBIT Summary of this function goes here
%   Detailed explanation goes here
    num_state = 4;

    state(1).mat = [1.0 - p, 0.0; 0.0, p];% Z+
    state(2).mat = [p, 0.0; 0.0, 1.0 - p];% Z-
    state(3).mat = [0.50, 0.50; 0.50, 0.50];% X+
    state(4).mat = [0.50, -0.50i; 0.50i, 0.50];% Y+

    list = vertcat(state(1).mat, state(2).mat, state(3).mat, state(4).mat);   

    dlmwrite(filename, list, 'precision', char_precision, 'delimiter', ',');

end