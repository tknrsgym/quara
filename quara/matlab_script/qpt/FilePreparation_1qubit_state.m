function num_state = FilePreparation_1qubit_state(filename, char_precision)
%FILEPREPARATION_STATE_1QUBIT 
    num_state = 4;

    state(1).mat = [1.0, 0.0; 0.0, 0.0];% Z+
    state(2).mat = [0.0, 0.0; 0.0, 1.0];% Z-
    state(3).mat = [0.50, 0.50; 0.50, 0.50];% X+
    state(4).mat = [0.50, -0.50i; 0.50i, 0.50];% Y+

    list = vertcat(state(1).mat, state(2).mat, state(3).mat, state(4).mat);   

    dlmwrite(filename, list, 'precision', char_precision, 'delimiter', ',');

end