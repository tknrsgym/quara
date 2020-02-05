function num_state = FilePreparation_2qubit_state(filename, char_precision)
%FILEPREPARATION_STATE_1QUBIT 
    num_state = 16;

    state_1q(1).mat = [1.0, 0.0; 0.0, 0.0];% Z+
    state_1q(2).mat = [0.0, 0.0; 0.0, 1.0];% Z-
    state_1q(3).mat = [0.50, 0.50; 0.50, 0.50];% X+
    state_1q(4).mat = [0.50, -0.50i; 0.50i, 0.50];% Y+

    for i1 = 1:4
        for i2 = 1:4
            j = 4*(i1 -1) + i2;
            state_2q(j).mat = kron(state_1q(i1).mat, state_1q(i2).mat);
        end
    end
    
    list = state_2q(1).mat;
    for j = 2:16
        list = vertcat(list, state_2q(j).mat);   

    dlmwrite(filename, list, 'precision', char_precision, 'delimiter', ',');

end