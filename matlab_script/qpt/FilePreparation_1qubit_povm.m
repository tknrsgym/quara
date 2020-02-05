function [num_povm, num_outcome] = FilePreparation_1qubit_povm(filename, char_precision)
%FILEPREPARATION_STATE_1QUBIT 
    num_povm    = 3;
    num_outcome = 2;
   
    povm(1,1).mat = [0.50,  0.50;  0.50, 0.50];% X+
    povm(1,2).mat = [0.50, -0.50; -0.50, 0.50];% X-
    
    povm(2,1).mat = [0.50, -0.50i;  0.50i, 0.50];% Y+
    povm(2,2).mat = [0.50,  0.50i; -0.50i, 0.50];% Y+
    
    povm(3,1).mat = [1.0, 0.0; 0.0, 0.0];% Z+
    povm(3,2).mat = [0.0, 0.0; 0.0, 1.0];% Z-

    list = vertcat(povm(1,1).mat, povm(1,2).mat,povm(2,1).mat, povm(2,2).mat, povm(3,1).mat, povm(3,2).mat);

    dlmwrite(filename, list, 'precision', char_precision, 'delimiter', ',');

end
