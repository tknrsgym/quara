function num_state = FilePreparation_matrix(matA, filename, char_precision)
%FILEPREPARATION_MATRIX 
    dlmwrite(filename, matA, 'precision', char_precision, 'delimiter', ',');
end