function [num_povm, num_outcome] = FilePreparation_2qubit_povm(filename, char_precision)
%FILEPREPARATION_STATE_1QUBIT Summary of this function goes here
%   Detailed explanation goes here
    num_povm    = 9;
    num_outcome = 4;
   
    povm_1q(1,1).mat = [0.50,  0.50;  0.50, 0.50];% X+
    povm_1q(1,2).mat = [0.50, -0.50; -0.50, 0.50];% X-
    
    povm_1q(2,1).mat = [0.50, -0.50i;  0.50i, 0.50];% Y+
    povm_1q(2,2).mat = [0.50,  0.50i; -0.50i, 0.50];% Y+
    
    povm_1q(3,1).mat = [1.0, 0.0; 0.0, 0.0];% Z+
    povm_1q(3,2).mat = [0.0, 0.0; 0.0, 1.0];% Z-

    for i1 = 1:3
        for i2 = 1:3
            j = 3*(i1 -1) + i2;
            for omega1 = 1:2
                for omega2 = 1:2
                    omega = 2*(omega1 - 1) + omega2;
                    povm_2q(j, omega).mat = kron(povm_1q(i1, omega1).mat, povm_1q(i2, omega2).mat);
                end
            end
        end
    end
    
    list = povm_2q(1,1).mat;
    for omega = 2:num_outcome
        list = vertcat(list, povm_2q(1, omega).mat);
    end
    for j = 2:num_povm
        for omega = 1:num_outcome
            list = vertcat(list, povm_2q(j,omega).mat);
        end
    end

    dlmwrite(filename, list, 'precision', char_precision, 'delimiter', ',');

end
