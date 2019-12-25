function [] = FilePreparation_2qubit_listProbDist( filename, Choi, list_state, list_povm, list_schedule, char_precision )
%FILEPREPARATION_1QUBIT_LISTPROBDIST Summary of this function goes here
%   Detailed explanation goes here
    list_probDist = ListProbDist_QPT( Choi, list_state, list_povm, list_schedule );
    num_schedule = size(list_schedule, 1);
    list = list_probDist(1).vec;
    for id = 2:num_schedule
        list = vertcat(list, list_probDist(id).vec); 
    end    

    dlmwrite(filename, list, 'precision', char_precision, 'delimiter', ',');

end
