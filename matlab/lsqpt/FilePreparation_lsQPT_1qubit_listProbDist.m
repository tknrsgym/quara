function [] = FilePreparation_lsQPT_1qubit_listProbDist( filename, k, Choi, list_state, list_povm, list_schedule, char_precision )
%FILEPREPARATION_1QUBIT_LISTPROBDIST Summary of this function goes here
%   Detailed explanation goes here
    HS = HSpb_from_Choi_1qubit(Choi);
    HS_k = HS^k;
    Choi_k = Choi_from_HSpb_1qubit(HS_k);

    list_probDist = ListProbDist_QPT( Choi_k, list_state, list_povm, list_schedule );
    num_schedule = size(list_schedule, 1);
    list = list_probDist(1).vec;
    for id = 2:num_schedule
        list = vertcat(list, list_probDist(id).vec); 
    end    

    dlmwrite(filename, list, 'precision', char_precision, 'delimiter', ',');

end
