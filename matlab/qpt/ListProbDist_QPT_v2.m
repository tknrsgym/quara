function list_probDist = ListProbDist_QPT( Choi, list_state, list_povm, list_schedule )
%LISTPROBDIST_QPT Summary of this function goes here
%   Detailed explanation goes here
    num_schedule = size(list_schedule, 1);
    num_povm     = size(list_povm, 1);
    num_outcome  = size(list_povm, 2);
    
    for id = 1:num_schedule
        i_state = list_schedule(id, 1);
        j_povm  = list_schedule(id, 2);
        
        state = list_state(i_state).mat;
        
        for i_omega = 1:num_outcome
            povm(i_omega).mat  = list_povm(j_povm, i_omega).mat;
        end

        list(id).vec = ProbDist_QPT(Choi, state, povm);
    end
    
    list_probDist = list(1).vec;
    for id = 2:num_schedule
        list_probDist = vertcat(list_probDist, list(id).vec); 
    end 

end

